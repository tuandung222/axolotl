#!/usr/bin/env python3
"""PTQ checkpoint -> QAD student init: dense bf16 text-only model (untied lm_head) + fixed PTQ scales.

  00_prepare_init.py --ptq-dir <gptq ckpt> --untied-dir <bf16 untied teacher dir> --out-dir artifacts/student_init_4b_r64

Checks: (1) fake_quant(W_dequant, scale) == W_dequant exactly (grid parity); (2) lm_head / embed identical to the
untied BF16 source (they are never quantised); (3) every PTQ-quantised tensor is present in the student.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axolotl.integrations.qad.fixed_scale_fq import fake_quant  # noqa: E402
from axolotl.integrations.qad.ptq_io import load_ptq  # noqa: E402

os.umask(0)
ap = argparse.ArgumentParser()
ap.add_argument("--ptq-dir", type=Path, required=True)
ap.add_argument("--untied-dir", type=Path, required=True)
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

dense, scales, gs = load_ptq(a.ptq_dir, a.device)
untied = {}
with safe_open(a.untied_dir / "model.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        untied[k] = f.get_tensor(k)

# (3) key coverage
missing = [k for k in untied if k not in dense and k != "lm_head.weight"]
extra = [k for k in dense if k not in untied]
assert not missing and not extra, f"key mismatch missing={missing[:3]} extra={extra[:3]}"

# (2) never-quantised tensors must equal the BF16 source
if "lm_head.weight" not in dense:  # 4B PTQ ckpts are tied: lm_head == embed_tokens
    assert torch.equal(untied["lm_head.weight"], dense["model.embed_tokens.weight"]), "untied lm_head != PTQ embed"
    dense["lm_head.weight"] = untied["lm_head.weight"].clone()
for k in ("lm_head.weight", "model.embed_tokens.weight"):
    assert dense[k].dtype == torch.bfloat16 and torch.equal(dense[k], untied[k]), f"{k} differs from BF16 source"

# (1) grid parity on every quantised tensor
worst = 0.0
for name, s in scales.items():
    w = dense[f"{name}.weight"]
    err = (fake_quant(w, s, gs) - w).abs().max().item()
    worst = max(worst, err)
    assert err == 0.0, f"{name}: fake_quant(W_ptq) != W_ptq (max err {err})"
print(f"grid parity OK on {len(scales)} tensors (max err {worst}), group_size={gs}")

a.out_dir.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in dense.items()}, a.out_dir / "model.safetensors", metadata={"format": "pt"})
save_file({k: v.contiguous() for k, v in scales.items()}, a.out_dir / "qad_scales.safetensors", metadata={"format": "pt"})
(a.out_dir / "qad_meta.json").write_text(json.dumps({"group_size": gs, "ptq_dir": str(a.ptq_dir), "n_quantised": len(scales)}, indent=1))
for f in a.untied_dir.iterdir():  # config (tie=False), tokenizer, chat template
    if f.name != "model.safetensors":
        shutil.copy2(f, a.out_dir / f.name)
os.system(f"chmod -R a+rwX {a.out_dir}")
print("wrote", a.out_dir)
