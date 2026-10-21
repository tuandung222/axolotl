#!/usr/bin/env python3
"""Trained dense BF16 student -> compressed-tensors W4A16 g128 on the SAME grid (scales from the PTQ init).

  30_export_w4a16.py --dense-dir <axolotl output> --init-dir artifacts/student_init_4b_r64 --ptq-dir <orig ptq> --out-dir <dir>
lm_head / embed_tokens are exported untouched (frozen during QAD). The PTQ quantization_config is reused verbatim.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compressed_tensors.compressors.pack_quantized import pack_to_int32  # noqa: E402
from axolotl.integrations.qad.fixed_scale_fq import QMAX, QMIN  # noqa: E402

os.umask(0)
ap = argparse.ArgumentParser()
ap.add_argument("--dense-dir", type=Path, required=True)
ap.add_argument("--init-dir", type=Path, required=True)
ap.add_argument("--ptq-dir", type=Path, required=True)
ap.add_argument("--out-dir", type=Path, required=True)
a = ap.parse_args()

scales = load_file(a.init_dir / "qad_scales.safetensors")
gs = json.loads((a.init_dir / "qad_meta.json").read_text())["group_size"]
dense = {}
for shard in sorted(a.dense_dir.glob("*.safetensors")):
    with safe_open(shard, framework="pt", device="cpu") as f:
        for k in f.keys():
            dense[k] = f.get_tensor(k)

out = {}
logs = {k[: -len(".fq_logs")]: v for k, v in dense.items() if k.endswith(".fq_logs")}  # learned-scale runs
if logs:
    scales = {n: (scales[n].float() * torch.exp(logs[n].float())).to(torch.bfloat16) if n in logs else scales[n] for n in scales}
    print(f"learned scales applied to {len(logs)} modules (scale = init * exp(fq_logs), rounded to bf16 as stored)")
for k, v in dense.items():
    if k.endswith((".fq_logs", ".lora_A", ".lora_B", ".lora_scale")): continue
    name = k[: -len(".weight")] if k.endswith(".weight") else None
    if name in scales:
        s = scales[name].float().unsqueeze(-1)
        vf = v.float()
        if f"{name}.lora_A" in dense:  # internal LoRA: merge the delta, then round on the fixed grid (same ops as the training forward)
            vf = vf + (dense[f"{name}.lora_B"].float() @ dense[f"{name}.lora_A"].float()) * float(dense[f"{name}.lora_scale"])
        w = vf.view(v.shape[0], -1, gs)
        q = torch.clamp(torch.round(w / s.clamp_min(1e-30)), QMIN, QMAX).to(torch.int8).view(v.shape)
        out[f"{name}.weight_packed"] = pack_to_int32(q, 4)
        out[f"{name}.weight_scale"] = scales[name].to(torch.bfloat16)
        out[f"{name}.weight_shape"] = torch.tensor(v.shape, dtype=torch.int64)
    else:
        out[k] = v
ptq_cfg = json.loads((a.ptq_dir / "config.json").read_text())
cfg = json.loads((a.dense_dir / "config.json").read_text())
cfg["quantization_config"] = ptq_cfg["quantization_config"]
cfg["tie_word_embeddings"] = False  # lm_head exported explicitly (== embed for 4B, but never rely on tie)
a.out_dir.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, a.out_dir / "model.safetensors", metadata={"format": "pt"})
(a.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
for f in a.dense_dir.iterdir():
    if f.suffix in (".json", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(f, a.out_dir / f.name)
os.system(f"chmod -R a+rwX {a.out_dir}")
print("exported", len([k for k in out if k.endswith('.weight_packed')]), "packed tensors ->", a.out_dir)
