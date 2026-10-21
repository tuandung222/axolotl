#!/usr/bin/env python3
"""RTN student init from a BF16 (un-finetuned) base: dense BF16 weights UNCHANGED + per-group scales = max|w|/7.5 (bf16).

  00b_prepare_init_rtn.py --base-dir outputs/merged/qwen35_4b_base_untied --like artifacts/student_init_4b_r64 --out-dir artifacts/student_init_4b_base_rtn

`--like` supplies the list of modules to quantise (the same 200 Linear modules as the v7 PTQ: in_proj_a/b, lm_head, embed stay BF16).
Weights are NOT snapped to the grid: the fake-quant forward applies RTN on the fly, and export rounds the trained weights."""
import argparse, json, os, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.umask(0)
ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", type=Path, required=True); ap.add_argument("--like", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--group-size", type=int, default=128)
a = ap.parse_args()
names = list(safe_open(a.like / "qad_scales.safetensors", "pt").keys())
base = {}
with safe_open(a.base_dir / "model.safetensors", "pt", device="cpu") as f:
    for k in f.keys(): base[k] = f.get_tensor(k)
assert "lm_head.weight" in base and not torch.equal(base["lm_head.weight"], torch.zeros(1)), "base must be untied text-only"
scales, qmax = {}, 7.5
for n in names:
    w = base[f"{n}.weight"]; assert w.dtype == torch.bfloat16 and w.shape[1] % a.group_size == 0, n
    g = w.float().view(w.shape[0], -1, a.group_size)
    scales[n] = (g.abs().amax(-1) / qmax).clamp_min(1e-8).to(torch.bfloat16)
print(f"RTN scales for {len(scales)} modules; median scale {torch.cat([s.float().flatten() for s in list(scales.values())[::10]]).median():.3e}")
a.out_dir.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in base.items()}, a.out_dir / "model.safetensors", metadata={"format": "pt"})
save_file({k: v.contiguous() for k, v in scales.items()}, a.out_dir / "qad_scales.safetensors", metadata={"format": "pt"})
(a.out_dir / "qad_meta.json").write_text(json.dumps({"group_size": a.group_size, "init": "RTN", "base_dir": str(a.base_dir), "n_quantised": len(scales)}, indent=1))
for f in a.base_dir.iterdir():
    if f.name != "model.safetensors": shutil.copy2(f, a.out_dir / f.name)
os.system(f"chmod -R a+rwX {a.out_dir}"); print("wrote", a.out_dir)
