#!/usr/bin/env python3
"""Check that an exported W4A16 checkpoint IS the model that was trained (bit-exact), and that it is structurally servable.
  31_verify_export.py --dense-dir <axolotl ckpt> --init-dir <student_init dir with qad_scales.safetensors> --export-dir <W4A16 dir> [--n 12]
For N sampled quantised modules the training-time effective weight is recomputed from the dense checkpoint and compared with the
dequantised export:  full-FT / fixed scale -> Q(W, s) ; learned scale -> Q(W, bf16(s*exp(logs))) ; internal LoRA -> Q(W + (a/r)BA, s)."""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axolotl.integrations.qad.fixed_scale_fq import fake_quant  # noqa: E402
from axolotl.integrations.qad.ptq_io import load_ptq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dense-dir", type=Path, required=True); ap.add_argument("--init-dir", type=Path, required=True); ap.add_argument("--export-dir", type=Path, required=True); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args(); bad = []
def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg); 
    if not cond: bad.append(msg)

cfg = json.loads((a.export_dir / "config.json").read_text()); q = cfg.get("quantization_config", {})
print("[structure]")
check(bool(q) and q.get("format") == "pack-quantized", f"quantization_config format = {q.get('format')}")
w = list(q.get("config_groups", {}).values())[0]["weights"] if q else {}
check(w.get("num_bits") == 4 and w.get("group_size") == 128 and w.get("symmetric") is True, f"weights scheme 4-bit g{w.get('group_size')} symmetric={w.get('symmetric')}")
ign = " ".join(q.get("ignore", []))
check("lm_head" in ign and "in_proj_a" in ign and "in_proj_b" in ign, "ignore list covers lm_head / in_proj_a / in_proj_b")
check(cfg.get("tie_word_embeddings") is False and cfg.get("architectures") == ["Qwen3_5ForCausalLM"], f"tie={cfg.get('tie_word_embeddings')} arch={cfg.get('architectures')}")
for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
    check((a.export_dir / f).exists(), f"{f} present")
ex = safe_open(a.export_dir / "model.safetensors", "pt", device="cpu"); keys = set(ex.keys())
check("lm_head.weight" in keys and "lm_head.weight_packed" not in keys, "lm_head stored dense BF16 (not quantised)")
check(not any(k.endswith(".lora_A") or k.endswith(".fq_logs") for k in keys), "no adapter / training-only tensors leaked into the export")
scales = load_file(a.init_dir / "qad_scales.safetensors"); gs = json.loads((a.init_dir / "qad_meta.json").read_text())["group_size"]
check(sum(k.endswith(".weight_packed") for k in keys) == len(scales), f"{sum(k.endswith('.weight_packed') for k in keys)} packed tensors == {len(scales)} expected modules")

print("[exactness]")
dense_f = sorted(a.dense_dir.glob("*.safetensors")); D = {}
for f in dense_f:
    with safe_open(f, "pt", device="cpu") as h:
        for k in h.keys(): D[k] = h.get_tensor(k)
exd, _, _ = load_ptq(a.export_dir)
names = sorted(scales)[:: max(1, len(scales) // a.n)][: a.n]
for n in names:
    W = D[f"{n}.weight"]; s = scales[n]
    kind = "full/fixed"
    if f"{n}.lora_A" in D:
        kind = "internal-LoRA"; W = (W.float() + (D[f"{n}.lora_B"].float() @ D[f"{n}.lora_A"].float()) * float(D[f"{n}.lora_scale"]))
    elif f"{n}.fq_logs" in D:
        kind = "learned-scale"; s = (s.float() * torch.exp(D[f"{n}.fq_logs"].float())).to(torch.bfloat16)
    eff = fake_quant(W if W.dtype == torch.float32 else W, s, gs).to(torch.bfloat16)
    check(torch.equal(eff, exd[f"{n}.weight"]), f"{n} [{kind}]: dequantised export == training-time effective weight")
for k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"):
    check(torch.equal(D[k], exd[k]), f"{k} identical to the dense checkpoint")
print("RESULT:", "ALL CHECKS PASSED" if not bad else f"{len(bad)} FAILED"); sys.exit(1 if bad else 0)
