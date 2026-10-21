#!/usr/bin/env python3
"""Text-only exported W4A16 checkpoint (Qwen3_5ForCausalLM, keys `model.*`) -> vLLM layout
(Qwen3_5ForConditionalGeneration, keys `model.language_model.*`, no visual weights).

  60_to_vllm_layout.py --src artifacts/export/<tag> --template <existing *_vllm_mapped dir> --out artifacts/vllm/<tag>
Tensor data are not modified (only renamed); config comes from the template (top-level VLM config + text_config + quantization_config).
"""
import argparse, json, os, shutil, struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

os.umask(0)
ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True); ap.add_argument("--template", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--extra-from", type=Path, default=Path("/home/dungvpt/axolotl_planning_v2/outputs/merged/qwen35_4b_v7_r64_untied"), help="dir with processor_config.json (vLLM loads an image processor for the VLM architecture)")
a = ap.parse_args()
t = {}
with safe_open(a.src / "model.safetensors", "pt", device="cpu") as f:
    for k in f.keys():
        t[("model.language_model." + k[len("model."):]) if k.startswith("model.") else k] = f.get_tensor(k)
assert "lm_head.weight" in t and "model.language_model.embed_tokens.weight" in t
a.out.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in t.items()}, a.out / "model.safetensors", metadata={"format": "pt"})
cfg_t = json.loads((a.template / "config.json").read_text()); cfg_s = json.loads((a.src / "config.json").read_text())
assert cfg_t["quantization_config"]["config_groups"] == cfg_s["quantization_config"]["config_groups"], "quant scheme differs from template"
(a.out / "config.json").write_text(json.dumps(cfg_t, indent=2))
for n in ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"):
    for d in (a.src, a.template, a.extra_from):
        if (d / n).exists():
            shutil.copy2((d / n).resolve(), a.out / n); break
os.system(f"chmod -R a+rwX {a.out}")
print("wrote", a.out, len(t), "tensors")
