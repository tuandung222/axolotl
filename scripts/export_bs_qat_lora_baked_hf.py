#!/usr/bin/env python
"""Merge/export a BS QAT LoRA checkpoint and save a baked HF model.

This script performs the HF-side part of the GGUF export pipeline:

1. Load the full-precision HF base model.
2. Load either a PEFT external LoRA adapter or KH internal-QAT LoRA
   tensors.
3. Replace the configured target ``nn.Linear`` modules with KH
   ``BitLinear`` fake-quant wrappers.
4. Merge PEFT LoRA deltas into the ``BitLinear.weight`` shadow weights.
5. Bake every ``BitLinear`` into a plain ``nn.Linear`` whose weights equal the
   dequantized Q2_0 grid weights.
6. Save a normal HF directory for llama.cpp conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axolotl.integrations.bs_qat.bitnet_qat.bitlinear import BitLinear  # noqa: E402
from axolotl.integrations.bs_qat.bitnet_qat.export.bake import bake_bitlinear_to_linear  # noqa: E402
from axolotl.integrations.bs_qat.bitnet_qat.internal_lora import (  # noqa: E402
    load_full_checkpoint_state_dict,
    load_internal_lora,
)
from axolotl.integrations.bs_qat.plugin import _replace_target_linears  # noqa: E402


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="prism-ml/Ternary-Bonsai-4B-unpacked")
    parser.add_argument(
        "--checkpoint",
        help="PEFT adapter dir or internal-QAT checkpoint/adapter dir. "
        "Required unless --lora-mode none.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lora-mode",
        choices=("external_peft", "internal_qat", "full_weight", "none"),
        default="external_peft",
        help="full_weight: load a full-parameter QAT Trainer checkpoint (no LoRA "
        "at all) directly into BitLinear(use_lora=False) -- nothing to merge. "
        "none: bake the fake-quantized base model as-is, no LoRA/adapter/checkpoint "
        "(e.g. for a base-vs-finetuned Q2_0 GGUF comparison).",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qat-mode", default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-names", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--skip-names", nargs="*", default=["lm_head"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_full_or_internal_lora_state(model: torch.nn.Module, checkpoint: Path) -> dict:
    """Load an internal-QAT checkpoint into an already patched model.

    There are two expected internal-QAT artifacts:

    * ``internal_qat_lora/bs_internal_qat_lora.safetensors`` sidecars
      written by the plugin callback. These are small and preferred for export.
    * Full Trainer model shards from a non-PEFT run. These are accepted as a
      fallback so interrupted experiments remain exportable, and are also what
      ``full_weight`` mode checkpoints are (no sidecar is ever written for
      full_weight, since it has no LoRA tensors to save separately).
    """
    internal_adapter = checkpoint / "bs_internal_qat_lora.safetensors"
    if internal_adapter.exists():
        return load_internal_lora(model, checkpoint, strict=True)
    return load_full_checkpoint_state_dict(model, checkpoint)


def resolve_tokenizer_source(checkpoint: Path | None, base_model: str) -> str:
    """Use checkpoint tokenizer when present, otherwise fall back to base model.

    PEFT checkpoints usually carry tokenizer files. Internal-QAT adapter sidecars
    deliberately do not, so requiring a tokenizer there would make the lightweight
    export path unusable. ``checkpoint`` is None for ``--lora-mode none``, which
    has no checkpoint dir to check at all.
    """
    if checkpoint is None:
        return base_model
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any((checkpoint / name).exists() for name in tokenizer_files):
        return str(checkpoint)
    return base_model


def main() -> None:
    args = parse_args()
    if args.lora_mode != "none" and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --lora-mode none")
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype_from_arg(args.dtype),
        device_map={"": device.index or 0} if device.type == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    load_info = {}
    if args.lora_mode == "external_peft":
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    # "none" attaches nothing here; the base model is patched/baked as-is below.

    bitlinear_kwargs = {}
    if args.lora_mode == "internal_qat":
        # In internal_qat, BitLinear must be created with LoRA slots before state
        # loading. The checkpoint keys target those in-layer lora_A/lora_B params.
        bitlinear_kwargs = {
            "use_lora": True,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }

    patched = _replace_target_linears(
        model,
        mode=args.qat_mode,
        group_size=args.group_size,
        act_quant=False,
        bitlinear_kwargs=bitlinear_kwargs,
        target_names=tuple(args.target_names) if args.target_names else None,
        skip_names=tuple(args.skip_names),
    )
    bitlinear_before_merge = sum(isinstance(module, BitLinear) for module in model.modules())
    if args.lora_mode == "external_peft":
        # Historical path: PEFT delta is merged into the BitLinear shadow weight,
        # then bake quantizes/dequantizes the merged tensor for HF/GGUF export.
        model = model.merge_and_unload(progressbar=False)
    elif args.lora_mode == "internal_qat":
        # Internal-QAT path: LoRA is already part of BitLinear.forward/export_codes,
        # so do not PEFT-merge. Just load internal tensors and let bake export
        # QDQ(W + LoRA) exactly as trained.
        load_info = load_full_or_internal_lora_state(model, checkpoint)
    elif args.lora_mode == "full_weight":
        # full_weight has no LoRA at all -- the checkpoint's model*.safetensors
        # shards are the complete trained BitLinear.weight state. Nothing to
        # merge; just overwrite the patched (but untrained) shadow weights.
        load_info = load_full_or_internal_lora_state(model, checkpoint)
    # "none": nothing to merge/load -- bake the patched base weights directly.
    bitlinear_after_merge = sum(isinstance(module, BitLinear) for module in model.modules())
    baked = bake_bitlinear_to_linear(model)
    bitlinear_after_bake = sum(isinstance(module, BitLinear) for module in model.modules())

    model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    tokenizer_source = resolve_tokenizer_source(checkpoint, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": args.base_model,
        "checkpoint": str(checkpoint),
        "lora_mode": args.lora_mode,
        "load_info": load_info,
        "output_dir": str(output_dir),
        "tokenizer_source": tokenizer_source,
        "dtype": args.dtype,
        "qat_mode": args.qat_mode,
        "group_size": args.group_size,
        "target_names": args.target_names,
        "skip_names": args.skip_names,
        "patched_linears": patched,
        "bitlinear_before_merge": bitlinear_before_merge,
        "bitlinear_after_merge": bitlinear_after_merge,
        "baked_bitlinear": baked,
        "bitlinear_after_bake": bitlinear_after_bake,
    }
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "bake_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
