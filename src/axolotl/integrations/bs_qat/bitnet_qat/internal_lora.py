"""Utilities for KH internal-QAT LoRA checkpoints.

Internal-QAT LoRA stores ``lora_A``/``lora_B`` directly on ``BitLinear`` so the
forward path is ``QDQ(W + LoRA)``. These tensors are intentionally not PEFT
adapters, so export/eval tooling needs small helpers to save and restore them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .bitlinear import BitLinear


ADAPTER_WEIGHTS = "bs_internal_qat_lora.safetensors"
ADAPTER_CONFIG = "bs_internal_qat_lora_config.json"


def internal_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect only BitLinear-owned LoRA tensors.

    This intentionally does not mimic PEFT's adapter key layout. Internal-QAT
    checkpoints are tied to BitLinear module names because the adapter is part of
    the quantized layer composition, not an external module wrapping it.
    """
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.use_lora:
            # The root-module case matters for unit tests and tiny smoke models:
            # root BitLinear keys are "lora_A"/"lora_B", not ".lora_A".
            prefix = f"{name}." if name else ""
            state[f"{prefix}lora_A"] = module.lora_A.detach().cpu().contiguous()
            state[f"{prefix}lora_B"] = module.lora_B.detach().cpu().contiguous()
    return state


def save_internal_lora(
    model: nn.Module,
    out_dir: str | Path,
    *,
    config: dict | None = None,
) -> dict:
    """Write a lightweight adapter sidecar for an internal-QAT model.

    Axolotl sees internal_qat as a non-PEFT run, so normal Trainer checkpoints
    can contain the full model. This helper writes just the trainable internal
    LoRA tensors in safetensors format for quick evaluation and GGUF export.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = internal_lora_state_dict(model)
    save_file(state, str(out_dir / ADAPTER_WEIGHTS))
    cfg = {"format": "bs_internal_qat_lora", **(config or {})}
    (out_dir / ADAPTER_CONFIG).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"path": str(out_dir), "tensors": len(state), "parameters": sum(t.numel() for t in state.values())}


def load_full_checkpoint_state_dict(
    model: nn.Module,
    checkpoint_dir: str | Path,
) -> dict:
    """Load a full (non-adapter) Trainer checkpoint into an already patched model.

    ``full_weight`` mode has no LoRA tensors of its own -- every BitLinear shadow
    weight is directly trainable, so the checkpoint's ``model*.safetensors``
    shards *are* the complete trained state. Used for both eval and export of
    ``bs_qat_lora_mode: full_weight`` checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    if (checkpoint_dir / "model.safetensors").exists():
        candidates.append(checkpoint_dir / "model.safetensors")
    candidates.extend(sorted(checkpoint_dir.glob("model-*.safetensors")))
    if not candidates:
        raise FileNotFoundError(f"{checkpoint_dir} contains no model safetensors shards")

    loaded: dict[str, torch.Tensor] = {}
    for path in candidates:
        loaded.update(load_file(str(path)))
    load_result = model.load_state_dict(loaded, strict=False)
    return {
        "loaded_tensors": len(loaded),
        "missing_model_keys": list(load_result.missing_keys),
        "unexpected_model_keys": list(load_result.unexpected_keys),
    }


def load_internal_lora(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    strict: bool = True,
) -> dict:
    """Load internal-QAT LoRA tensors into an already patched BitLinear model.

    Callers must first construct/patch the base model with the same internal
    LoRA rank and target module names. With ``strict=True`` we fail fast if the
    adapter keys do not exist on that model, which catches accidental attempts to
    load an internal adapter into an external-PEFT or unpatched model.
    """
    adapter_dir = Path(adapter_dir)
    state = load_file(str(adapter_dir / ADAPTER_WEIGHTS))
    model_state = model.state_dict()
    missing = sorted(set(state) - set(model_state))
    if missing and strict:
        raise KeyError(f"Internal-QAT LoRA keys not present in model: {missing[:20]}")
    compatible = {key: value for key, value in state.items() if key in model_state}
    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "missing_adapter_keys": missing,
        "missing_model_keys": list(load_result.missing_keys),
        "unexpected_model_keys": list(load_result.unexpected_keys),
    }
