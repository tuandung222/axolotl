"""Patch Axolotl models with OneBitLLMs llama.cpp fake quant linear layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None or str(name).lower() in {"none", "null"}:
        return None
    normalized = str(name).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(
            f"unsupported llama_cpp_qat_accumulator_dtype={name!r}; "
            f"allowed: {allowed}, none"
        )
    return mapping[normalized]


def _find_module_by_path(model: nn.Module, path: str) -> nn.Module | None:
    module = model
    for part in path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def _language_model_submodule(model: nn.Module) -> nn.Module:
    for path in ("model.language_model", "language_model"):
        module = _find_module_by_path(model, path)
        if module is not None:
            return module
    return model


def _replace_target_linears(
    module: nn.Module,
    *,
    quant_type: str,
    target_names: tuple[str, ...] | None,
    skip_names: tuple[str, ...],
    activation_quant: str | None,
    accumulator_dtype: torch.dtype | None,
    backend: str,
    path: str = "",
) -> int:
    from onebitllms.layers import LlamaCppFakeQuantLinear

    block_sizes = {"Q1_0": 128, "Q2_0": 128, "Q4_0": 32, "Q4_1": 32, "Q8_0": 32, "Q8_1": 32}
    quant_key = quant_type.upper()
    if quant_key not in block_sizes:
        allowed = ", ".join(sorted(block_sizes))
        raise ValueError(f"unsupported llama_cpp_qat_quant_type={quant_type!r}; allowed: {allowed}")
    block_size = block_sizes[quant_key]
    if activation_quant is not None:
        block_size = max(block_size, 32)

    patched = 0
    for child_name, child in list(module.named_children()):
        child_path = f"{path}.{child_name}" if path else child_name
        if child_name in skip_names:
            continue
        if isinstance(child, LlamaCppFakeQuantLinear):
            continue
        if isinstance(child, nn.Linear):
            dotted_child_path = f".{child_path}."
            is_lora_adapter = ".lora_A." in dotted_child_path or ".lora_B." in dotted_child_path
            is_target = target_names is None or any(target in child_path for target in target_names)
            if is_target and not is_lora_adapter and child.in_features % block_size == 0:
                setattr(
                    module,
                    child_name,
                    LlamaCppFakeQuantLinear.from_linear(
                        child,
                        quant_type=quant_key,
                        activation_quant=activation_quant.upper() if activation_quant else None,
                        accumulator_dtype=accumulator_dtype,
                        backend=backend,
                    ),
                )
                patched += 1
                continue
        patched += _replace_target_linears(
            child,
            quant_type=quant_key,
            target_names=target_names,
            skip_names=skip_names,
            activation_quant=activation_quant,
            accumulator_dtype=accumulator_dtype,
            backend=backend,
            path=child_path,
        )
    return patched


class LlamaCppQATPlugin(BasePlugin):
    """Apply llama.cpp-compatible fake quant wrappers after Axolotl loads the model."""

    def get_input_args(self):
        return "axolotl.integrations.llama_cpp_qat.args.LlamaCppQATArgs"

    def post_lora_load(self, cfg, model):
        self._apply_qat(cfg, model, hook_name="post_lora_load")

    def post_model_load(self, cfg, model):
        self._apply_qat(cfg, model, hook_name="post_model_load")

    def _apply_qat(self, cfg, model, *, hook_name: str):
        if not cfg.llama_cpp_qat:
            return

        from onebitllms.layers import LlamaCppFakeQuantLinear

        before = sum(isinstance(module, LlamaCppFakeQuantLinear) for module in model.modules())
        if before:
            LOG.info("OneBitLLMs llama.cpp QAT already applied before %s: patched_linears=%s", hook_name, before)
            return
        target_names = tuple(cfg.llama_cpp_qat_target_names) if cfg.llama_cpp_qat_target_names else None
        skip_names = tuple(cfg.llama_cpp_qat_skip_names or ["lm_head"])

        patched = _replace_target_linears(
            _language_model_submodule(model),
            quant_type=cfg.llama_cpp_qat_quant_type,
            target_names=target_names,
            skip_names=skip_names,
            activation_quant=cfg.llama_cpp_qat_activation_quant,
            accumulator_dtype=_dtype_from_name(cfg.llama_cpp_qat_accumulator_dtype),
            backend=cfg.llama_cpp_qat_backend,
        )

        after = sum(isinstance(module, LlamaCppFakeQuantLinear) for module in model.modules())
        LOG.info(
            "Applied OneBitLLMs llama.cpp QAT at %s: quant_type=%s activation_quant=%s backend=%s patched_linears=%s",
            hook_name,
            cfg.llama_cpp_qat_quant_type,
            cfg.llama_cpp_qat_activation_quant,
            cfg.llama_cpp_qat_backend,
            patched or after - before,
        )
