"""Axolotl plugin for Bonsai-2 (BS2) Sylvester Walsh-Hadamard + PQ2_0 Ternary QAT."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import DEFAULT_BS2_SKIP_NAMES, DEFAULT_BS2_TARGET_NAMES
from .builtin_signs import infer_bonsai2_block_size, load_signs_table
from .layers import (
    Bonsai2LinearFakeQuant,
    mark_only_bs2_internal_lora_as_trainable,
    save_bs2_internal_lora,
    snap_bs2_qat_to_standard_linear,
)

LOG = get_logger(__name__)

MODE_FULL_WEIGHT = "full_weight"
MODE_INTERNAL_QAT = "internal_qat"
MODE_EXTERNAL_PEFT = "external_peft"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        val = cfg.get(key, default)
        return default if val is None else val
    val = getattr(cfg, key, default)
    return default if val is None else val


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


def _language_model_submodule(model: nn.Module) -> nn.Module:
    """Return the language model subtree so Vision Tower (`model.visual`) is never quantized."""
    for path in (
        "model.language_model",
        "base_model.model.model.language_model",
        "language_model",
    ):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.Module):
            return cur
    return model


def _replace_target_linears_bs2(
    module: nn.Module,
    *,
    block_size: int,
    signs_table: dict[int, torch.Tensor],
    signs_seed: int,
    group_size: int,
    store_rotated: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    target_names: tuple[str, ...] | None,
    skip_names: tuple[str, ...],
    path: str = "",
) -> int:
    from .builtin_signs import get_signs_for_width

    patched = 0
    for child_name, child in list(module.named_children()):
        child_path = f"{path}.{child_name}" if path else child_name
        # Never descend into vision tower modules
        if child_name in ("visual", "vision_tower", "vision_model"):
            continue
        if any(skip == child_name or child_path.endswith(f".{skip}") for skip in skip_names):
            continue

        if isinstance(child, nn.Linear) and not isinstance(child, Bonsai2LinearFakeQuant):
            is_target = target_names is None or any(
                t == child_name or child_path.endswith(f".{t}") for t in target_names
            )
            is_lora_leaf = child_name in ("lora_A", "lora_B") or ".lora_" in child_path
            if (
                is_target
                and not is_lora_leaf
                and child.in_features % group_size == 0
                and (block_size <= 0 or child.in_features % block_size == 0)
            ):
                signs = get_signs_for_width(
                    child.in_features,
                    signs_table=signs_table,
                    seed=signs_seed,
                    device=child.weight.device,
                )
                q_layer = Bonsai2LinearFakeQuant.from_linear(
                    child,
                    block_size=block_size,
                    signs=signs,
                    group_size=group_size,
                    store_rotated=store_rotated,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(module, child_name, q_layer)
                patched += 1
                continue

        patched += _replace_target_linears_bs2(
            child,
            block_size=block_size,
            signs_table=signs_table,
            signs_seed=signs_seed,
            group_size=group_size,
            store_rotated=store_rotated,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_names=target_names,
            skip_names=skip_names,
            path=child_path,
        )
    return patched


class BS2InternalQATLoRASaveCallback(TrainerCallback):
    """Save BS2 internal-QAT LoRA sidecar (`bs2_internal_qat_lora.safetensors`) on checkpoint save."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        model = getattr(self.trainer, "model", None) or kwargs.get("model")
        if model is None:
            return control
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "bs2_internal_qat_lora"
        info = save_bs2_internal_lora(
            model,
            ckpt_dir,
            config={"global_step": int(state.global_step), "source": "axolotl_bs2_qat_checkpoint"},
        )
        LOG.info(
            "Saved BS2 internal-QAT LoRA sidecar at step %s: tensors=%s params=%s path=%s",
            state.global_step,
            info["tensors"],
            info["parameters"],
            info["path"],
        )
        return control


class BS2QATPlugin(BasePlugin):
    """Axolotl plugin for Bonsai-2 Sylvester Walsh-Hadamard + PQ2_0 Ternary QAT."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.bs2_qat.args.BS2QATArgs"

    def pre_lora_load(self, cfg, model):
        if _cfg_get(cfg, "bs2_qat_lora_mode", MODE_FULL_WEIGHT) != MODE_EXTERNAL_PEFT:
            self._apply_bs2_qat(cfg, model, hook_name="pre_lora_load")

    def post_lora_load(self, cfg, model):
        self._apply_bs2_qat(cfg, model, hook_name="post_lora_load")

    def post_model_load(self, cfg, model):
        if not _cfg_get(cfg, "adapter", None):
            self._apply_bs2_qat(cfg, model, hook_name="post_model_load")

    def add_callbacks(self, cfg, trainer):
        if not _cfg_get(cfg, "bs2_qat", False):
            return []
        if _cfg_get(cfg, "bs2_qat_lora_mode", MODE_FULL_WEIGHT) != MODE_INTERNAL_QAT:
            return []
        return [BS2InternalQATLoRASaveCallback(trainer)]

    def post_train(self, cfg, model):
        if not _cfg_get(cfg, "bs2_qat", False):
            return
        lora_mode = _cfg_get(cfg, "bs2_qat_lora_mode", MODE_FULL_WEIGHT)
        output_dir = _cfg_get(cfg, "output_dir", None)
        if lora_mode == MODE_INTERNAL_QAT and output_dir and _is_rank_zero():
            info = save_bs2_internal_lora(
                model,
                Path(output_dir) / "bs2_internal_qat_lora",
                config={"source": "axolotl_bs2_qat_post_train"},
            )
            LOG.info(
                "Saved final BS2 internal-QAT LoRA adapter: tensors=%s params=%s path=%s",
                info["tensors"],
                info["parameters"],
                info["path"],
            )
        if _cfg_get(cfg, "bs2_qat_snap_on_finish", False):
            snapped = snap_bs2_qat_to_standard_linear(model)
            LOG.info(
                "Snapped %s Bonsai2LinearFakeQuant layers onto Hadamard PQ2_0 ternary grid and unfolded to nn.Linear",
                snapped,
            )

    def _validate_cfg(self, cfg) -> None:
        lora_mode = _cfg_get(cfg, "bs2_qat_lora_mode", MODE_FULL_WEIGHT)
        adapter = _cfg_get(cfg, "adapter", None)
        if lora_mode == MODE_INTERNAL_QAT and adapter:
            raise ValueError(
                "bs2_qat_lora_mode='internal_qat' is incompatible with Axolotl `adapter: lora/qlora`. "
                "Set `adapter:` to null/empty so Bonsai2LinearFakeQuant owns lora_A/lora_B inside "
                "the Hadamard + PQ2_0 quantizer."
            )
        if lora_mode == MODE_FULL_WEIGHT and adapter:
            raise ValueError(
                "bs2_qat_lora_mode='full_weight' is incompatible with Axolotl `adapter: lora/qlora`. "
                "Set `adapter:` to null/empty, or use bs2_qat_lora_mode='external_peft'."
            )

    def _apply_bs2_qat(self, cfg, model: nn.Module, *, hook_name: str) -> None:
        if not _cfg_get(cfg, "bs2_qat", False):
            return
        self._validate_cfg(cfg)

        existing = sum(isinstance(m, Bonsai2LinearFakeQuant) for m in model.modules())
        if existing > 0:
            LOG.info("BS2 QAT already applied before %s: layers=%s", hook_name, existing)
            return

        block_size = _cfg_get(cfg, "bs2_qat_block_size", None)
        if block_size is None:
            block_size = infer_bonsai2_block_size(model)
        group_size = int(_cfg_get(cfg, "bs2_qat_group_size", 128))
        store_rotated = bool(_cfg_get(cfg, "bs2_qat_store_rotated", False))
        signs_path = _cfg_get(cfg, "bs2_qat_signs_path", None)
        signs_seed = int(_cfg_get(cfg, "bs2_qat_signs_seed", 0))
        signs_table = load_signs_table(signs_path)

        lora_mode = _cfg_get(cfg, "bs2_qat_lora_mode", MODE_FULL_WEIGHT)
        use_lora = lora_mode == MODE_INTERNAL_QAT
        lora_r = int(_cfg_get(cfg, "bs2_qat_lora_r", _cfg_get(cfg, "lora_r", 16)))
        lora_alpha = float(_cfg_get(cfg, "bs2_qat_lora_alpha", _cfg_get(cfg, "lora_alpha", 32.0)))
        lora_dropout = float(_cfg_get(cfg, "bs2_qat_lora_dropout", _cfg_get(cfg, "lora_dropout", 0.0)))

        raw_targets = _cfg_get(cfg, "bs2_qat_target_names", None)
        target_names = tuple(raw_targets) if raw_targets else tuple(DEFAULT_BS2_TARGET_NAMES)
        raw_skips = _cfg_get(cfg, "bs2_qat_skip_names", None)
        skip_names = tuple(raw_skips) if raw_skips is not None else tuple(DEFAULT_BS2_SKIP_NAMES)

        lm_root = _language_model_submodule(model)
        patched = _replace_target_linears_bs2(
            lm_root,
            block_size=block_size,
            signs_table=signs_table,
            signs_seed=signs_seed,
            group_size=group_size,
            store_rotated=store_rotated,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_names=target_names,
            skip_names=skip_names,
        )

        trainable_tensors = trainable_params = 0
        if use_lora and _cfg_get(cfg, "bs2_qat_freeze_non_lora", True):
            trainable_tensors, trainable_params = mark_only_bs2_internal_lora_as_trainable(model)

        LOG.info(
            "Applied BS2 Hadamard Ternary QAT at %s: block_size=%s group_size=%s "
            "lora_mode=%s store_rotated=%s patched_linears=%s",
            hook_name,
            block_size,
            group_size,
            lora_mode,
            store_rotated,
            patched,
        )
        if use_lora:
            LOG.info(
                "BS2 internal-QAT LoRA trainable_tensors=%s trainable_parameters=%s (r=%s, alpha=%s)",
                trainable_tensors,
                trainable_params,
                lora_r,
                lora_alpha,
            )
