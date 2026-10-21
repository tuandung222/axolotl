"""Patch Axolotl models with BS ternary bitlinear layers.

Three LoRA/QAT compositions are supported (``bs_qat_lora_mode``):

* ``external_peft`` keeps Axolotl/PEFT LoRA outside the fake-quantized layer.
  The forward path is ``QDQ(W_base) + LoRA_delta``. This is convenient for
  normal PEFT training, but a single baked GGUF must quantize the merged tensor,
  so export becomes ``QDQ(W_base + LoRA_delta)`` and can lose quality.
* ``internal_qat`` creates ``lora_A``/``lora_B`` inside ``BitLinear`` before
  PEFT is applied. The forward path is ``QDQ(W_base + LoRA_delta)`` during
  training, matching the baked Q2_0 GGUF deployment objective.
* ``full_weight`` has no LoRA at all: every target ``BitLinear.weight`` is
  directly trainable and fake-quantized as ``QDQ(W)``, i.e. full-parameter QAT.
  Code-path-wise this patches identically to ``external_peft`` (BitLinear with
  ``use_lora=False``) -- the distinct name exists so a config that intentionally
  has no adapter at all doesn't have to claim "external_peft" (which implies a
  PEFT adapter exists outside the layer, which is precisely what full_weight
  does not have).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _find_module_by_path(model: nn.Module, path: str) -> nn.Module | None:
    module = model
    for part in path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def _language_model_submodule(model: nn.Module) -> nn.Module:
    # Some wrapped architectures put the actual causal LM under a language_model
    # child. Patching the smallest relevant subtree avoids touching unrelated
    # wrapper heads while still falling back to the full model for normal HF CMs.
    for path in ("model.language_model", "language_model"):
        module = _find_module_by_path(model, path)
        if module is not None:
            return module
    return model


def _matches_target(child_path: str, target: str) -> bool:
    # Match on whole dot-separated path segments, not raw substring: a bare leaf
    # name like "up_proj" must not match a fused module named "gate_up_proj".
    if "." in target:
        return child_path == target or child_path.endswith(f".{target}")
    return target in child_path.split(".")


def _replace_target_linears(
    module: nn.Module,
    *,
    mode: str,
    group_size: int,
    act_quant: bool,
    bitlinear_kwargs: dict | None = None,
    target_names: tuple[str, ...] | None,
    skip_names: tuple[str, ...],
    path: str = "",
) -> int:
    from .bitnet_qat.bitlinear import BitLinear

    patched = 0
    bitlinear_kwargs = bitlinear_kwargs or {}
    for child_name, child in list(module.named_children()):
        child_path = f"{path}.{child_name}" if path else child_name
        if child_name in skip_names:
            continue
        if isinstance(child, BitLinear):
            continue
        if isinstance(child, nn.Linear):
            dotted_child_path = f".{child_path}."
            # PEFT adapters contain their own lora_A/lora_B Linear modules. In
            # external_peft mode those must never be recursively replaced by
            # BitLinear, otherwise the adapter math and PEFT state dict no longer
            # match what Axolotl/PEFT expect.
            is_lora_adapter = ".lora_A." in dotted_child_path or ".lora_B." in dotted_child_path
            is_target = target_names is None or any(
                _matches_target(child_path, target) for target in target_names
            )
            # GGUF Q2_0 packs fixed-size groups along input features. Layers that
            # are not divisible by group_size are skipped instead of padded here,
            # keeping training and export tensor names/shapes one-to-one.
            if is_target and not is_lora_adapter and child.in_features % group_size == 0:
                setattr(
                    module,
                    child_name,
                    BitLinear.from_linear(
                        child,
                        mode=mode,
                        group_size=group_size,
                        act_quant=act_quant,
                        **bitlinear_kwargs,
                    ),
                )
                patched += 1
                continue
        patched += _replace_target_linears(
            child,
            mode=mode,
            group_size=group_size,
            act_quant=act_quant,
            bitlinear_kwargs=bitlinear_kwargs,
            target_names=target_names,
            skip_names=skip_names,
            path=child_path,
        )
    return patched


def _patch_cce_for_bitlinear_head(model: nn.Module) -> bool:
    """Make Cut-Cross-Entropy see the fake-quantized (and LoRA-merged) lm_head.

    The CCE forward patch passes ``self.lm_head.weight`` straight to the fused
    loss, which would bypass ``BitLinear``'s fake-quant and LoRA entirely when
    ``lm_head`` is a ``BitLinear``. Wrap CCE's ``apply_lce`` so that, when it is
    handed the raw head parameter, it gets ``QDQ(W + LoRA)`` instead (gradients
    reach ``lora_A/lora_B`` through the STE). Returns True if a wrapper was
    installed.
    """
    from .bitnet_qat.bitlinear import BitLinear

    head = getattr(model, "lm_head", None)
    if not isinstance(head, BitLinear):
        return False
    try:
        import cut_cross_entropy.transformers.llama as cce_llama
    except ImportError:
        return False
    if getattr(cce_llama.apply_lce, "_bs_head_patch", False):
        cce_llama.apply_lce._bs_head = head  # refresh target on re-apply
        return True
    orig = cce_llama.apply_lce

    def apply_lce(e, c, labels, opts, *args, **kwargs):
        target = apply_lce._bs_head
        if c is target.weight:
            c = target._fake_quantized_weight()
        return orig(e, c, labels, opts, *args, **kwargs)

    apply_lce._bs_head_patch = True
    apply_lce._bs_head = head
    cce_llama.apply_lce = apply_lce
    LOG.info("Patched CCE apply_lce to use the fake-quantized BitLinear lm_head")
    return True


MODE_EXTERNAL_PEFT = "external_peft"
MODE_INTERNAL_QAT = "internal_qat"
MODE_FULL_WEIGHT = "full_weight"


def _cfg_get(cfg, key: str, default=None):
    return getattr(cfg, key, default)


def _lora_mode(cfg) -> str:
    return _cfg_get(cfg, "bs_qat_lora_mode", MODE_EXTERNAL_PEFT)


def _is_internal_qat(cfg) -> bool:
    return _lora_mode(cfg) == MODE_INTERNAL_QAT


def _internal_lora_kwargs(cfg) -> dict:
    # Prefer the integration-specific keys when set, but keep compatibility with
    # existing configs that already define lora_r/lora_alpha/lora_dropout.
    return {
        "use_lora": True,
        "lora_r": _cfg_get(cfg, "bs_qat_lora_r", None) or _cfg_get(cfg, "lora_r", 16),
        "lora_alpha": _cfg_get(cfg, "bs_qat_lora_alpha", None)
        or _cfg_get(cfg, "lora_alpha", 32),
        "lora_dropout": _cfg_get(cfg, "bs_qat_lora_dropout", None)
        if _cfg_get(cfg, "bs_qat_lora_dropout", None) is not None
        else _cfg_get(cfg, "lora_dropout", 0.05),
    }


def mark_only_internal_lora_as_trainable(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except KH internal LoRA parameters.

    Returns ``(trainable_tensors, trainable_parameters)``.
    """
    # Internal-QAT is intended to preserve the pretrained base as the QAT shadow
    # weight and learn only the low-rank delta that will be quantized together
    # with it. Freezing first makes the whitelist below explicit and robust if
    # new modules are added to the architecture.
    for param in model.parameters():
        param.requires_grad = False

    trainable_tensors = 0
    trainable_parameters = 0
    for name, param in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            param.requires_grad = True
            trainable_tensors += 1
            trainable_parameters += param.numel()
    return trainable_tensors, trainable_parameters


def _is_rank_zero() -> bool:
    # axolotl's train() calls cleanup_distributed() (destroys the raw torch.distributed
    # process group) before firing the post_train plugin hook, but Accelerate's
    # PartialState cache still reports initialized=True afterward. Routing through
    # is_main_process()/dist.get_rank() at that point raises "Default process group
    # has not been initialized" on every non-zero rank, so check the torchrun-set env
    # vars directly instead of touching torch.distributed here.
    rank = os.environ.get("RANK")
    if rank is not None:
        return rank == "0"
    return os.environ.get("LOCAL_RANK", "0") == "0"


def _unwrap_trainer_model(trainer):
    # Trainer may hold the model under Accelerate/DDP wrappers by save time. The
    # adapter helper needs real module names, so unwrap before collecting state.
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(trainer.model)
    return getattr(trainer.model, "module", trainer.model)


class InternalQATLoRASaveCallback(TrainerCallback):
    """Save lightweight internal-QAT LoRA tensors next to Trainer checkpoints."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        # HuggingFace Trainer calls callbacks on all ranks. Only rank zero should
        # write adapter files; the full Trainer checkpoint is handled separately.
        if not getattr(state, "is_world_process_zero", True):
            return control
        if not state.global_step:
            return control

        from .bitnet_qat.internal_lora import save_internal_lora

        model = _unwrap_trainer_model(self.trainer)
        # Because internal_qat sets cfg.adapter=None, Axolotl's default checkpoint
        # path is a full model save. This extra sidecar keeps a small LoRA-only
        # artifact for fast eval/export without changing Trainer resume behavior.
        out_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "internal_qat_lora"
        info = save_internal_lora(
            model,
            out_dir,
            config={
                "global_step": state.global_step,
                "source": "axolotl_bs_qat_callback",
            },
        )
        LOG.info(
            "Saved KH internal-QAT LoRA adapter: tensors=%s parameters=%s path=%s",
            info["tensors"],
            info["parameters"],
            info["path"],
        )
        return control


class BSQATPlugin(BasePlugin):
    """Apply BS ternary quant wrappers at the hook matching LoRA mode."""

    def get_input_args(self):
        return "axolotl.integrations.bs_qat.args.BSQATArgs"

    def post_model_build(self, cfg, model):
        # internal_qat must run before Axolotl attaches PEFT. The BitLinear owns
        # LoRA params itself, so there should be no PEFT adapter wrapping later.
        if not _is_internal_qat(cfg):
            return
        self._apply_qat(cfg, model, hook_name="post_model_build")

    def post_lora_load(self, cfg, model):
        # external_peft intentionally waits until PEFT has created/wrapped LoRA,
        # then replaces only the base target linears while leaving adapter modules
        # intact. That preserves the historical QDQ(W) + LoRA behavior.
        if _is_internal_qat(cfg):
            return
        self._apply_qat(cfg, model, hook_name="post_lora_load")

    def post_model_load(self, cfg, model):
        # Idempotent fallback for non-LoRA/full-model flows. If post_lora_load
        # already patched, _apply_qat sees existing BitLinear modules and exits.
        if _is_internal_qat(cfg):
            return
        self._apply_qat(cfg, model, hook_name="post_model_load")

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.bs_qat or not _is_internal_qat(cfg):
            return []
        return [InternalQATLoRASaveCallback(trainer)]

    def post_train(self, cfg, model):
        if not cfg.bs_qat or not _is_internal_qat(cfg):
            return
        if not _cfg_get(cfg, "output_dir", None):
            return
        if not _is_rank_zero():
            # This hook fires on every rank after cleanup_distributed() has already
            # torn down the process group; only rank zero should write the sidecar
            # (avoids N ranks racing to write the same file) or log through it.
            return
        from .bitnet_qat.internal_lora import save_internal_lora

        # The final output directory may not coincide with a checkpoint step if
        # training stops between saves. Persist a final sidecar as the canonical
        # lightweight adapter for downstream PPL eval and GGUF export.
        info = save_internal_lora(
            model,
            Path(cfg.output_dir) / "internal_qat_lora",
            config={"source": "axolotl_bs_qat_post_train"},
        )
        LOG.info(
            "Saved final KH internal-QAT LoRA adapter: tensors=%s parameters=%s path=%s",
            info["tensors"],
            info["parameters"],
            info["path"],
        )

    def _validate_lora_mode_cfg(self, cfg):
        if not cfg.bs_qat:
            return
        mode = _lora_mode(cfg)
        if mode == MODE_INTERNAL_QAT and _cfg_get(cfg, "adapter", None):
            # Allowing PEFT here would silently reintroduce the exact mismatch we
            # are avoiding: training would optimize QDQ(W) + LoRA but baked GGUF
            # deploys QDQ(W + LoRA).
            raise ValueError(
                "bs_qat_lora_mode=internal_qat is incompatible with Axolotl "
                "`adapter: lora/qlora`. Internal-QAT LoRA must own the LoRA params "
                "inside BitLinear so training uses QDQ(W + LoRA), not QDQ(W) + LoRA. "
                "Set `adapter:` to null/empty for this mode."
            )
        if mode == MODE_FULL_WEIGHT and _cfg_get(cfg, "adapter", None):
            # full_weight means "every BitLinear weight is directly trainable, no
            # LoRA anywhere" -- a PEFT adapter layered on top would train a LoRA
            # delta around an already-fully-trainable base, which defeats the
            # point and was almost certainly not intended.
            raise ValueError(
                "bs_qat_lora_mode=full_weight is incompatible with Axolotl "
                "`adapter: lora/qlora`. full_weight trains every target BitLinear "
                "shadow weight directly (QDQ(W), no LoRA). Set `adapter:` to "
                "null/empty, or use bs_qat_lora_mode: external_peft if you "
                "do want a PEFT LoRA adapter."
            )

    def _apply_qat(self, cfg, model, *, hook_name: str):
        if not cfg.bs_qat:
            return
        self._validate_lora_mode_cfg(cfg)

        from .bitnet_qat.bitlinear import BitLinear

        before = sum(isinstance(module, BitLinear) for module in model.modules())
        if before:
            # Axolotl can call multiple lifecycle hooks for the same load. Treat
            # repeated hooks as successful no-ops instead of recursively nesting
            # fake-quant wrappers.
            LOG.info("BS QAT already applied before %s: patched_linears=%s", hook_name, before)
            return
        target_names = tuple(cfg.bs_qat_target_names) if cfg.bs_qat_target_names else None
        skip_names = tuple(cfg.bs_qat_skip_names or ["lm_head"])
        bitlinear_kwargs = _internal_lora_kwargs(cfg) if _is_internal_qat(cfg) else {}

        patched = _replace_target_linears(
            _language_model_submodule(model),
            mode=cfg.bs_qat_mode,
            group_size=cfg.bs_qat_group_size,
            act_quant=cfg.bs_qat_act_quant,
            bitlinear_kwargs=bitlinear_kwargs,
            target_names=target_names,
            skip_names=skip_names,
        )

        _patch_cce_for_bitlinear_head(model)
        after = sum(isinstance(module, BitLinear) for module in model.modules())
        trainable_tensors = trainable_parameters = 0
        if _is_internal_qat(cfg) and _cfg_get(cfg, "bs_qat_freeze_non_lora", True):
            trainable_tensors, trainable_parameters = mark_only_internal_lora_as_trainable(model)
        LOG.info(
            "Applied BS QAT at %s: mode=%s lora_mode=%s patched_linears=%s",
            hook_name,
            cfg.bs_qat_mode,
            _lora_mode(cfg),
            patched or after - before,
        )
        if _is_internal_qat(cfg):
            LOG.info(
                "KH internal QAT LoRA trainable_tensors=%s trainable_parameters=%s",
                trainable_tensors,
                trainable_parameters,
            )


# Alias for branding
