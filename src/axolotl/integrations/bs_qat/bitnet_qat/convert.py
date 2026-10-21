"""Config-driven conversion of ``nn.Linear`` layers to ternary ``BitLinear``.

This is architecture-agnostic: target layers are selected purely by their
fully-qualified module name using glob patterns, so the same machinery works for
Qwen3, Llama, Mistral, etc. -- only the YAML ``include``/``exclude`` lists change.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field

import torch.nn as nn

from .bitlinear import BinaryBitLinear, BitLinear, TernaryBitLinear
from .ternary import (
    DEFAULT_GROUP_SIZE,
    MODE_BINARY,
    MODE_TERNARY,
    QUANT_MODES,
    is_group_divisible,
)

logger = logging.getLogger(__name__)

# Module-name globs that usually denote a MoE *router* (gate) rather than an
# expert FFN. Routers must stay full precision, so converting them is almost
# always a mistake -- we warn if any matched layer looks like one.
_ROUTER_HINT_PATTERNS = [
    "*.mlp.gate",
    "*.shared_expert_gate",
    "*.block_sparse_moe.gate",
]


@dataclass
class ConvertConfig:
    """Selection + quantization options for :func:`convert_model`."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    group_size: int = DEFAULT_GROUP_SIZE
    act_quant: bool = False
    # Default quant mode for matched layers: "ternary" (Q2_0) or "binary" (Q1_0).
    mode: str = MODE_TERNARY
    # Optional per-pattern mode overrides (glob -> mode), checked before ``mode``.
    mode_overrides: dict[str, str] = field(default_factory=dict)
    # If True, skip (keep FP) any matched layer whose in_features is not a
    # multiple of group_size (cannot be packed as GGUF Q1_0/Q2_0). If False, raise.
    skip_non_divisible: bool = True

    def __post_init__(self) -> None:
        if self.mode not in QUANT_MODES:
            raise ValueError(f"mode={self.mode!r} not in {QUANT_MODES}")
        for pat, m in self.mode_overrides.items():
            if m not in QUANT_MODES:
                raise ValueError(f"mode_overrides[{pat!r}]={m!r} not in {QUANT_MODES}")


@dataclass
class ConvertReport:
    converted: list[str] = field(default_factory=list)
    converted_modes: dict[str, str] = field(default_factory=dict)
    skipped_not_divisible: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n_bin = sum(m == MODE_BINARY for m in self.converted_modes.values())
        n_ter = sum(m == MODE_TERNARY for m in self.converted_modes.values())
        return (
            f"converted={len(self.converted)} (ternary={n_ter} binary={n_bin}) "
            f"skipped_not_divisible={len(self.skipped_not_divisible)} "
            f"excluded={len(self.excluded)}"
        )


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _mode_for(name: str, config: ConvertConfig) -> str:
    """Quant mode for ``name``: first matching per-pattern override, else default."""
    for pattern, mode in config.mode_overrides.items():
        if fnmatch.fnmatchcase(name, pattern):
            return mode
    return config.mode


def list_linear_modules(model: nn.Module) -> list[tuple[str, int, int]]:
    """Return ``(name, in_features, out_features)`` for every ``nn.Linear``.

    Useful for discovering the right ``include`` patterns on a new architecture.
    """
    out: list[tuple[str, int, int]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out.append((name, module.in_features, module.out_features))
    return out


def _get_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_model(model: nn.Module, config: ConvertConfig, **kwargs) -> ConvertReport:
    """Replace matched ``nn.Linear`` modules in-place with ``BitLinear`` layers.

    Each matched layer is quantized in ``binary`` (Q1_0) or ``ternary`` (Q2_0)
    mode per ``config.mode`` / ``config.mode_overrides``. Returns a
    :class:`ConvertReport` describing what was changed.
    """
    report = ConvertReport()

    # Collect first (mutating during named_modules() iteration is unsafe).
    targets: list[str] = []
    lora_targets: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.exclude and _matches(name, config.exclude):
            report.excluded.append(name)
            continue
        if not config.include or _matches(name, config.include):
            targets.append(name)
            if _matches(name, config.include):
                lora_targets.add(name)

    router_like = [n for n in targets if _matches(n, _ROUTER_HINT_PATTERNS)]
    if router_like:
        shown = ", ".join(router_like[:4]) + ("..." if len(router_like) > 4 else "")
        logger.warning(
            "convert_model: %d matched layer(s) look like MoE router gates and will "
            "be quantized (%s). Routers should stay full precision -- add them to "
            "`exclude` (e.g. 'model.layers.*.mlp.gate').",
            len(router_like),
            shown,
        )

    for name in targets:
        parent, attr = _get_parent(model, name)
        linear: nn.Linear = getattr(parent, attr)

        if not is_group_divisible(linear.in_features, config.group_size):
            if config.skip_non_divisible:
                report.skipped_not_divisible.append(name)
                continue
            raise ValueError(
                f"{name}: in_features={linear.in_features} not divisible by "
                f"group_size={config.group_size}; cannot pack as Q1_0/Q2_0."
        )

        mode = _mode_for(name, config)
        cls = BinaryBitLinear if mode == MODE_BINARY else TernaryBitLinear

        is_lora_target = not config.include or name in lora_targets
        layer_kwargs = kwargs.copy()
        if "use_lora" in layer_kwargs:
            layer_kwargs["use_lora"] = layer_kwargs["use_lora"] and is_lora_target

        bit_linear = cls.from_linear(
            linear,
            mode=mode,
            group_size=config.group_size,
            act_quant=config.act_quant,
            **layer_kwargs,
        )
        setattr(parent, attr, bit_linear)
        report.converted.append(name)
        report.converted_modes[name] = mode

    return report


# Backward-compatible alias (ternary was the original only mode).
convert_to_ternary = convert_model


def count_bitlinear_modules(model: nn.Module, mode: str | None = None) -> int:
    """Count :class:`BitLinear` layers, optionally filtered by quant ``mode``."""
    return sum(
        isinstance(m, BitLinear) and (mode is None or m.mode == mode)
        for m in model.modules()
    )


def count_ternary_modules(model: nn.Module) -> int:
    """Count ternary :class:`BitLinear` layers (backward-compatible helper)."""
    return count_bitlinear_modules(model, mode=MODE_TERNARY)
