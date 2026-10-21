"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .convert import ConvertConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_convert_config(cfg: dict[str, Any]) -> ConvertConfig:
    """Build a :class:`ConvertConfig` from the top-level YAML dict."""
    return ConvertConfig(
        include=list(cfg.get("include", [])),
        exclude=list(cfg.get("exclude", [])),
        group_size=int(cfg.get("group_size", 128)),
        act_quant=bool(cfg.get("activation_quant", False)),
        mode=str(cfg.get("mode", "ternary")),
        mode_overrides=dict(cfg.get("mode_overrides", {}) or {}),
    )
