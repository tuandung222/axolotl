"""bitnet_qat -- generic 1-bit / 1.58-bit quantization-aware training toolkit.

Converts any Hugging Face causal LM (Qwen3 as the worked example) to Bonsai-style
group-128 binary ({-1,+1}) or ternary ({-1,0,+1}) weights and exports to
safetensors / GGUF ``Q1_0``/``Q2_0`` / MLX.
"""

from __future__ import annotations

from .bitlinear import BinaryBitLinear, BitLinear, TernaryBitLinear
from .convert import ConvertConfig, convert_model, convert_to_ternary
from .distill import (
    attention_relation_loss,
    capture_qkv,
    resolve_relation_heads,
)
from .ternary import (
    DEFAULT_GROUP_SIZE,
    MODE_BINARY,
    MODE_INT4,
    MODE_TERNARY,
    QUANT_MODES,
    dequantize,
    dequantize_ternary,
    fake_quantize,
    fake_quantize_binary,
    fake_quantize_ternary,
    is_group_divisible,
    quantize,
    quantize_binary,
    quantize_int4,
    quantize_ternary,
)

__all__ = [
    "BitLinear",
    "BinaryBitLinear",
    "TernaryBitLinear",
    "ConvertConfig",
    "convert_model",
    "convert_to_ternary",
    "attention_relation_loss",
    "capture_qkv",
    "resolve_relation_heads",
    "DEFAULT_GROUP_SIZE",
    "MODE_BINARY",
    "MODE_INT4",
    "MODE_TERNARY",
    "QUANT_MODES",
    "dequantize",
    "dequantize_ternary",
    "fake_quantize",
    "fake_quantize_binary",
    "fake_quantize_ternary",
    "is_group_divisible",
    "quantize",
    "quantize_binary",
    "quantize_int4",
    "quantize_ternary",
]

__version__ = "0.1.0"
