"""QAD (quantization-aware distillation) for W4A16 g128 models on Axolotl.

Student: PTQ (GPTQ/RTN W4A16 g128 symmetric) weights, fake-quantised on the PTQ grid (fixed scales).
Teacher: the BF16 model the PTQ was made from.
"""
from .args import QADArgs, QADTrainingArgsMixin
from .fixed_scale_fq import (
    QMAX,
    QMIN,
    FQLinear,
    FQLoRALinear,
    add_internal_lora,
    apply_fixed_scale_fq,
    fake_quant,
    fake_quant_lsq,
    set_fake_quant,
)
from .kl import chunked_kl_loss
from .ptq_io import load_ptq


def __getattr__(name: str):
    if name == "QADPlugin":
        from .plugin import QADPlugin

        return QADPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QADPlugin",
    "QADArgs",
    "QADTrainingArgsMixin",
    "QMIN",
    "QMAX",
    "FQLinear",
    "FQLoRALinear",
    "fake_quant",
    "fake_quant_lsq",
    "apply_fixed_scale_fq",
    "add_internal_lora",
    "set_fake_quant",
    "chunked_kl_loss",
    "load_ptq",
]
