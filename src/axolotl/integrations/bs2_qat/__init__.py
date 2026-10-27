"""Bonsai-2 (BS2) Sylvester Walsh-Hadamard + PQ2_0 Ternary QAT Integration for Axolotl."""

from .args import BS2QATArgs
from .builtin_signs import get_signs_for_width, infer_bonsai2_block_size, load_signs_table
from .layers import (
    Bonsai2LinearFakeQuant,
    fwht_fast,
    load_bs2_internal_lora,
    mark_only_bs2_internal_lora_as_trainable,
    save_bs2_internal_lora,
    snap_bs2_qat_to_standard_linear,
    ternary_quantize_pq2_0_no_grad,
)
from .plugin import BS2QATPlugin

__all__ = [
    "BS2QATArgs",
    "BS2QATPlugin",
    "Bonsai2LinearFakeQuant",
    "fwht_fast",
    "get_signs_for_width",
    "infer_bonsai2_block_size",
    "load_bs2_internal_lora",
    "load_signs_table",
    "mark_only_bs2_internal_lora_as_trainable",
    "save_bs2_internal_lora",
    "snap_bs2_qat_to_standard_linear",
    "ternary_quantize_pq2_0_no_grad",
]
