"""Rewrite selected tensors of an F16/F32 GGUF into Bonsai ternary ``Q2_0``.

This is the generic, architecture-agnostic GGUF export path:

  1. Train with QAT, "bake" the on-grid weights, then run the fork's
     ``convert_hf_to_gguf.py --outtype f16`` to get a standard F16 GGUF
     (correct metadata + tokenizer for *any* supported architecture).
  2. Call :func:`requantize_gguf` to repack the projection/LM-head tensors as
     ``Q2_0`` (group 128). Because QAT weights already lie on the ternary grid,
     the absmax repack is lossless; the same function also serves as a
     post-training ternary quantizer for arbitrary F16 GGUFs.

Tensor selection uses GGUF tensor names (``blk.*.attn_q.weight`` etc.), so it is
decoupled from PyTorch module names.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import numpy as np

from ..gguf_quant import pack_q1_0, pack_q2_0, quantize_ternary_absmax
from ._gguf import gguf

logger = logging.getLogger(__name__)

# Standard dense projection + LM-head tensor names (llama.cpp convention), plus
# MoE expert tensors. Qwen-style MoE stacks experts into 3D tensors
# (``blk.*.ffn_{gate,up,down}_exps.weight``) and may add a dense shared expert
# (``*_shexp``). The router (``blk.*.ffn_gate_inp.weight``) is intentionally NOT
# listed, so it is copied at full precision (it must stay FP for routing).
DEFAULT_QUANT_PATTERNS = [
    "blk.*.attn_q.weight",
    "blk.*.attn_k.weight",
    "blk.*.attn_v.weight",
    "blk.*.attn_output.weight",
    "blk.*.ffn_gate.weight",
    "blk.*.ffn_up.weight",
    "blk.*.ffn_down.weight",
    "blk.*.ffn_gate_exps.weight",
    "blk.*.ffn_up_exps.weight",
    "blk.*.ffn_down_exps.weight",
    "blk.*.ffn_gate_shexp.weight",
    "blk.*.ffn_up_shexp.weight",
    "blk.*.ffn_down_shexp.weight",
    "output.weight",
]

_COPY_TYPES = {
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F64,
    gguf.GGMLQuantizationType.I8,
    gguf.GGMLQuantizationType.I16,
    gguf.GGMLQuantizationType.I32,
    gguf.GGMLQuantizationType.I64,
}
_FLOAT_TYPES = {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _copy_kv(reader, writer) -> None:
    for field in reader.fields.values():
        if (
            field.name == gguf.Keys.General.ARCHITECTURE
            or field.name == gguf.Keys.General.FILE_TYPE
            or field.name.startswith("GGUF.")
        ):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)


def requantize_gguf(
    in_path: str | Path,
    out_path: str | Path,
    patterns: list[str] | None = None,
    binary_patterns: list[str] | None = None,
    quantize_embeddings: bool = False,
    group_size: int = 128,
    use_temp_file: bool = True,
) -> dict:
    """Repack matched F16/F32 tensors of ``in_path`` into Bonsai low-bit GGUF.

    Tensors matching ``binary_patterns`` are packed as 1-bit ``Q1_0``; tensors
    matching ``patterns`` (the ternary set) are packed as 1.58-bit ``Q2_0``.
    ``binary_patterns`` takes precedence when a name matches both. Because QAT
    weights already lie on the binary/ternary grid the repack is lossless; this
    also doubles as a post-training quantizer for arbitrary F16 GGUFs.

    Returns ``{quantized, quantized_q1_0, quantized_q2_0, copied, quantized_names}``.
    """
    patterns = list(DEFAULT_QUANT_PATTERNS if patterns is None else patterns)
    binary_patterns = list(binary_patterns or [])
    if quantize_embeddings:
        patterns.append("token_embd.weight")

    reader = gguf.GGUFReader(str(in_path), "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else None
    if not arch:
        raise ValueError(f"{in_path}: missing general.architecture")

    # Preserve the source ftype only as a fallback (pure passthrough); when we
    # actually quantize, general.file_type is overwritten below.
    ft_field = reader.get_field(gguf.Keys.General.FILE_TYPE)
    src_ftype = ft_field.contents() if ft_field is not None else None

    writer = gguf.GGUFWriter(
        str(out_path), arch=arch, endianess=reader.endianess, use_temp_file=use_temp_file
    )
    align_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align_field is not None:
        writer.data_alignment = align_field.contents()

    _copy_kv(reader, writer)

    n_q1 = n_q2 = n_copy = 0
    quantized_names: list[str] = []
    for t in reader.tensors:
        name = t.name
        data = np.array(t.data)  # materialize out of the mmap
        # 2D dense weights and 3D stacked MoE experts are both packable; grouping
        # is over the last (input/contraction) axis.
        quantizable = (
            t.tensor_type in _FLOAT_TYPES
            and data.ndim >= 2
            and data.shape[-1] % group_size == 0
        )
        as_binary = quantizable and _matches(name, binary_patterns)
        as_ternary = quantizable and not as_binary and _matches(name, patterns)

        if as_binary:
            packed = pack_q1_0(data.astype(np.float32))
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q1_0)
            n_q1 += 1
            quantized_names.append(name)
        elif as_ternary:
            codes, scales = quantize_ternary_absmax(data.astype(np.float32), group_size)
            packed = pack_q2_0(codes, scales)
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q2_0)
            n_q2 += 1
            quantized_names.append(name)
        elif t.tensor_type in _COPY_TYPES:
            writer.add_tensor(name, data)  # dtype inferred (F16/F32/int)
            n_copy += 1
        else:
            writer.add_tensor(name, data, raw_dtype=t.tensor_type)  # passthrough quant
            n_copy += 1

    # Stamp general.file_type with the headline quantization so loaders/tools
    # (llama-bench, llama-cli) report the real type instead of the inherited
    # source F16. Q2_0 takes precedence over Q1_0; fall back to the source ftype
    # when nothing was quantized (pure passthrough).
    if n_q2:
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q2_0))
    elif n_q1:
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q1_0))
    elif src_ftype is not None:
        writer.add_file_type(int(src_ftype))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    n_quant = n_q1 + n_q2
    logger.info(
        "requantize_gguf: Q1_0=%d Q2_0=%d copied=%d -> %s", n_q1, n_q2, n_copy, out_path
    )
    return {
        "quantized": n_quant,
        "quantized_q1_0": n_q1,
        "quantized_q2_0": n_q2,
        "copied": n_copy,
        "quantized_names": quantized_names,
    }
