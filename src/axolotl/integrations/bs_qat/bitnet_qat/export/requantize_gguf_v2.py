"""Selective GGUF Q2_0 repacker for Bonsai LoRA-over-QAT exports.

Why this file exists:

* The Prism llama.cpp fork already knows the Q2_0 block format and kernels, but
  its ``convert_hf_to_gguf.py`` path is a broad "mostly Q2_0" converter. That is
  not the same as our QAT contract. Our training config fake-quantizes only the
  seven transformer projection weights listed below and explicitly skips
  ``lm_head``. A broad converter can quantize embeddings or ``output.weight``
  even though those tensors were not trained under QAT.
* The original ``requantize_gguf.py`` in this integration is generic and includes
  ``output.weight`` in its default patterns. That was convenient for experiments,
  but it is too permissive for the current LoRA-over-QAT Bonsai export path.
* This v2 helper encodes the export contract directly: first merge LoRA into the
  QAT shadow weights, then bake to a normal F16 HF/GGUF model, then repack only
  the tensors that actually used fake quant during training. If the tensor count
  or names do not match expectations, the optional strict checks fail the export
  instead of silently producing a mixed or incorrectly quantized model.

The intended export pipeline is:

1. Load the HF base model and the selected PEFT LoRA checkpoint.
2. Patch QAT target linears with ``BitLinear``.
3. Merge LoRA into the ``BitLinear.weight`` shadow weights.
4. Bake ``BitLinear`` modules to plain ``nn.Linear`` dequantized Q2_0 weights.
5. Convert the baked HF model to an F16 GGUF with the Prism llama.cpp fork.
6. Repack only the QAT-trained GGUF tensors to Q2_0 with this module.

Unlike the original ``requantize_gguf.py``, v2 defaults to the seven QAT target
projections used by the current Axolotl Bonsai config and does not quantize
``output.weight``/``lm_head`` or token embeddings unless explicitly requested.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..gguf_quant import pack_q1_0, pack_q2_0, quantize_ternary_absmax
from ._gguf import gguf

LOG = logging.getLogger(__name__)


# QAT target set from examples/bs_bonsai_lora_q2_0_fa3_run.yaml:
# q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
QAT_Q2_0_PATTERNS = [
    "blk.*.attn_q.weight",
    "blk.*.attn_k.weight",
    "blk.*.attn_v.weight",
    "blk.*.attn_output.weight",
    "blk.*.ffn_gate.weight",
    "blk.*.ffn_up.weight",
    "blk.*.ffn_down.weight",
]

MOE_QAT_Q2_0_PATTERNS = [
    "blk.*.ffn_gate_exps.weight",
    "blk.*.ffn_up_exps.weight",
    "blk.*.ffn_down_exps.weight",
    "blk.*.ffn_gate_shexp.weight",
    "blk.*.ffn_up_shexp.weight",
    "blk.*.ffn_down_shexp.weight",
]

OUTPUT_PATTERNS = ["output.weight"]
EMBEDDING_PATTERNS = ["token_embd.weight"]

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
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


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


def _field_contents(reader, key: str):
    field = reader.get_field(key)
    return field.contents() if field is not None else None


def _write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_q2_0_patterns(
    *,
    patterns: list[str] | None = None,
    include_moe: bool = False,
    include_output: bool = False,
    include_embeddings: bool = False,
) -> list[str]:
    """Return the Q2_0 tensor-name patterns for this export.

    ``patterns`` replaces the default target list. The include flags append
    optional tensor classes explicitly; they are off by default to mirror the
    current QAT config where ``lm_head`` is skipped.
    """
    out = list(QAT_Q2_0_PATTERNS if patterns is None else patterns)
    if include_moe:
        out.extend(MOE_QAT_Q2_0_PATTERNS)
    if include_output:
        out.extend(OUTPUT_PATTERNS)
    if include_embeddings:
        out.extend(EMBEDDING_PATTERNS)
    return out


def requantize_gguf_v2(
    in_path: str | Path,
    out_path: str | Path,
    *,
    q2_0_patterns: list[str] | None = None,
    q1_0_patterns: list[str] | None = None,
    include_moe: bool = False,
    include_output: bool = False,
    include_embeddings: bool = False,
    group_size: int = 128,
    strict_count: int | None = None,
    strict_names: list[str] | None = None,
    manifest_path: str | Path | None = None,
    use_temp_file: bool = True,
) -> dict[str, Any]:
    """Repack selected F16/F32 GGUF tensors as Q2_0/Q1_0.

    Defaults are intentionally conservative for the current Bonsai QAT flow:
    only q/k/v/o/gate/up/down projection tensors are Q2_0. ``output.weight`` and
    embeddings stay copied unless explicitly included.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    patterns = build_q2_0_patterns(
        patterns=q2_0_patterns,
        include_moe=include_moe,
        include_output=include_output,
        include_embeddings=include_embeddings,
    )
    binary_patterns = list(q1_0_patterns or [])

    reader = gguf.GGUFReader(str(in_path), "r")
    arch = _field_contents(reader, gguf.Keys.General.ARCHITECTURE)
    if not arch:
        raise ValueError(f"{in_path}: missing general.architecture")
    src_ftype = _field_contents(reader, gguf.Keys.General.FILE_TYPE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(
        str(out_path), arch=arch, endianess=reader.endianess, use_temp_file=use_temp_file
    )
    alignment = _field_contents(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    _copy_kv(reader, writer)

    quantized_q2_0: list[str] = []
    quantized_q1_0: list[str] = []
    copied: list[str] = []
    skipped_matched_not_float: list[str] = []
    skipped_matched_bad_shape: list[str] = []

    for tensor in reader.tensors:
        name = tensor.name
        data = np.array(tensor.data)
        matches_q1 = _matches(name, binary_patterns)
        matches_q2 = not matches_q1 and _matches(name, patterns)
        is_float = tensor.tensor_type in _FLOAT_TYPES
        is_packable = data.ndim >= 2 and data.shape[-1] % group_size == 0

        if (matches_q1 or matches_q2) and not is_float:
            skipped_matched_not_float.append(name)
        elif (matches_q1 or matches_q2) and not is_packable:
            skipped_matched_bad_shape.append(name)

        if matches_q1 and is_float and is_packable:
            packed = pack_q1_0(data.astype(np.float32))
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q1_0)
            quantized_q1_0.append(name)
        elif matches_q2 and is_float and is_packable:
            codes, scales = quantize_ternary_absmax(data.astype(np.float32), group_size)
            packed = pack_q2_0(codes, scales)
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q2_0)
            quantized_q2_0.append(name)
        elif tensor.tensor_type in _COPY_TYPES:
            writer.add_tensor(name, data)
            copied.append(name)
        else:
            writer.add_tensor(name, data, raw_dtype=tensor.tensor_type)
            copied.append(name)

    if strict_count is not None and len(quantized_q2_0) != strict_count:
        raise RuntimeError(
            f"Q2_0 tensor count mismatch: got {len(quantized_q2_0)}, expected {strict_count}. "
            f"Matched bad-shape={len(skipped_matched_bad_shape)} not-float={len(skipped_matched_not_float)}"
        )
    if strict_names is not None:
        missing = sorted(set(strict_names) - set(quantized_q2_0))
        extra = sorted(set(quantized_q2_0) - set(strict_names))
        if missing or extra:
            raise RuntimeError(
                f"Q2_0 tensor name mismatch: missing={missing[:20]} extra={extra[:20]}"
            )

    if quantized_q2_0:
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q2_0))
    elif quantized_q1_0:
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q1_0))
    elif src_ftype is not None:
        writer.add_file_type(int(src_ftype))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    result: dict[str, Any] = {
        "input": str(in_path),
        "output": str(out_path),
        "arch": arch,
        "source_file_type": int(src_ftype) if src_ftype is not None else None,
        "group_size": group_size,
        "q2_0_patterns": patterns,
        "q1_0_patterns": binary_patterns,
        "quantized": len(quantized_q2_0) + len(quantized_q1_0),
        "quantized_q2_0": len(quantized_q2_0),
        "quantized_q1_0": len(quantized_q1_0),
        "copied": len(copied),
        "quantized_q2_0_names": quantized_q2_0,
        "quantized_q1_0_names": quantized_q1_0,
        "skipped_matched_not_float": skipped_matched_not_float,
        "skipped_matched_bad_shape": skipped_matched_bad_shape,
    }
    if manifest_path is not None:
        _write_manifest(manifest_path, result)

    LOG.info(
        "requantize_gguf_v2: Q2_0=%d Q1_0=%d copied=%d bad_shape=%d not_float=%d -> %s",
        len(quantized_q2_0),
        len(quantized_q1_0),
        len(copied),
        len(skipped_matched_bad_shape),
        len(skipped_matched_not_float),
        out_path,
    )
    return result


def _load_strict_names(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Q2_0 GGUF tensor glob. Replaces defaults when provided.")
    parser.add_argument("--q1-pattern", action="append", dest="q1_patterns", help="Q1_0 GGUF tensor glob.")
    parser.add_argument("--include-moe", action="store_true", help="Also quantize MoE expert/shared-expert FFN tensors.")
    parser.add_argument("--include-output", action="store_true", help="Also quantize output.weight/lm_head.")
    parser.add_argument("--include-embeddings", action="store_true", help="Also quantize token_embd.weight.")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--strict-count", type=int)
    parser.add_argument("--strict-names-file", help="Require exact Q2_0 tensor names from this newline-delimited file.")
    parser.add_argument("--manifest", help="Write JSON manifest with quantized/copied tensor lists.")
    parser.add_argument("--no-temp-file", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = requantize_gguf_v2(
        args.in_path,
        args.out_path,
        q2_0_patterns=args.patterns,
        q1_0_patterns=args.q1_patterns,
        include_moe=args.include_moe,
        include_output=args.include_output,
        include_embeddings=args.include_embeddings,
        group_size=args.group_size,
        strict_count=args.strict_count,
        strict_names=_load_strict_names(args.strict_names_file),
        manifest_path=args.manifest,
        use_temp_file=not args.no_temp_file,
    )
    print(
        f"Wrote {result['output']} "
        f"(Q2_0={result['quantized_q2_0']} Q1_0={result['quantized_q1_0']} copied={result['copied']})"
    )


if __name__ == "__main__":
    main()
