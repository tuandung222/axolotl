"""Read a compressed-tensors pack-quantized W4A16 checkpoint -> dense bf16 weights + per-group scales."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_ptq(ptq_dir: Path, device: str = "cpu"):
    """Returns (dense_state_dict, scales, group_size). dense has bf16 dequantised weights for quantised Linears
    and untouched tensors otherwise; scales maps module name -> bf16 [out, in/gs]."""
    from compressed_tensors.compressors.pack_quantized import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationStrategy

    ptq_dir = Path(ptq_dir)
    quant = json.loads((ptq_dir / "config.json").read_text())["quantization_config"]
    bits = {g["weights"]["num_bits"] for g in quant["config_groups"].values()}
    assert bits == {4}, f"expected 4-bit W4A16 checkpoint, got {bits}"
    tensors = {}
    for shard in sorted(ptq_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    prefixes = [k[: -len(".weight_packed")] for k in tensors if k.endswith(".weight_packed")]
    dense = {k: v for k, v in tensors.items() if not any(k.startswith(f"{p}.weight_") for p in prefixes)}
    scales, gs_seen = {}, set()
    for p in prefixes:
        shape = torch.Size(tensors[f"{p}.weight_shape"].tolist())
        scale = tensors[f"{p}.weight_scale"].to(device)
        gs = int(shape[1]) // scale.shape[-1]
        gs_seen.add(gs)
        args = QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy=QuantizationStrategy.GROUP, group_size=gs)
        q = unpack_from_int32(tensors[f"{p}.weight_packed"].to(device), 4, shape)
        dense[f"{p}.weight"] = dequantize(x_q=q, scale=scale, zero_point=None, args=args).to(torch.bfloat16).cpu()
        scales[p] = scale.to(torch.bfloat16).cpu()
    assert len(gs_seen) == 1, f"mixed group sizes {gs_seen}"
    return dense, scales, gs_seen.pop()
