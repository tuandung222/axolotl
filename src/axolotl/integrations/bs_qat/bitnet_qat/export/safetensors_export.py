"""Export a QAT model to a low-bit safetensors checkpoint.

Quantized layers are stored as ``int8`` codes (``{-1,0,+1}`` ternary or
``{-1,+1}`` binary) + ``fp16`` group scales (research-friendly, portable, and a
convenient source for MLX conversion); all other parameters keep their original
dtype. The per-module quant mode is recorded in the file metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..bitlinear import BitLinear


def export_safetensors(model: torch.nn.Module, out_path: str | Path, group_size: int = 128) -> dict:
    """Save ``model`` to ``out_path`` with codes/scales for every BitLinear layer.

    Layout per quantized module ``<name>``:
      * ``<name>.weight.codes``  -- int8 codes (``{-1,0,+1}`` or ``{-1,+1}``), ``(out, in)``
      * ``<name>.weight.scales`` -- fp16 group scales, ``(out, in/group)``
      * ``<name>.bias``          -- bias (if present)
    Every other parameter is stored under its original name and dtype. File
    metadata ``quantized_modules`` maps each module name to its quant mode.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    quantized: dict[str, str] = {}  # module name -> quant mode
    quant_modules: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            codes, scales = module.export_codes()
            tensors[f"{name}.weight.codes"] = codes.to(torch.int8).contiguous()
            tensors[f"{name}.weight.scales"] = scales.to(torch.float16).contiguous()
            if module.bias is not None:
                tensors[f"{name}.bias"] = module.bias.detach().cpu().contiguous()
            quantized[name] = module.mode
            quant_modules.add(name)

    for pname, param in model.named_parameters():
        owner, _, leaf = pname.rpartition(".")
        if owner in quant_modules and leaf in ("weight", "bias"):
            continue  # handled above as codes/scales
        tensors[pname] = param.detach().cpu().contiguous()

    metadata = {
        "format": "bitnet-g128",
        "group_size": str(group_size),
        "quantized_modules": json.dumps(quantized),
    }
    save_file(tensors, str(out_path), metadata=metadata)
    return {
        "path": str(out_path),
        "quantized_modules": len(quantized),
        "tensors": len(tensors),
    }
