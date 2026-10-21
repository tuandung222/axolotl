"""Bake binary/ternary fake-quant into plain ``nn.Linear`` weights.

After QAT, each :class:`BitLinear` holds a full-precision shadow weight. Baking
replaces it with a standard ``nn.Linear`` whose weight equals the dequantized
(on-grid) weight ``code * scale`` for the layer's quant ``mode``. The resulting
model is a normal HF model that can be saved with ``save_pretrained`` and
converted to GGUF by the fork's ``convert_hf_to_gguf.py``; the subsequent
``Q1_0``/``Q2_0`` repack is then lossless because the weights already lie on the
binary/ternary grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..bitlinear import BitLinear
from ..convert import _get_parent
from ..ternary import dequantize, quantize


@torch.no_grad()
def bake_bitlinear_to_linear(model: nn.Module) -> int:
    """Replace every :class:`BitLinear` with an on-grid ``nn.Linear`` in place.

    Returns the number of modules baked.
    """
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    ]
    for name, module in targets:
        codes, scales = module.export_codes()
        w = dequantize(codes, scales, module.group_size).to(module.weight.dtype)

        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        linear.weight.copy_(w)
        if module.bias is not None:
            linear.bias.copy_(module.bias)

        parent, attr = _get_parent(model, name)
        setattr(parent, attr, linear)

    return len(targets)


# Backward-compatible alias (ternary was the original only mode).
bake_ternary_to_linear = bake_bitlinear_to_linear
