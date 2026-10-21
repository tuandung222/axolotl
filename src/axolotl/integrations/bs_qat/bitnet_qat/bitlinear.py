"""``TernaryBitLinear``: a drop-in ``nn.Linear`` replacement for ternary QAT.

During the forward pass the full-precision *shadow* weight is fake-quantized to
group-wise ternary values (see :mod:`bitnet_qat.ternary`); gradients flow to the
shadow weight via a straight-through estimator. After training, the shadow
weight is exported to GGUF ``Q2_0`` (group 128) by the exporters.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import (
    DEFAULT_GROUP_SIZE,
    EPS,
    MODE_BINARY,
    MODE_INT4,
    MODE_TERNARY,
    activation_quant,
    fake_quantize,
    quantize,
)


class BitLinear(nn.Module):
    """Linear layer with group-wise low-bit weight fake-quant (QAT).

    ``mode`` selects the weight grid: ``"ternary"`` (``{-1,0,+1}``, exported to
    GGUF ``Q2_0``) or ``"binary"`` (``{-1,+1}``, exported to GGUF ``Q1_0``). The
    stored ``weight`` stays full precision (the "shadow" weight) and is
    fake-quantized on-the-fly in :meth:`forward`; gradients reach it via a
    straight-through estimator. Set ``act_quant=True`` for full BitNet-style
    8-bit activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mode: str = MODE_TERNARY,
        group_size: int = DEFAULT_GROUP_SIZE,
        act_quant: bool = False,
        eps: float = EPS,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        self.act_quant = act_quant
        self.eps = eps
        self._cached_weight_key: Optional[tuple[torch.device, torch.dtype, int]] = None
        self._cached_weight: Optional[torch.Tensor] = None

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        self.use_lora = use_lora
        if use_lora:
            import math
            # BitNet mixed precision: trainable params stay FP32 for stable
            # accumulation even though the base weight / compute is bf16.
            self.lora_A = nn.Parameter(torch.empty(lora_r, in_features, device=device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_r, device=device, dtype=torch.float32))
            self.lora_scaling = lora_alpha / lora_r
            if lora_dropout > 0.0:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = nn.Identity()
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.use_lora:
            import math
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        self.clear_weight_cache()

    def clear_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None

    def _cacheable_weight(self) -> bool:
        return not self.use_lora and not self.weight.requires_grad

    def _fake_quantized_weight(self) -> torch.Tensor:
        if not self._cacheable_weight():
            w = self.weight.to(torch.float32)
            if self.use_lora:
                a = self.lora_dropout(self.lora_A)
                w = w + self.lora_scaling * (self.lora_B @ a)
            if self.mode == MODE_INT4:
                # The exporter merges W0 + BA into a bf16 checkpoint before it
                # quantises; round to bf16 here (straight-through) so the codes
                # trained on are exactly the codes exported.
                w = w + (w.to(torch.bfloat16).to(torch.float32) - w).detach()
            return fake_quantize(w, self.mode, self.group_size, self.eps).to(
                self.weight.dtype
            )

        key = (self.weight.device, self.weight.dtype, self.weight._version)
        if self._cached_weight_key != key or self._cached_weight is None:
            with torch.no_grad():
                w = self.weight.to(torch.float32)
                self._cached_weight = fake_quantize(
                    w, self.mode, self.group_size, self.eps
                ).to(self.weight.dtype)
            self._cached_weight_key = key
        return self._cached_weight

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        mode: str = MODE_TERNARY,
        group_size: int = DEFAULT_GROUP_SIZE,
        act_quant: bool = False,
        eps: float = EPS,
        **kwargs,
    ) -> "BitLinear":
        """Build a ``BitLinear`` (or subclass) copying ``linear``'s weights/bias."""
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            mode=mode,
            group_size=group_size,
            act_quant=act_quant,
            eps=eps,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            module.weight.requires_grad = linear.weight.requires_grad
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
                module.bias.requires_grad = linear.bias.requires_grad
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base weights are common for LoRA-over-QAT-base. Cache their
        # Q2_0/Q1_0 dequantized view instead of recomputing it every forward.
        w_eff = self._fake_quantized_weight()
        if self.act_quant:
            x = activation_quant(x, self.eps)
        return F.linear(x, w_eff, self.bias)

    @torch.no_grad()
    def export_codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(codes, scales)`` for the current shadow weight (per ``mode``)."""
        w = self.weight.to(torch.float32)
        if getattr(self, "use_lora", False):
            w = w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return quantize(w.detach(), self.mode, self.group_size, self.eps)

    # Backward-compatible alias (ternary was the original only mode).
    export_ternary = export_codes

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode}, "
            f"group_size={self.group_size}, act_quant={self.act_quant}"
        )


class TernaryBitLinear(BitLinear):
    """:class:`BitLinear` fixed to ternary (1.58-bit, GGUF ``Q2_0``)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("mode", None)
        super().__init__(*args, mode=MODE_TERNARY, **kwargs)


class BinaryBitLinear(BitLinear):
    """:class:`BitLinear` fixed to binary (1-bit, GGUF ``Q1_0``)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("mode", None)
        super().__init__(*args, mode=MODE_BINARY, **kwargs)
