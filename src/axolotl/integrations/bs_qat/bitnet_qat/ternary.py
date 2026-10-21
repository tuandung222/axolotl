"""Group-wise ternary quantization primitives (Bonsai-style g128).

Weights are quantized to ``{-1, 0, +1}`` with one shared scale per group of
``group_size`` weights along the input (contraction) dimension. This matches the
Ternary-Bonsai weight format described in the whitepaper:

    w_i = s_g * t_i,   t_i in {-1, 0, +1}

where ``s_g`` is the shared scale for the group ``g`` that contains weight ``i``.
The scale is computed using the GGUF Q2_0 ``absmax`` rule, ``s_g = max(|w_g|)``,
applied independently per group of 128 weights.

These primitives are backend-agnostic (pure ``torch``) and are reused by:
  * ``bitlinear.TernaryBitLinear`` (QAT fake-quant during training), and
  * the exporters (``export/``) that pack the trained weights into GGUF ``Q2_0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Group size for the Bonsai ternary format. Confirmed from the PrismML llama.cpp
# fork: ``#define QK2_0 128`` (block_q2_0 = fp16 scale + 32 bytes of 2-bit codes).
DEFAULT_GROUP_SIZE = 128

# Floor for the per-group scale to avoid division by zero on all-zero groups.
EPS = 1e-5

# Quantization modes: ternary (1.58-bit, GGUF Q2_0) and binary (1-bit, GGUF Q1_0).
MODE_TERNARY = "ternary"
MODE_BINARY = "binary"
# Symmetric int4 (W4A16), bit-compatible with the compressed-tensors
# ``pack-quantized`` export used for vLLM/Marlin serving (see ``quantize_int4``).
MODE_INT4 = "int4"
QUANT_MODES = (MODE_TERNARY, MODE_BINARY, MODE_INT4)
INT4_QMIN, INT4_QMAX = -8, 7


def _padded_in_features(in_features: int, group_size: int) -> int:
    """Smallest multiple of ``group_size`` that is >= ``in_features``."""
    rem = in_features % group_size
    return in_features if rem == 0 else in_features + (group_size - rem)



def quantize_ternary(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to ternary codes + per-group scales.

    Args:
        w: weight matrix, shape ``(out_features, in_features)``.
        group_size: weights per shared scale (default 128).
        eps: lower clamp for the scale.
        fp16_scale: round the group scale to fp16 precision (default True) so the
            result matches the deployed GGUF ``Q2_0`` container exactly.

    Returns:
        ``(codes, scales)`` where ``codes`` is ``int8`` in ``{-1, 0, +1}`` with
        shape ``(out_features, in_features)`` and ``scales`` is ``float32`` with
        shape ``(out_features, n_groups)``.
    """
    out_features, in_features = w.shape
    in_padded = _padded_in_features(in_features, group_size)
    w_pad = w if in_padded == in_features else F.pad(w, (0, in_padded - in_features))
    n_groups = in_padded // group_size

    wg = w_pad.reshape(out_features, n_groups, group_size).to(torch.float32)
    # --- Kiet Hoang's original implementation (kept for reference, superseded below) ---
    # scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=eps)  # (out, n_groups, 1)
    # if fp16_scale:
    #     # The deployed Q2_0 container stores the group scale in fp16; round here
    #     # so codes (and the dequantized weights) match inference bit-for-bit.
    #     scale = scale.to(torch.float16).to(torch.float32)
    # codes = torch.round(wg / scale).clamp_(-1, 1).to(torch.int8)

    # --- John's fix: divide by the unrounded fp32 amax (not the fp16-stored scale)
    # and round half-away-from-zero. This is what llama.cpp's own ternary quantizer
    # does; matching it bit-for-bit here matters more than matching the stored scale
    # used for dequantization (torch.round above rounds half-to-even, which diverges
    # from llama.cpp's roundf-style half-away-from-zero on exact .5 boundaries). ---
    amax = wg.abs().amax(dim=-1, keepdim=True)
    inv = torch.where(amax > 0.0, 1.0 / amax, 0.0)
    scale = amax.to(torch.float16).to(torch.float32) if fp16_scale else amax.clamp(min=eps)
    x = wg * inv
    codes = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
    codes = codes.clamp_(-1, 1).to(torch.int8)
    codes = codes.reshape(out_features, in_padded)[:, :in_features].contiguous()
    return codes, scale.squeeze(-1).contiguous()


def quantize_binary(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to 1-bit binary codes + per-group absmean scales.

    BitNet 1-bit rule: ``code = sign(w)`` in ``{-1, +1}`` (``w == 0`` maps to
    ``+1`` to match the GGUF ``Q1_0`` sign-bit convention) with
    ``scale = mean(|w_g|)`` per group. The container dequantizes as ``code * scale``.

    Returns:
        ``(codes, scales)`` -- ``int8`` codes in ``{-1, +1}`` of shape
        ``(out_features, in_features)`` and ``float32`` scales
        ``(out_features, n_groups)``.
    """
    out_features, in_features = w.shape
    in_padded = _padded_in_features(in_features, group_size)
    w_pad = w if in_padded == in_features else F.pad(w, (0, in_padded - in_features))
    n_groups = in_padded // group_size

    wg = w_pad.reshape(out_features, n_groups, group_size).to(torch.float32)
    scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=eps)  # (out, n_groups, 1)
    if fp16_scale:
        scale = scale.to(torch.float16).to(torch.float32)
    codes = torch.where(wg >= 0, 1, -1).to(torch.int8)
    codes = codes.reshape(out_features, in_padded)[:, :in_features].contiguous()
    return codes, scale.squeeze(-1).contiguous()


def quantize_int4(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group int4 codes + scales, matching the compressed-tensors export.

    Mirrors ``torchao_to_compressed_tensors.quantize.quantize_weight`` (symmetric,
    ``num_bits=4``, bf16 weights): ``scale = max|w_g| / 7.5`` (``(qmax - qmin) / 2``),
    the scale is rounded to **bf16 before** quantising, and
    ``code = clamp(round_half_even(w / scale), -8, 7)``. Training with this
    quantizer therefore produces exactly the grid the exported checkpoint has.

    ``fp16_scale`` is accepted for signature compatibility and ignored: the scale
    dtype is bf16 because that is what the exporter stores for a bf16 model.

    Returns:
        ``(codes, scales)``: ``int8`` codes in ``[-8, 7]`` of shape
        ``(out_features, in_features)`` and ``float32`` (bf16-rounded) scales of
        shape ``(out_features, n_groups)``.
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
    wg = w.reshape(out_features, in_features // group_size, group_size).to(torch.float32)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / ((INT4_QMAX - INT4_QMIN) / 2)).to(torch.bfloat16).to(torch.float32)
    scale = scale.clamp(min=torch.finfo(torch.bfloat16).tiny)
    codes = torch.round(wg / scale).clamp_(INT4_QMIN, INT4_QMAX).to(torch.int8)
    return codes.reshape(out_features, in_features).contiguous(), scale.squeeze(-1).contiguous()


def quantize(
    w: torch.Tensor,
    mode: str = MODE_TERNARY,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to :func:`quantize_ternary`, :func:`quantize_binary` or :func:`quantize_int4` by ``mode``."""
    if mode == MODE_TERNARY:
        return quantize_ternary(w, group_size, eps, fp16_scale)
    if mode == MODE_BINARY:
        return quantize_binary(w, group_size, eps, fp16_scale)
    if mode == MODE_INT4:
        return quantize_int4(w, group_size, eps, fp16_scale)
    raise ValueError(f"unknown quant mode {mode!r}; expected one of {QUANT_MODES}")


def dequantize_ternary(
    codes: torch.Tensor, scales: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE
) -> torch.Tensor:
    """Reconstruct float weights from ternary codes + per-group scales.

    Inverse of :func:`quantize_ternary` (up to the rounding already applied).

    Args:
        codes: ``int8``/float codes, shape ``(out_features, in_features)``.
        scales: per-group scales, shape ``(out_features, n_groups)``.
        group_size: weights per shared scale.

    Returns:
        ``float32`` weight matrix, shape ``(out_features, in_features)``.
    """
    out_features, in_features = codes.shape
    in_padded = _padded_in_features(in_features, group_size)
    n_groups = in_padded // group_size

    c = codes.to(torch.float32)
    if in_padded != in_features:
        c = F.pad(c, (0, in_padded - in_features))
    cg = c.reshape(out_features, n_groups, group_size)
    w = cg * scales.to(torch.float32).unsqueeze(-1)
    return w.reshape(out_features, in_padded)[:, :in_features].contiguous()


# ``code * scale`` reconstruction is identical for binary and ternary codes.
dequantize = dequantize_ternary
dequantize_binary = dequantize_ternary


class _FakeQuantSTE(torch.autograd.Function):
    """Forward = group-wise fake-quant (binary/ternary); backward = identity (STE)."""

    @staticmethod
    def forward(
        ctx, w: torch.Tensor, mode: str, group_size: int, eps: float, fp16_scale: bool
    ) -> torch.Tensor:
        codes, scales = quantize(w, mode, group_size, eps, fp16_scale)
        return dequantize(codes, scales, group_size).to(w.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through estimator: pass gradient unchanged to the shadow weight.
        return grad_output, None, None, None, None


def fake_quantize(
    w: torch.Tensor,
    mode: str = MODE_TERNARY,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> torch.Tensor:
    """Differentiable group-wise fake-quant (STE). Returns dequantized weights."""
    return _FakeQuantSTE.apply(w, mode, group_size, eps, fp16_scale)


def fake_quantize_ternary(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> torch.Tensor:
    """Differentiable ternary fake-quant (STE). Returns dequantized weights."""
    return _FakeQuantSTE.apply(w, MODE_TERNARY, group_size, eps, fp16_scale)


def fake_quantize_binary(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = EPS,
    fp16_scale: bool = True,
) -> torch.Tensor:
    """Differentiable 1-bit binary fake-quant (STE). Returns dequantized weights."""
    return _FakeQuantSTE.apply(w, MODE_BINARY, group_size, eps, fp16_scale)


class _ActQuantSTE(torch.autograd.Function):
    """Per-token 8-bit absmax activation fake-quant with STE (BitNet-style)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
        return (torch.round(x * scale).clamp(-128, 127) / scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def activation_quant(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Optional 8-bit activation fake-quant (off by default; Bonsai is weight-only)."""
    return _ActQuantSTE.apply(x, eps)


def is_group_divisible(in_features: int, group_size: int = DEFAULT_GROUP_SIZE) -> bool:
    """Whether a tensor's input dim can be packed as GGUF Q2_0 without padding."""
    return in_features % group_size == 0
