"""Bit-exact packing/unpacking of the Bonsai GGUF ternary block formats.

Layouts confirmed from the PrismML ``llama.cpp`` fork (``prism`` branch):

``block_q2_0`` (ternary, group 128) -- 34 bytes per 128 weights::

    { ggml_half d;          // fp16 group scale (little-endian)
      uint8_t   qs[32]; }   // 2 bits per weight, 4 weights per byte

    weight j: byte = j // 4, shift = (j % 4) * 2
    stored value q = code + 1 in {0,1,2,3}
    dequant:  y = (q - 1) * d   ->  codes {0,1,2,3} map to {-1,0,+1,+2}

``block_q1_0`` (binary, group 128) -- 18 bytes per 128 weights::

    { ggml_half d;          // fp16 group scale (absmean)
      uint8_t   qs[16]; }   // 1 bit per weight; bit set => +d, else -d

GGML type ids: ``Q1_0 = 41``, ``Q2_0 = 42``.

``llama.cpp`` dequantizes as ``code * d``, so storing the absmax scale (Q2_0)
or absmean scale (Q1_0) here reproduces the training-time weights.

All pack/unpack helpers operate on the **last axis** and preserve leading dims,
so they handle 2D dense weights ``(rows, in_features)`` and 3D stacked MoE
expert tensors ``(n_expert, rows, in_features)`` alike.
"""

from __future__ import annotations

import numpy as np

QK1_0 = 128
QK2_0 = 128

GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q2_0 = 42

BLOCK_SIZE_Q1_0 = 2 + QK1_0 // 8   # 18 bytes
BLOCK_SIZE_Q2_0 = 2 + QK2_0 // 4   # 34 bytes


def _fp16_bytes(scales: np.ndarray) -> np.ndarray:
    """Cast a ``(..., n_groups)`` float array to little-endian fp16 bytes.

    Returns ``uint8`` array of shape ``(..., n_groups, 2)`` (leading dims kept).
    """
    d16 = np.ascontiguousarray(scales.astype("<f2"))
    return d16.view(np.uint8).reshape(*scales.shape, 2)


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Match C ``roundf`` (round half away from zero), unlike numpy's half-to-even."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_ternary_absmax(
    weights: np.ndarray, group_size: int = QK2_0, eps: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize weights to ternary using per-group absmax (matches ``quantize_row_q2_0_ref``).

    The fork's reference Q2_0 quantizer uses ``d = max(|w_g|)`` and
    ``code = round(w / d)`` (clamped). Because the GGUF container dequantizes as
    ``code * d``, weights that already lie on a ternary grid are repacked
    losslessly; arbitrary weights get a sensible post-training ternary quant.

    Args:
        weights: float array ``(..., in_features)`` (e.g. 2D dense or 3D MoE
            ``(n_expert, rows, in_features)``); grouping is over the last axis.
        group_size: weights per shared scale (default 128).

    Returns:
        ``(codes, scales)`` -- ``int8`` codes ``(..., in_features)`` in
        ``{-1,0,+1}`` and fp16-rounded ``float32`` scales ``(..., n_groups)``.
    """
    *lead, in_features = weights.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by {group_size}")
    n_groups = in_features // group_size
    wg = weights.reshape(*lead, n_groups, group_size).astype(np.float32)

    amax = np.abs(wg).max(axis=-1)  # (..., n_groups), full precision
    inv = np.where(amax > 0.0, 1.0 / np.maximum(amax, eps), 0.0)
    codes = _round_half_away_from_zero(wg * inv[..., None])
    codes = np.clip(codes, -1, 1).astype(np.int8).reshape(*lead, in_features)
    scales = amax.astype("<f2").astype(np.float32)  # stored as fp16 in the block
    return codes, scales


def pack_q2_0(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack ternary codes + per-group scales into ``Q2_0`` block bytes.

    Args:
        codes: integer array ``(..., in_features)`` with values in ``{-1,0,+1}``
            (``+2`` is also representable but unused by the ternary path).
        scales: float array ``(..., n_groups)`` where
            ``n_groups = in_features // 128``. Leading dims (e.g. MoE experts)
            are preserved.

    Returns:
        ``uint8`` array ``(..., n_groups * 34)`` -- row-major GGUF tensor data.
    """
    *lead, in_features = codes.shape
    if in_features % QK2_0 != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by {QK2_0} for Q2_0"
        )
    n_groups = in_features // QK2_0
    if scales.shape != (*lead, n_groups):
        raise ValueError(f"scales shape {scales.shape} != {(*lead, n_groups)}")

    q = np.clip(codes.astype(np.int32) + 1, 0, 3).astype(np.uint8)
    q = q.reshape(*lead, n_groups, QK2_0 // 4, 4)
    packed = (
        q[..., 0]
        | (q[..., 1] << 2)
        | (q[..., 2] << 4)
        | (q[..., 3] << 6)
    ).astype(np.uint8)  # (..., n_groups, 32)

    d_bytes = _fp16_bytes(scales)  # (..., n_groups, 2)
    block = np.concatenate([d_bytes, packed], axis=-1)  # (..., n_groups, 34)
    return np.ascontiguousarray(block.reshape(*lead, n_groups * BLOCK_SIZE_Q2_0))


def unpack_q2_0(block_bytes: np.ndarray, in_features: int) -> np.ndarray:
    """Inverse of :func:`pack_q2_0`; returns float32 weights ``(rows, in_features)``.

    Mirrors ``dequantize_row_q2_0`` (``y = (q - 1) * d``) for parity checks.
    """
    n_groups = in_features // QK2_0
    *lead, _ = block_bytes.shape
    blk = block_bytes.reshape(*lead, n_groups, BLOCK_SIZE_Q2_0)
    d = np.ascontiguousarray(blk[..., :2]).view("<f2").astype(np.float32)
    d = d.reshape(*lead, n_groups)
    qs = blk[..., 2:]  # (..., n_groups, 32)

    codes = np.empty((*lead, n_groups, QK2_0), dtype=np.int16)
    codes[..., 0::4] = qs & 0x03
    codes[..., 1::4] = (qs >> 2) & 0x03
    codes[..., 2::4] = (qs >> 4) & 0x03
    codes[..., 3::4] = (qs >> 6) & 0x03
    codes -= 1  # q -> code in {-1,0,+1,+2}

    w = codes.astype(np.float32) * d[..., None]
    return w.reshape(*lead, n_groups * QK2_0)


def pack_q1_0(weights: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Pack a weight matrix into ``Q1_0`` (binary, absmean scale) block bytes.

    Args:
        weights: float array ``(..., in_features)`` (leading dims, e.g. MoE
            experts, are preserved; grouping is over the last axis).

    Returns:
        ``uint8`` array ``(..., n_groups * 18)``.
    """
    *lead, in_features = weights.shape
    if in_features % QK1_0 != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by {QK1_0} for Q1_0"
        )
    n_groups = in_features // QK1_0
    wg = weights.reshape(*lead, n_groups, QK1_0).astype(np.float32)
    scales = np.abs(wg).mean(axis=-1).clip(min=eps)  # (..., n_groups)

    sign_bit = (wg >= 0.0).astype(np.uint8)  # 1 where >= 0
    sign_bit = sign_bit.reshape(*lead, n_groups, QK1_0 // 8, 8)
    packed = np.zeros((*lead, n_groups, QK1_0 // 8), dtype=np.uint8)
    for b in range(8):
        packed |= sign_bit[..., b] << b

    d_bytes = _fp16_bytes(scales)  # (..., n_groups, 2)
    block = np.concatenate([d_bytes, packed], axis=-1)  # (..., n_groups, 18)
    return np.ascontiguousarray(block.reshape(*lead, n_groups * BLOCK_SIZE_Q1_0))


def unpack_q1_0(block_bytes: np.ndarray, in_features: int) -> np.ndarray:
    """Inverse of :func:`pack_q1_0`; returns float32 weights ``(rows, in_features)``."""
    n_groups = in_features // QK1_0
    *lead, _ = block_bytes.shape
    blk = block_bytes.reshape(*lead, n_groups, BLOCK_SIZE_Q1_0)
    d = np.ascontiguousarray(blk[..., :2]).view("<f2").astype(np.float32)
    d = d.reshape(*lead, n_groups)
    qs = blk[..., 2:]  # (..., n_groups, 16)

    bits = np.empty((*lead, n_groups, QK1_0), dtype=np.uint8)
    for b in range(8):
        bits[..., b::8] = (qs >> b) & 1
    w = np.where(bits == 1, d[..., None], -d[..., None]).astype(np.float32)
    return w.reshape(*lead, n_groups * QK1_0)
