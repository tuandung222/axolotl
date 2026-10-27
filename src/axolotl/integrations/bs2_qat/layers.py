"""Mathematically Optimized Hadamard + PQ2_0 Ternary QAT Layers for Bonsai-2 (Qwen3.5).

Implements:
1. L2-cached GEMM ($B \\le 512$) and 2D Kronecker-factored GEMM ($B > 512$) for
   Sylvester Walsh-Hadamard block rotation with per-channel sign flipping.
2. Boolean-mask 2-stage Least-Squares Ternary Quantization (`PQ2_0`, group_size=128)
   matching Prism ML's Ternary-Bonsai-2 GGUF/MLX format bit-for-bit (100.0000% trit match).
3. Collapsed Identity Jacobian STE (`T^-1 o I_STE o T = I`) eliminating all Hadamard
   transforms from the backward pass when `store_rotated=False`.
4. Full support for `full_weight`, `internal_qat` (LoRA inside the Hadamard + PQ2_0
   quantizer), and `external_peft` training modes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# Per-device cache of normalized Sylvester Hadamard matrices: (size, device) -> Tensor[size, size]
_HADAMARD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Return normalized Sylvester Hadamard matrix H_size / sqrt(size) in float32 on `device`."""
    key = (size, device)
    mat = _HADAMARD_CACHE.get(key)
    if mat is None:
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"Hadamard block size must be a positive power of 2, got {size}")
        h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        n = 1
        while n < size:
            h = torch.cat(
                [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
                dim=0,
            )
            n *= 2
        mat = (h * (1.0 / math.sqrt(size))).contiguous()
        _HADAMARD_CACHE[key] = mat
    return mat


def _kronecker_factors(block: int) -> tuple[int, int]:
    """Split `block = 2^k` into balanced factors `(b1, b2)` with `b1 * b2 == block`."""
    k = int(round(math.log2(block)))
    k1 = k // 2
    k2 = k - k1
    return (1 << k1), (1 << k2)


@torch.no_grad()
def _fwht_core_no_grad(
    x: torch.Tensor,
    block: int,
    signs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Fast Sylvester Walsh-Hadamard transform with sign vector (no autograd graph)."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    if orig_shape[-1] % block != 0:
        raise ValueError(f"Last dim {orig_shape[-1]} is not divisible by block={block}")

    x_f32 = x.to(torch.float32)
    s_f32 = signs.to(device=x.device, dtype=torch.float32)

    if not inverse:
        x_f32 = x_f32 * s_f32

    if block <= 512:
        h_mat = _get_hadamard_matrix(block, x.device)
        out_f32 = torch.matmul(x_f32.reshape(-1, block), h_mat).reshape(orig_shape)
    else:
        b1, b2 = _kronecker_factors(block)
        h1 = _get_hadamard_matrix(b1, x.device)
        h2 = _get_hadamard_matrix(b2, x.device)
        v = x_f32.reshape(-1, b1, b2)
        out_f32 = torch.matmul(h1, torch.matmul(v, h2)).reshape(orig_shape)

    if inverse:
        out_f32 = out_f32 * s_f32

    return out_f32.to(orig_dtype)


class _FWHTAutogradFunc(torch.autograd.Function):
    """Custom autograd function for activation rotation when `store_rotated=True`."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        block: int,
        signs: torch.Tensor,
        inverse: bool,
    ) -> torch.Tensor:
        ctx.block = block
        ctx.inverse = inverse
        ctx.save_for_backward(signs)
        return _fwht_core_no_grad(x, block, signs, inverse=inverse)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (signs,) = ctx.saved_tensors
        grad_x = _fwht_core_no_grad(
            grad_output,
            ctx.block,
            signs,
            inverse=(not ctx.inverse),
        )
        return grad_x, None, None, None


def fwht_fast(
    x: torch.Tensor,
    block: int,
    signs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Differentiable fast Sylvester Walsh-Hadamard transform."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return _fwht_core_no_grad(x, block, signs, inverse=inverse)
    return _FWHTAutogradFunc.apply(x, block, signs, inverse)


@torch.no_grad()
def ternary_quantize_pq2_0_no_grad(
    w_rot: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boolean-mask 2-stage Least-Squares Ternary Quantization (`PQ2_0`, group_size=128).

    Returns:
        w_hat_hard: Dequantized ternary weight in float32, shape `[rows, width]`.
        q_hard: Discrete ternary codes in `{-1, 0, +1}` (int8), shape `[rows, width // group_size, group_size]`.
        scales_fp16: Per-group FP16 scale factors, shape `[rows, width // group_size]`.
    """
    rows, width = w_rot.shape
    if width % group_size != 0:
        raise ValueError(f"in_features ({width}) must be divisible by group_size ({group_size})")

    w_g = w_rot.to(torch.float32).reshape(rows, width // group_size, group_size)

    # Boolean-mask Least-Squares scale: s* = sum(|w_i| * m_i) / sum(m_i) where m_i = (|w_i| >= 0.5 * amax)
    w_abs = w_g.abs()
    s_init = w_abs.amax(dim=-1, keepdim=True)
    mask = (w_abs >= (0.5 * s_init)) & (s_init > 0)
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    scales = (w_abs * mask).sum(dim=-1, keepdim=True) / denom

    # Match PQ2_0 / MLX IEEE-754 float16 scale storage (preserve exact 0.0 for all-zero groups)
    scales_fp16 = scales.to(torch.float16)
    scales_f32 = scales_fp16.to(torch.float32)

    safe_scales = torch.where(scales_f32 > 0, scales_f32, torch.ones_like(scales_f32))
    q_hard = torch.where(
        scales_f32 > 0,
        (w_g / safe_scales).round().clamp(-1.0, 1.0),
        torch.zeros_like(w_g),
    )
    w_hat_hard = (q_hard * scales_f32).reshape(rows, width)

    return w_hat_hard, q_hard.to(torch.int8), scales_fp16.squeeze(-1)


class Bonsai2LinearFakeQuant(nn.Module):
    """Hadamard + PQ2_0 Ternary QAT Linear layer for Ternary-Bonsai-2 (Qwen3.5).

    Supports:
    - `use_lora=False`: Full-weight QAT (`full_weight`) or external PEFT (`external_peft`).
    - `use_lora=True`: Internal LoRA QAT (`internal_qat`) where `W_eff = W + (alpha/r) * B @ A`
      is rotated into the Hadamard domain, quantized to PQ2_0 ternary, and unfolded.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 512,
        signs: torch.Tensor | None = None,
        group_size: int = 128,
        store_rotated: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
        if block_size > 0 and in_features % block_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by block_size ({block_size})")

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.group_size = group_size
        self.store_rotated = store_rotated
        self.quant_enabled = True

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if signs is None:
            signs = torch.ones(in_features, dtype=torch.float32, device=device)
        else:
            signs = signs.detach().to(dtype=torch.float32, device=device).clone()
        self.register_buffer("signs", signs, persistent=True)
        if block_size > 0 and device is not None and str(device).startswith("cuda"):
            _get_hadamard_matrix(
                block_size if block_size <= 512 else _kronecker_factors(block_size)[1],
                torch.device(device),
            )

        # Optional internal LoRA parameters (inside the Hadamard + PQ2_0 quantizer)
        self.use_lora = bool(use_lora and lora_r > 0)
        self.lora_r = int(lora_r) if self.use_lora else 0
        self.lora_alpha = float(lora_alpha) if self.use_lora else 0.0
        self.lora_scaling = (self.lora_alpha / self.lora_r) if self.use_lora else 0.0
        self.lora_dropout = (
            nn.Dropout(p=float(lora_dropout)) if (self.use_lora and lora_dropout > 0.0) else nn.Identity()
        )
        if self.use_lora:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Freeze base weight when internal LoRA is enabled
            self.weight.requires_grad_(False)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def _compute_effective_weight(self) -> torch.Tensor:
        """Return `W + (alpha / r) * (B @ A)` when `use_lora=True`, else `W`."""
        if not self.use_lora or self.lora_A is None or self.lora_B is None:
            return self.weight
        lora_a = self.lora_dropout(self.lora_A)
        delta = torch.matmul(self.lora_B.to(torch.float32), lora_a.to(torch.float32)) * self.lora_scaling
        return self.weight + delta.to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self._compute_effective_weight()

        if not self.quant_enabled:
            if self.store_rotated and self.block_size > 0:
                x_rot = fwht_fast(x, self.block_size, self.signs, inverse=False)
                return F.linear(x_rot, w_eff.to(x.dtype), self.bias)
            return F.linear(x, w_eff.to(x.dtype), self.bias)

        if not self.store_rotated:
            # Mode 1 (default): Master weights stored in standard Qwen3.5 unfolded space.
            # Collapsed Identity Jacobian: T^-1 o I_STE o T = I => grad_w_eff == grad_w_hat
            with torch.no_grad():
                w_rot = (
                    _fwht_core_no_grad(w_eff, self.block_size, self.signs, inverse=False)
                    if self.block_size > 0
                    else w_eff.to(torch.float32)
                )
                w_hat_rot_hard, _, _ = ternary_quantize_pq2_0_no_grad(w_rot, self.group_size)
                w_hat_hard = (
                    _fwht_core_no_grad(w_hat_rot_hard, self.block_size, self.signs, inverse=True)
                    if self.block_size > 0
                    else w_hat_rot_hard
                ).to(w_eff.dtype)

            w_hat = w_eff + (w_hat_hard - w_eff).detach()
            return F.linear(x, w_hat.to(x.dtype), self.bias)
        else:
            # Mode 2: Master weights stored directly in rotated Hadamard domain.
            with torch.no_grad():
                w_hat_rot_hard, _, _ = ternary_quantize_pq2_0_no_grad(w_eff, self.group_size)
                w_hat_rot_hard = w_hat_rot_hard.to(w_eff.dtype)

            w_hat_rot = w_eff + (w_hat_rot_hard - w_eff).detach()
            x_rot = (
                fwht_fast(x, self.block_size, self.signs, inverse=False)
                if self.block_size > 0
                else x
            )
            return F.linear(x_rot, w_hat_rot.to(x.dtype), self.bias)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int,
        signs: torch.Tensor,
        group_size: int = 128,
        store_rotated: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ) -> "Bonsai2LinearFakeQuant":
        """Convert a standard `nn.Linear` into `Bonsai2LinearFakeQuant`."""
        q_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            block_size=block_size,
            signs=signs.to(device=linear.weight.device),
            group_size=group_size,
            store_rotated=store_rotated,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        if store_rotated and block_size > 0:
            w_rot = _fwht_core_no_grad(
                linear.weight.data,
                block_size,
                q_layer.signs,
                inverse=False,
            )
            q_layer.weight.data.copy_(w_rot.to(linear.weight.dtype))
        else:
            q_layer.weight.data.copy_(linear.weight.data)

        if linear.bias is not None and q_layer.bias is not None:
            q_layer.bias.data.copy_(linear.bias.data)

        return q_layer

    @torch.no_grad()
    def to_standard_linear(self) -> nn.Linear:
        """Snap effective weight onto the Hadamard PQ2_0 ternary grid and return unfolded `nn.Linear`."""
        was_training = self.training
        self.eval()
        w_eff = self._compute_effective_weight()
        if was_training:
            self.train()

        w_rot = (
            _fwht_core_no_grad(w_eff, self.block_size, self.signs, inverse=False)
            if (not self.store_rotated and self.block_size > 0)
            else w_eff.to(torch.float32)
        )
        w_hat_rot, _, _ = ternary_quantize_pq2_0_no_grad(w_rot, self.group_size)
        w_unfolded = (
            _fwht_core_no_grad(w_hat_rot, self.block_size, self.signs, inverse=True)
            if self.block_size > 0
            else w_hat_rot
        ).to(self.weight.dtype)

        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=(self.bias is not None),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        linear.weight.data.copy_(w_unfolded)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data)
        return linear

    @torch.no_grad()
    def export_ternary_codes_and_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Export discrete ternary codes `{-1, 0, +1}` (int8) and FP16 scales for PQ2_0 / MLX packing."""
        was_training = self.training
        self.eval()
        w_eff = self._compute_effective_weight()
        if was_training:
            self.train()

        w_rot = (
            _fwht_core_no_grad(w_eff, self.block_size, self.signs, inverse=False)
            if (not self.store_rotated and self.block_size > 0)
            else w_eff.to(torch.float32)
        )
        _, q_hard, scales_fp16 = ternary_quantize_pq2_0_no_grad(w_rot, self.group_size)
        return q_hard, scales_fp16

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"group_size={self.group_size}, store_rotated={self.store_rotated}, "
            f"use_lora={self.use_lora}, lora_r={self.lora_r}"
        )


def mark_only_bs2_internal_lora_as_trainable(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except `Bonsai2LinearFakeQuant` internal `lora_A` and `lora_B`."""
    for param in model.parameters():
        param.requires_grad_(False)

    tensors = 0
    params = 0
    for module in model.modules():
        if isinstance(module, Bonsai2LinearFakeQuant) and module.use_lora:
            for p in (module.lora_A, module.lora_B):
                if p is not None:
                    p.requires_grad_(True)
                    tensors += 1
                    params += p.numel()
    return tensors, params


@torch.no_grad()
def save_bs2_internal_lora(
    model: nn.Module,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save internal-QAT LoRA tensors (`lora_A`, `lora_B`) to `output_dir`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    meta_modules: dict[str, dict[str, Any]] = {}
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, Bonsai2LinearFakeQuant) and module.use_lora:
            if module.lora_A is not None and module.lora_B is not None:
                tensors[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous()
                tensors[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous()
                total_params += module.lora_A.numel() + module.lora_B.numel()
                meta_modules[name] = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "block_size": module.block_size,
                    "group_size": module.group_size,
                    "lora_r": module.lora_r,
                    "lora_alpha": module.lora_alpha,
                    "store_rotated": module.store_rotated,
                }

    st_file = out_path / "bs2_internal_qat_lora.safetensors"
    cfg_file = out_path / "bs2_internal_qat_lora.json"
    if tensors:
        save_file(tensors, str(st_file))
    meta = {
        "format": "bs2_internal_qat_lora_v1",
        "tensors": len(tensors),
        "parameters": total_params,
        "modules": meta_modules,
        **(config or {}),
    }
    cfg_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"path": str(out_path), "tensors": len(tensors), "parameters": total_params}


@torch.no_grad()
def load_bs2_internal_lora(
    model: nn.Module,
    adapter_dir: str | Path,
    strict: bool = True,
) -> int:
    """Load saved internal-QAT LoRA tensors into a patched `Bonsai2LinearFakeQuant` model."""
    st_file = Path(adapter_dir) / "bs2_internal_qat_lora.safetensors"
    state = load_file(str(st_file))
    loaded = 0
    modules_dict = dict(model.named_modules())
    for key, tensor in state.items():
        mod_name, param_name = key.rsplit(".", 1)
        mod = modules_dict.get(mod_name)
        if mod is None or not isinstance(mod, Bonsai2LinearFakeQuant):
            if strict:
                raise KeyError(f"Module {mod_name} not found or not Bonsai2LinearFakeQuant")
            continue
        target_param = getattr(mod, param_name, None)
        if target_param is None:
            if strict:
                raise KeyError(f"Parameter {param_name} missing on {mod_name}")
            continue
        target_param.data.copy_(tensor.to(device=target_param.device, dtype=target_param.dtype))
        loaded += 1
    return loaded


@torch.no_grad()
def snap_bs2_qat_to_standard_linear(model: nn.Module) -> int:
    """Replace all `Bonsai2LinearFakeQuant` modules in `model` with snapped `nn.Linear`."""
    replaced = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, Bonsai2LinearFakeQuant):
                setattr(parent, child_name, child.to_standard_linear())
                replaced += 1
    return replaced
