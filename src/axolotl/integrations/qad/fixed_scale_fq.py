"""Fake quantisation on a fixed (PTQ-provided) per-group scale grid, with a straight-through estimator.

Why fixed: torchao QAT recomputes scale = max|w| / 7.5 from the current weights on every forward. A GPTQ
weight sits on the grid q*s with q in [-8, 7]; the group's extreme element is clamped (7.5*s -> 7 or -8), so
max|w_dequant| / 7.5 != s and the recomputed grid differs from the PTQ grid by up to ~7%. Starting QAD from a
PTQ checkpoint with a dynamic scale would therefore silently re-round the weights away from the GPTQ solution.
Here the scales are buffers loaded from the PTQ checkpoint, so fake_quant(W_ptq) == W_ptq exactly at step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QMIN, QMAX = -8, 7  # symmetric int4, same as compressed-tensors W4A16


class _STEGate(torch.autograd.Function):
    """Zero-allocation Straight-Through Estimator gate: forward returns `wq`, backward passes `grad_output` to `w`."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
        return wq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


@torch.no_grad()
def _fake_quant_no_grad(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = w.shape
    wg = w.float().view(out_f, in_f // group_size, group_size)
    s = scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    return (torch.clamp(torch.round(wg / s), QMIN, QMAX) * s).view(out_f, in_f).to(w.dtype)


def fake_quant(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """w: [out, in]; scale: [out, in/group_size]. STE: forward = quantised, backward = identity wrt w."""
    wq = _fake_quant_no_grad(w, scale, group_size)
    if not torch.is_grad_enabled() or not w.requires_grad:
        return wq
    return _STEGate.apply(w, wq)


def fake_quant_lsq(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Like fake_quant but differentiable wrt `scale` (LSQ): round is STE, clamp is clip-STE, so
    d(wq)/d(s) = q - w/s inside the range and q_min/q_max when clipped. Gradient wrt w is 1 inside the range, 0 when clipped."""
    out_f, in_f = w.shape
    s = scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    x = w.float().view(out_f, in_f // group_size, group_size) / s
    x_round = _STEGate.apply(x, torch.round(x)) if (torch.is_grad_enabled() and x.requires_grad) else torch.round(x)
    q = torch.clamp(x_round, QMIN, QMAX)
    return (q * s).view(out_f, in_f).to(w.dtype)


class _FQLoRALinearSTE(torch.autograd.Function):
    """Fused Internal-LoRA Fixed-Scale FakeQuant Linear with rank-r factored backward.

    Eliminates the dense `[out_features, in_features]` weight-gradient GEMM
    `grad_W = grad_Y^T @ X` when the base weight `W` is frozen (`requires_grad=False`)
    and only `lora_A` `[r, in_features]` and `lora_B` `[out_features, r]` are trained.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        bias: torch.Tensor | None,
        fq_scale: torch.Tensor,
        lora_scale: float,
        group_size: int,
    ) -> torch.Tensor:
        delta = (lora_B.float() @ lora_A.float()) * lora_scale
        v = weight.float() + delta
        out_f, in_f = v.shape
        s = fq_scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        vg = v.view(out_f, in_f // group_size, group_size)
        w = (torch.clamp(torch.round(vg / s), QMIN, QMAX) * s).view(out_f, in_f).to(weight.dtype)

        empty = torch.empty(0, device=x.device)
        need_lora_grad = ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
        ctx.save_for_backward(
            x if need_lora_grad else empty,
            w if ctx.needs_input_grad[0] else empty,
            lora_A if ctx.needs_input_grad[3] else empty,
            lora_B if ctx.needs_input_grad[2] else empty,
        )
        ctx.lora_scale = lora_scale
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w, lora_A, lora_B = ctx.saved_tensors
        lora_scale = ctx.lora_scale
        compute_dtype = grad_output.dtype
        grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_x = grad_lora_A = grad_lora_B = grad_bias = None

        if ctx.needs_input_grad[0]:
            w_in = w.to(compute_dtype) if w.dtype != compute_dtype else w
            grad_x = torch.matmul(grad_out_2d, w_in).reshape(grad_output.shape[:-1] + (w_in.shape[1],))

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            x_2d = x.reshape(-1, x.shape[-1])
            if x_2d.dtype != compute_dtype:
                x_2d = x_2d.to(compute_dtype)
            if ctx.needs_input_grad[2]:
                b_cast = lora_B.to(compute_dtype) if lora_B.dtype != compute_dtype else lora_B
                gb = torch.matmul(grad_out_2d, b_cast)
                grad_lora_A = (torch.matmul(gb.t(), x_2d).to(torch.float32) * lora_scale).to(lora_A.dtype)
            if ctx.needs_input_grad[3]:
                a_cast = lora_A.to(compute_dtype) if lora_A.dtype != compute_dtype else lora_A
                xa = torch.matmul(x_2d, a_cast.t())
                grad_lora_B = (torch.matmul(grad_out_2d.t(), xa).to(torch.float32) * lora_scale).to(lora_B.dtype)

        if ctx.has_bias and ctx.needs_input_grad[4]:
            grad_bias = grad_out_2d.sum(dim=0).to(ctx.bias_dtype)

        return grad_x, None, grad_lora_A, grad_lora_B, grad_bias, None, None, None


class FQLinear(nn.Linear):
    """nn.Linear whose weight is fake-quantised per group. state_dict stays a plain Linear's (+ `fq_logs` if scales are learned).

    fixed scale   : scale = fq_scale (buffer, from PTQ / RTN)
    learned scale : scale = fq_scale * exp(fq_logs), fq_logs a parameter initialised at 0 (so step 0 == the init grid and the
                    relative change per optimiser step is ~LR, independent of the absolute scale)."""

    group_size: int = 128
    fq_enabled: bool = True

    def effective_scale(self) -> torch.Tensor:
        if getattr(self, "fq_logs", None) is not None:
            return (self.fq_scale.float() * torch.exp(self.fq_logs.float()))
        return self.fq_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fq_enabled:
            w = self.weight
        elif getattr(self, "fq_logs", None) is not None:
            w = fake_quant_lsq(self.weight, self.effective_scale(), self.group_size)
        else:
            w = fake_quant(self.weight, self.fq_scale, self.group_size)
        return F.linear(x, w, self.bias)


class FQLoRALinear(FQLinear):
    """INTERNAL LoRA: W_eff = Q(W + (alpha/r) * B @ A) on the fixed scale grid (STE). W is frozen; only lora_A / lora_B train.
    At B = 0 this is exactly the RTN/PTQ model. After training the delta is merged and rounded to INT4 codes (export) -> a pure
    INT4 model with no adapter. (EXTERNAL LoRA, by contrast, keeps Wq untouched and adds a separate BF16 branch.)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fq_enabled:
            return F.linear(x, self.weight, self.bias)
        if not self.weight.requires_grad:
            return _FQLoRALinearSTE.apply(
                x,
                self.weight,
                self.lora_A,
                self.lora_B,
                self.bias,
                self.fq_scale,
                float(self.lora_scale),
                self.group_size,
            )
        delta = (self.lora_B.float() @ self.lora_A.float()) * float(self.lora_scale)
        v = self.weight.float() + delta
        w = fake_quant(v, self.fq_scale, self.group_size).to(self.weight.dtype)
        return F.linear(x, w, self.bias)


def add_internal_lora(model: nn.Module, names: list[str], r: int, alpha: float) -> int:
    """Turn the FQLinear modules in `names` into FQLoRALinear and freeze everything except lora_A / lora_B."""
    mods = dict(model.named_modules())
    for n in names:
        m = mods[n]
        assert isinstance(m, FQLinear), n
        m.__class__ = FQLoRALinear
        dev, dt = m.weight.device, m.weight.dtype
        a = torch.empty(r, m.in_features, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(a, a=5**0.5)
        m.lora_A = nn.Parameter(a)
        m.lora_B = nn.Parameter(torch.zeros(m.out_features, r, device=dev, dtype=dt))
        m.register_buffer("lora_scale", torch.tensor(float(alpha) / r), persistent=True)
    for n_, p in model.named_parameters():
        p.requires_grad_(n_.endswith("lora_A") or n_.endswith("lora_B"))
    return len(names)


def apply_fixed_scale_fq(model: nn.Module, scales: dict[str, torch.Tensor], group_size: int = 128, learn_scale: bool = False) -> list[str]:
    """Swap matching nn.Linear modules to FQLinear. `scales` keys are module names (no '.weight_scale' suffix).
    learn_scale=True adds a per-group trainable log-multiplier `fq_logs` (init 0)."""
    done, mods = [], dict(model.named_modules())
    for name, scale in scales.items():
        m = mods.get(name)
        if m is None or not isinstance(m, nn.Linear):
            raise KeyError(f"PTQ scale for '{name}' has no matching nn.Linear in the student")
        if m.in_features % group_size or scale.shape != (m.out_features, m.in_features // group_size):
            raise ValueError(f"{name}: scale {tuple(scale.shape)} vs weight {(m.out_features, m.in_features)} g={group_size}")
        m.__class__ = FQLinear
        m.group_size = group_size
        m.register_buffer("fq_scale", scale.detach().to(device=m.weight.device, dtype=torch.bfloat16), persistent=False)
        if learn_scale:
            m.fq_logs = nn.Parameter(torch.zeros(scale.shape, device=m.weight.device, dtype=torch.float32))
        done.append(name)
    return done


def set_fake_quant(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FQLinear):
            m.fq_enabled = enabled
