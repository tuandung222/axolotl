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


def fake_quant(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """w: [out, in]; scale: [out, in/group_size]. STE: forward = quantised, backward = identity wrt w."""
    out_f, in_f = w.shape
    wg = w.float().view(out_f, in_f // group_size, group_size)
    s = scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    wq = (torch.clamp(torch.round(wg / s), QMIN, QMAX) * s).view(out_f, in_f).to(w.dtype)
    return w + (wq - w).detach()


def fake_quant_lsq(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Like fake_quant but differentiable wrt `scale` (LSQ): round is STE, clamp is clip-STE, so
    d(wq)/d(s) = q - w/s inside the range and q_min/q_max when clipped. Gradient wrt w is 1 inside the range, 0 when clipped."""
    out_f, in_f = w.shape
    s = scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    x = w.float().view(out_f, in_f // group_size, group_size) / s
    q = torch.clamp(x + (torch.round(x) - x).detach(), QMIN, QMAX)
    return (q * s).view(out_f, in_f).to(w.dtype)


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
        delta = (self.lora_B.float() @ self.lora_A.float()) * float(self.lora_scale)
        v = self.weight.float() + delta
        out_f, in_f = v.shape
        s = self.fq_scale.float().unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        vg = v.view(out_f, in_f // self.group_size, self.group_size)
        q = torch.clamp(torch.round(vg / s), QMIN, QMAX) * s
        w = (vg + (q - vg).detach()).view(out_f, in_f).to(self.weight.dtype)  # STE on the merged weight
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
        m.register_buffer("fq_scale", scale.to(device=m.weight.device, dtype=torch.bfloat16), persistent=False)
        if learn_scale:
            m.fq_logs = nn.Parameter(torch.zeros(scale.shape, device=m.weight.device, dtype=torch.float32))
        done.append(name)
    return done


def set_fake_quant(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FQLinear):
            m.fq_enabled = enabled
