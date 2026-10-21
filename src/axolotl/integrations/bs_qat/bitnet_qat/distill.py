"""MiniLM-style multi-head attention-relation distillation (BitDistill section 3.3).

Transfers the Q-Q / K-K / V-V self-relations of a full-precision teacher into the
quantized student, on top of the existing logit distillation. This is
**training-only**: nothing here touches the exported weights, the GGUF container,
or inference -- it only shapes the QAT loss.

Reference: BitNet Distillation (``bitnet_distillation.pdf``), Eq. 10-12, following
MiniLM / MiniLMv2 [Wang et al. 2020]. We distill a *single* layer (|Y|=1, the last
decoder layer by default) and split each projection into ``Ar`` relation heads,
defaulting ``Ar = num_key_value_heads`` so the same split divides the Q, K and V
dims under grouped-query attention.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The three projection relations distilled (Phi = {Q, K, V} in the paper).
RELATIONS = ("q", "k", "v")


def get_decoder_layers(model: nn.Module) -> nn.Module:
    """Return the decoder layer list for a HF causal LM.

    Works for the common ``model.model.layers`` layout (Llama, Qwen2/3, Mistral,
    Qwen3-MoE, ...). Raises a clear error otherwise.
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "could not locate decoder layers (expected `model.model.layers`); "
            f"got {type(model).__name__}"
        )
    return layers


def resolve_relation_heads(model: nn.Module, override: int | None = None) -> int:
    """Number of relation heads ``Ar`` (default: model ``num_key_value_heads``)."""
    if override is not None:
        return int(override)
    cfg = getattr(model, "config", None)
    for attr in ("num_key_value_heads", "num_attention_heads"):
        val = getattr(cfg, attr, None)
        if val:
            return int(val)
    raise ValueError("cannot infer num_relation_heads; pass it explicitly")


@dataclass
class QKVCapture:
    """Holds the captured q/k/v projection outputs of one attention layer."""

    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q is None or self.k is None or self.v is None:
            raise RuntimeError(
                "QKV not captured; run the model forward inside `capture_qkv(...)`"
            )
        return self.q, self.k, self.v


@contextmanager
def capture_qkv(model: nn.Module, layer_idx: int = -1):
    """Capture the q/k/v projection outputs of one decoder layer.

    Registers forward hooks on ``layers[layer_idx].self_attn.{q,k,v}_proj`` (works
    for both an FP teacher and a converted ``BitLinear`` student; both project to
    ``(B, T, d)``). The yielded :class:`QKVCapture` is populated once a model
    forward runs within the block; the hooks are always removed on exit.
    """
    layers = get_decoder_layers(model)
    layer = layers[layer_idx]
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise AttributeError("decoder layer has no `self_attn`")

    cap = QKVCapture()
    handles = []

    def _mk(name: str):
        def _hook(_module, _inputs, output):
            setattr(cap, name, output)

        return _hook

    try:
        for name in RELATIONS:
            proj = getattr(attn, f"{name}_proj")
            handles.append(proj.register_forward_hook(_mk(name)))
        yield cap
    finally:
        for h in handles:
            h.remove()


def _relation_scores(a: torch.Tensor, num_relation_heads: int) -> torch.Tensor:
    """Scaled self-relation logits ``A.Aᵀ/sqrt(dr)`` split into ``Ar`` relation heads.

    Args:
        a: projection output, shape ``(B, T, d)``.
        num_relation_heads: ``Ar``; must divide ``d``.

    Returns:
        ``(B, Ar, T, T)`` relation logits (pre-softmax).
    """
    b, t, d = a.shape
    ar = num_relation_heads
    if d % ar != 0:
        raise ValueError(
            f"relation dim {d} not divisible by num_relation_heads={ar}"
        )
    dr = d // ar
    a = a.reshape(b, t, ar, dr).permute(0, 2, 1, 3)  # (B, Ar, T, dr)
    return torch.matmul(a, a.transpose(-1, -2)) / math.sqrt(dr)


def attention_relation_loss(
    teacher_qkv,
    student_qkv,
    num_relation_heads: int,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """MiniLM attention-relations KL between teacher and student (Eq. 10-12).

    For each of Q/K/V build the per-head self-relation distribution
    ``R = softmax(A.Aᵀ/sqrt(dr))`` and accumulate ``KL(R_teacher || R_student)``
    averaged over relation heads, query positions and batch; the three relations
    are summed with ``weights`` (the paper's alpha_i, default 1.0 each). Returns a
    scalar ``float32`` tensor on the student's device.

    ``teacher_qkv`` / ``student_qkv`` may be :class:`QKVCapture` instances or
    ``(q, k, v)`` tuples of ``(B, T, d)`` tensors.
    """
    teacher = teacher_qkv.as_tuple() if isinstance(teacher_qkv, QKVCapture) else teacher_qkv
    student = student_qkv.as_tuple() if isinstance(student_qkv, QKVCapture) else student_qkv

    total: torch.Tensor | None = None
    for w, t_a, s_a in zip(weights, teacher, student):
        if w == 0:
            continue
        # Teacher is detached (fixed target); compute in fp32 for stable softmax/KL.
        t_scores = _relation_scores(t_a.detach().float(), num_relation_heads)
        s_scores = _relation_scores(s_a.float(), num_relation_heads)
        logp_s = F.log_softmax(s_scores, dim=-1)
        p_t = F.softmax(t_scores, dim=-1)
        # KL(R_teacher || R_student): sum over keys, mean over (batch, heads, queries).
        kl = F.kl_div(logp_s, p_t, reduction="none").sum(dim=-1).mean()
        term = w * kl
        total = term if total is None else total + term

    if total is None:
        return student[0].new_zeros((), dtype=torch.float32)
    return total
