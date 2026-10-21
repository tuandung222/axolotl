from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel


class QADArgs(BaseModel):
    qad_trainer: bool | None = None
    qad_scales_path: str | None = None
    qad_group_size: int | None = 128
    qad_teacher_path: str | None = None
    qad_kl_alpha: float | None = 1.0
    qad_ce_alpha: float | None = 0.0
    qad_temperature: float | None = 1.0
    qad_chunk_tokens: int | None = 1024
    qad_learn_scale: bool | None = False  # trainable per-group log-multiplier on the init scales (LSQ)
    qad_internal_lora: bool | None = False  # LoRA inside the quantiser: Q(W + BA), merged+rounded at export
    qad_lora_r: int | None = 64
    qad_lora_alpha: float | None = 128.0
    qad_topk: int | None = None  # None = full-vocab KL; k = only the teacher top-k tokens
    qad_topk_mode: str | None = "truncated"  # "truncated" | "renorm"
    qad_freeze_head_and_embed: bool | None = True


@dataclass
class QADTrainingArgsMixin:
    qad_teacher_path: str | None = None
    qad_kl_alpha: float | None = 1.0
    qad_ce_alpha: float | None = 0.0
    qad_temperature: float | None = 1.0
    qad_chunk_tokens: int | None = 1024
    qad_topk: int | None = None
    qad_topk_mode: str | None = "truncated"
