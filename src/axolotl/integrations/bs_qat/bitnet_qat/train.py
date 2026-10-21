"""Quantization-aware training with optional logit distillation.

Supports the BitCPM-style **two-stage decoupled** recipe:

  * **qat** stage   -- pure cross-entropy LM training (``distill_alpha=0``) so
    the quantized (binary/ternary) student first adapts on the task;
  * **distill** stage -- ``alpha * KL(student||teacher, T) * T^2 + (1-alpha) * CE``
    to transfer the teacher's distribution into the quantized student.

A single combined stage is still supported (legacy flat ``train`` config). The
student is a model whose target ``nn.Linear`` layers have been swapped for
:class:`~bitnet_qat.bitlinear.BitLinear` (see :mod:`bitnet_qat.convert`); gradients
flow to the full-precision shadow weights via the straight-through estimator.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# Force the torch backend before importing transformers (avoids TF/Keras import).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from .distill import attention_relation_loss, capture_qkv, resolve_relation_heads

logger = logging.getLogger(__name__)


# Stages run in this canonical order regardless of YAML order: an optional
# general-corpus warm-up (BitDistill section 3.2) -> pure QAT -> distillation.
STAGE_ORDER = ["pretrain", "qat", "distill"]


@dataclass
class TrainConfig:
    teacher: str | None = None
    teacher_load_in_4bit: bool = False
    dataset: str = ""
    dataset_split: str = "train"
    text_field: str = "text"
    # Optional list of dataset specs to mix (concat + shuffle); when set it
    # overrides the single dataset/dataset_split/text_field above. See
    # build_hf_dataset / _render_dataset_spec for the per-spec schema.
    datasets: list | None = None
    max_seq_len: int = 2048
    micro_batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2.0e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str = "adamw_torch"   # e.g. "paged_adamw_8bit" to shrink optimizer state
    max_steps: int = 1000
    bf16: bool = True
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0
    use_cpu: bool = False
    distill_alpha: float = 0.9
    distill_temperature: float = 2.0
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # MiniLM attention-relation distillation (BitDistill section 3.3); off by
    # default so existing recipes are unchanged. Active only in a distill stage.
    attn_distill: bool = False
    attn_distill_layer: int = -1
    attn_distill_weight: float = 1.0
    attn_distill_relation_heads: int | None = None
    save_dir: str = "qat/runs/ternary"
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 0
    stage: str = "train"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_stage_configs(train: dict) -> list[tuple[str, TrainConfig]]:
    """Return ordered ``(stage_name, TrainConfig)`` from a ``train`` config dict.

    If ``train['stages']`` exists, each stage inherits the top-level ``train``
    keys and overrides them with its own; stages run in canonical order
    ``pretrain`` -> ``qat`` -> ``distill`` (any other names follow, in definition
    order). A ``pretrain`` (warm-up) or ``qat`` stage defaults to pure
    cross-entropy (``distill_alpha=0``) unless set explicitly. Without a
    ``stages`` block a single legacy stage named ``"train"`` is built.
    """
    stages = train.get("stages")
    if not stages:
        cfg = TrainConfig.from_dict(train)
        cfg.stage = "train"
        return [("train", cfg)]

    shared = {k: v for k, v in train.items() if k != "stages"}
    ordered = [s for s in STAGE_ORDER if s in stages]
    ordered += [s for s in stages if s not in STAGE_ORDER]

    out: list[tuple[str, TrainConfig]] = []
    for name in ordered:
        overrides = stages[name] or {}
        cfg = TrainConfig.from_dict({**shared, **overrides})
        cfg.stage = name
        if name in ("pretrain", "qat") and "distill_alpha" not in overrides and "distill_alpha" not in shared:
            cfg.distill_alpha = 0.0  # warm-up / QAT default to pure cross-entropy
        out.append((name, cfg))
    return out


def load_teacher(name: str, dtype, load_in_4bit: bool = False, trust_remote_code: bool = False):
    """Load a frozen teacher; optionally as a 4-bit (bitsandbytes) model."""
    from transformers import AutoModelForCausalLM

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            name, quantization_config=quant, device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, trust_remote_code=trust_remote_code
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    ce_loss: torch.Tensor,
    alpha: float,
    temperature: float,
    mask: torch.Tensor | None = None,
    parts: dict | None = None,
) -> torch.Tensor:
    """KL(student||teacher) at temperature T blended with the label CE loss.

    The KL is reduced as a **per-token mean** (sum over the vocab, averaged over
    the real, non-padding tokens) so it shares the same scale as the per-token
    ``ce_loss`` and the attention-relation term. Using ``reduction="batchmean"``
    on a ``[B, S, V]`` tensor would divide only by ``B`` and thus *sum* the KL
    across the ``S`` sequence positions, inflating it by ~``S`` and drowning out
    the other loss terms.
    """
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_p = F.softmax(teacher_logits / t, dim=-1)
    kl_tok = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1, dtype=torch.float32)
    if mask is not None:
        m = mask.to(kl_tok.dtype)
        kl_nats = (kl_tok * m).sum() / m.sum().clamp(min=1.0)
    else:
        kl_nats = kl_tok.mean()
    kl = (kl_nats * (t * t)).to(ce_loss.dtype)
    if parts is not None:
        parts["kl_term"] = float((alpha * kl).detach())
        parts["ce_term"] = float(((1.0 - alpha) * ce_loss).detach())
        parts["kl_nats"] = float(kl_nats.detach())
    return alpha * kl + (1.0 - alpha) * ce_loss


class DistillTrainer(Trainer):
    """HF ``Trainer`` that adds logit distillation from a frozen teacher."""

    def __init__(
        self,
        *args,
        teacher=None,
        alpha: float = 0.9,
        temperature: float = 2.0,
        attn_distill: bool = False,
        attn_distill_layer: int = -1,
        attn_distill_weight: float = 1.0,
        attn_distill_relation_heads: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.attn_distill = attn_distill
        self.attn_distill_layer = attn_distill_layer
        self.attn_distill_weight = attn_distill_weight
        self.attn_distill_relation_heads = attn_distill_relation_heads
        if self.teacher is not None:
            self.teacher.to(self.args.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        use_attn = self.attn_distill and self.teacher is not None
        need_teacher = self.teacher is not None and (self.alpha > 0.0 or use_attn)

        # Student forward (capturing q/k/v of one layer when attention-distilling).
        if use_attn:
            with capture_qkv(model, self.attn_distill_layer) as student_qkv:
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        ce = outputs.loss
        loss = ce
        parts: dict[str, float] = {}

        if need_teacher:
            # One frozen-teacher forward, reused for both logit KL and QKV capture.
            if use_attn:
                with torch.no_grad(), capture_qkv(
                    self.teacher, self.attn_distill_layer
                ) as teacher_qkv:
                    t_out = self.teacher(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
            else:
                with torch.no_grad():
                    t_out = self.teacher(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )

            if self.alpha > 0.0:
                loss = distillation_loss(
                    outputs.logits, t_out.logits, ce, self.alpha, self.temperature,
                    mask=inputs.get("attention_mask"), parts=parts,
                )
            if use_attn:
                ar = resolve_relation_heads(model, self.attn_distill_relation_heads)
                lad = attention_relation_loss(teacher_qkv, student_qkv, ar)
                loss = loss + self.attn_distill_weight * lad.to(loss.dtype)
                parts["attn_term"] = float((self.attn_distill_weight * lad).detach())

        self._last_parts = parts
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        parts = getattr(self, "_last_parts", None)
        if parts:
            for k, v in parts.items():
                logs.setdefault(k, round(v, 4))
        return super().log(logs, *args, **kwargs)


def tokenize_texts(tokenizer, ds, max_seq_len: int):
    """Tokenize a causal-LM dataset.

    No padding and no ``labels`` are added here: the data collator
    (:func:`_default_collator`) pads each batch dynamically to its longest
    sequence and builds ``labels`` with padding positions masked to ``-100``.
    This keeps the CE loss over real tokens only and avoids paying for a full
    ``max_seq_len`` forward on short samples.
    """
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def _render_dataset_spec(tokenizer, spec: dict, default_split: str):
    """Load one HF dataset spec and render it to a single ``text`` column.

    A spec is a dict with ``name`` and optional ``split`` / ``config``. Rendering:
      * ``prompt_field`` (+ optional ``response_field`` / ``system``) -> a
        user/assistant chat pair so Q/A corpora (GSM8K, MetaMathQA) teach the
        worked answer;
      * ``text_field`` holding a chat ``messages`` list -> chat template;
      * ``text_field`` holding a plain string -> used verbatim.

    For web-scale corpora (FineWeb-Edu, FineWeb-2, Wikipedia, ...) set
    ``streaming: true`` with ``max_samples: N`` so only the first ``N`` examples
    are pulled (via ``IterableDataset.take``) instead of downloading the entire
    split just to slice it. This is what makes broad knowledge-preserving KD
    affordable.
    """
    from datasets import Dataset, load_dataset, concatenate_datasets

    name = spec["name"]
    split = spec.get("split", default_split)
    config = spec.get("config")
    streaming = bool(spec.get("streaming", False))
    max_samples = spec.get("max_samples")

    data_files = spec.get("data_files")
    load_kwargs = {}
    if data_files:
        load_kwargs["data_files"] = data_files

    if streaming:
        raw = (
            load_dataset(name, config, split=split, streaming=True, **load_kwargs)
            if config
            else load_dataset(name, split=split, streaming=True, **load_kwargs)
        )
        if max_samples:
            raw = raw.take(int(max_samples))
    else:
        raw = load_dataset(name, config, split=split, **load_kwargs) if config else load_dataset(name, split=split, **load_kwargs)

    prompt_field = spec.get("prompt_field")
    response_field = spec.get("response_field")
    text_field = spec.get("text_field", "text")
    system = spec.get("system")

    def _to_text(example):
        if prompt_field:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": str(example[prompt_field])})
            if response_field:
                messages.append({"role": "assistant", "content": str(example[response_field])})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            value = example[text_field]
            if isinstance(value, str):
                try:
                    import json
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        value = parsed
                except:
                    pass
            if isinstance(value, list):  # chat messages
                text = tokenizer.apply_chat_template(
                    value, tokenize=False, add_generation_prompt=False
                )
            else:
                text = value
        return {"text": text}

    if streaming:
        # IterableDataset has no random access / column materialisation, so pull
        # the (bounded) stream into an in-memory Dataset of rendered text.
        texts = [_to_text(ex)["text"] for ex in raw]
        if max_samples and len(texts) < int(max_samples) and len(texts) > 0:
            import math
            repeats = math.ceil(int(max_samples) / len(texts))
            texts = (texts * repeats)[:int(max_samples)]
        return Dataset.from_dict({"text": texts})

    out = raw.map(_to_text, remove_columns=raw.column_names)
    if max_samples:
        if len(out) < int(max_samples) and len(out) > 0:
            import math
            repeats = math.ceil(int(max_samples) / len(out))
            out = concatenate_datasets([out] * repeats)
        out = out.select(range(int(max_samples)))
    return out


def build_hf_dataset(tokenizer, cfg: TrainConfig):
    """Load + render + tokenize one or more HF datasets for QAT.

    With ``cfg.datasets`` (a list of specs) the corpora are rendered, concatenated
    and shuffled into a single mix (e.g. chat + math + reasoning); see
    :func:`_render_dataset_spec`. Without it the legacy single-dataset path
    (``cfg.dataset`` / ``cfg.dataset_split`` / ``cfg.text_field``) is used.
    """
    from datasets import concatenate_datasets

    specs = cfg.datasets or [
        {"name": cfg.dataset, "split": cfg.dataset_split, "text_field": cfg.text_field}
    ]
    rendered = [_render_dataset_spec(tokenizer, s, cfg.dataset_split) for s in specs]
    merged = rendered[0] if len(rendered) == 1 else concatenate_datasets(rendered)
    if len(rendered) > 1:
        merged = merged.shuffle(seed=cfg.seed)
    return tokenize_texts(tokenizer, merged, cfg.max_seq_len)


def run_training_stage(
    student, teacher, tokenized_ds, tokenizer, cfg: TrainConfig,
    resume_from_checkpoint: str | None = None,
) -> DistillTrainer:
    """Run one training stage (pure QAT or distillation). Returns the trainer."""
    logger.info(
        "Stage %s: steps=%d lr=%g distill_alpha=%g attn_distill=%s teacher=%s",
        cfg.stage, cfg.max_steps, cfg.lr, cfg.distill_alpha,
        cfg.attn_distill, "yes" if teacher is not None else "no",
    )
    if cfg.grad_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(student, "config"):
            student.config.use_cache = False

    args = TrainingArguments(
        output_dir=cfg.save_dir,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        max_steps=cfg.max_steps,
        bf16=cfg.bf16,
        use_cpu=cfg.use_cpu,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=4,
    )

    trainer = DistillTrainer(
        model=student,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=_default_collator(tokenizer),
        teacher=teacher,
        alpha=cfg.distill_alpha,
        temperature=cfg.distill_temperature,
        attn_distill=cfg.attn_distill,
        attn_distill_layer=cfg.attn_distill_layer,
        attn_distill_weight=cfg.attn_distill_weight,
        attn_distill_relation_heads=cfg.attn_distill_relation_heads,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


# Backward-compatible alias (single-stage entry point).
train_qat = run_training_stage


def _default_collator(tokenizer):
    """Dynamic-padding causal-LM collator with strict SFT masking.

    Pads each batch to its own longest sequence and applies SFT masking:
    - Masks all system prompts and user prompts (setting labels to -100).
    - Calculates loss ONLY on the assistant's responses and the <|im_end|> token.
    - Excludes trailing padding positions from the loss.
    """
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_len = len(header)
    end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    def collate(features: list[dict]):
        batch = tokenizer.pad(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        
        # Apply SFT masking: only calculate loss for assistant responses
        for b_idx in range(len(labels)):
            ids = labels[b_idx].tolist()
            mask = True
            i = 0
            while i < len(ids):
                if mask:
                    if i + header_len <= len(ids) and ids[i:i+header_len] == header:
                        # Found assistant header! Mask the header itself.
                        for j in range(i, i + header_len):
                            labels[b_idx, j] = -100
                        i += header_len
                        mask = False  # Start unmasking (learning) the response
                    else:
                        labels[b_idx, i] = -100
                        i += 1
                else:
                    if ids[i] == end_token:
                        # Keep the end token unmasked, then mask the rest
                        i += 1
                        mask = True
                    else:
                        # Keep unmasked
                        i += 1

        # Also mask padding tokens
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collate
