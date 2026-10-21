"""Axolotl trainer: distil a fake-quantised student against a frozen in-process BF16 teacher (full-vocab KL)."""
from __future__ import annotations

import torch
from axolotl.core.trainers.base import AxolotlTrainer
from typing_extensions import override

from .kl import chunked_kl_loss


class AxolotlQADTrainer(AxolotlTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # we normalise by num_items_in_batch ourselves
        self._teacher = None

    def _get_teacher(self):
        if self._teacher is None:
            from transformers import AutoModelForCausalLM

            t = AutoModelForCausalLM.from_pretrained(
                self.args.qad_teacher_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            t.eval().requires_grad_(False)
            self._teacher = t.to(self.args.device)
        return self._teacher

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        inputs.pop("num_items_in_batch", None)
        if num_items_in_batch is None:
            num_items_in_batch = (labels != -100).sum().item()

        base = self.accelerator.unwrap_model(model)
        if hasattr(base, "get_base_model"):  # PEFT-wrapped (LoRA) student
            base = base.get_base_model()
        cap = {}
        hook = base.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
        try:
            outputs = model(**inputs, logits_to_keep=1)  # logits are not used; we need the final-norm hidden
        finally:
            hook.remove()
        hs = cap["h"]

        teacher = self._get_teacher()
        with torch.no_grad():
            ht = teacher.model(**inputs).last_hidden_state

        a = self.args
        total, n = chunked_kl_loss(
            hs, ht, base.lm_head.weight, teacher.lm_head.weight, labels,
            chunk=a.qad_chunk_tokens, temperature=a.qad_temperature, kl_alpha=a.qad_kl_alpha, ce_alpha=a.qad_ce_alpha,
            topk=a.qad_topk, topk_mode=a.qad_topk_mode or "truncated",
        )
        loss = total / max(num_items_in_batch, 1)
        return (loss, outputs) if return_outputs else loss
