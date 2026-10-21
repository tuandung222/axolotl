"""Axolotl plugin: fixed-grid fake quantisation + in-process-teacher distillation.

    plugins:
      - axolotl.integrations.qad.QADPlugin
    qad_trainer: true
    qad_scales_path: artifacts/student_init_4b_r64/qad_scales.safetensors
    qad_teacher_path: outputs/merged/qwen35_4b_v7_r64_untied
"""
from __future__ import annotations

from axolotl.integrations.base import BasePlugin
from .args import QADArgs, QADTrainingArgsMixin


class QADPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.qad.args.QADArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.qad.args.QADTrainingArgsMixin"

    def get_training_args(self, cfg):
        return {
            k: cfg.get(k)
            for k in (
                "qad_teacher_path",
                "qad_kl_alpha",
                "qad_ce_alpha",
                "qad_temperature",
                "qad_chunk_tokens",
                "qad_topk",
                "qad_topk_mode",
            )
        }

    def post_model_load(self, cfg, model):
        """Fake-quant the base Linears unless qad_scales_path is unset (LoRA-on-a-frozen-INT4-base runs: the base is already
        snapped to the INT4 grid, so nothing to fake-quantise and nothing but the adapters trains)."""
        if not cfg.qad_scales_path:
            print("[qad] no qad_scales_path: base used as is (adapter-only run)", flush=True)
            return
        from safetensors.torch import load_file
        from .fixed_scale_fq import apply_fixed_scale_fq

        scales = load_file(cfg.qad_scales_path)
        done = apply_fixed_scale_fq(
            model,
            scales,
            cfg.qad_group_size or 128,
            learn_scale=bool(cfg.qad_learn_scale),
        )
        if cfg.qad_internal_lora:
            from .fixed_scale_fq import add_internal_lora

            n = add_internal_lora(
                model,
                done,
                int(cfg.qad_lora_r or 64),
                float(cfg.qad_lora_alpha or 128.0),
            )
            tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[qad] INTERNAL LoRA on {n} modules; trainable params {tr:,}", flush=True)
            return
        if cfg.qad_freeze_head_and_embed is not False:
            for n, p in model.named_parameters():
                if n.endswith("lm_head.weight") or "embed_tokens" in n:
                    p.requires_grad_(False)
        print(f"[qad] fixed-scale fake-quant on {len(done)} Linear modules", flush=True)

    def get_trainer_cls(self, cfg):
        if cfg.qad_trainer:
            from .trainer import AxolotlQADTrainer

            return AxolotlQADTrainer
        return None
