"""Plugin config schema for BS ternary QAT."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BSQATArgs(BaseModel):
    bs_qat: bool = Field(
        default=False,
        description="Enable BS ternary fake quant layers.",
    )
    bs_qat_mode: str = Field(
        default="ternary",
        description=(
            "Fake quant mode: ternary, binary, or int4 (symmetric W4A16 g128, "
            "bit-compatible with the compressed-tensors pack-quantized export)."
        ),
    )
    bs_qat_group_size: int = Field(
        default=128,
        description="Group size for quantization.",
    )
    bs_qat_act_quant: bool = Field(
        default=False,
        description="Enable activation quantization.",
    )
    bs_qat_lora_mode: Literal["external_peft", "internal_qat", "full_weight"] = Field(
        default="external_peft",
        description=(
            "LoRA/QAT composition mode. external_peft keeps PEFT LoRA outside "
            "fake quant as QDQ(W) + LoRA (requires `adapter: lora/qlora`). "
            "internal_qat stores LoRA inside BitLinear and fake-quantizes the "
            "merged weight as QDQ(W + LoRA) (requires `adapter:` unset). "
            "full_weight has no LoRA at all -- every BitLinear shadow weight is "
            "directly trainable and fake-quantized as QDQ(W) (also requires "
            "`adapter:` unset). Unlike external_peft, full_weight really means "
            "'no adapter': the two behave identically in this plugin (both patch "
            "BitLinear with use_lora=False) but external_peft additionally "
            "implies Axolotl's PEFT wraps the model, which is misleading to set "
            "in a config that intentionally trains every parameter -- prefer "
            "full_weight there."
        ),
    )
    # These internal-QAT knobs deliberately mirror Axolotl's normal LoRA fields
    # so an existing config can switch composition mode with minimal churn. They
    # are not PEFT settings: in internal_qat mode the tensors live directly on
    # BitLinear and must be saved/loaded by the BS QAT integration.
    bs_qat_lora_r: Optional[int] = Field(
        default=None,
        description="Internal-QAT LoRA rank. Falls back to lora_r when unset.",
    )
    bs_qat_lora_alpha: Optional[int] = Field(
        default=None,
        description="Internal-QAT LoRA alpha. Falls back to lora_alpha when unset.",
    )
    bs_qat_lora_dropout: Optional[float] = Field(
        default=None,
        description="Internal-QAT LoRA dropout. Falls back to lora_dropout when unset.",
    )
    bs_qat_freeze_non_lora: bool = Field(
        default=True,
        description="For internal_qat mode, freeze every parameter except BitLinear lora_A/lora_B.",
    )
    bs_qat_target_names: Optional[list[str]] = Field(
        default=None,
        description="Optional child-module name substrings to patch.",
    )
    bs_qat_skip_names: list[str] = Field(
        default_factory=lambda: ["lm_head"],
        description="Child-module names to leave as nn.Linear.",
    )
