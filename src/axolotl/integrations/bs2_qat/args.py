"""Plugin configuration schema for Bonsai-2 (BS2) Hadamard Ternary QAT."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_BS2_TARGET_NAMES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_BS2_SKIP_NAMES: list[str] = [
    "lm_head",
    "in_proj_a",
    "in_proj_b",
]


class BS2QATArgs(BaseModel):
    """Configuration arguments for the Bonsai-2 Hadamard Ternary (PQ2_0) QAT plugin."""

    bs2_qat: bool = Field(
        default=False,
        description="Enable Bonsai-2 Sylvester Walsh-Hadamard + PQ2_0 Ternary QAT layers.",
    )
    bs2_qat_group_size: int = Field(
        default=128,
        description="Group size for PQ2_0 Least-Squares ternary quantization (default: 128).",
    )
    bs2_qat_block_size: Optional[int] = Field(
        default=None,
        description=(
            "Sylvester Walsh-Hadamard block size B (power of 2, e.g. 512, 1024, 4096). "
            "When None, automatically inferred from the model hidden_size "
            "(2B/4B -> 512, 9B -> 4096, 27B -> 1024)."
        ),
    )
    bs2_qat_signs_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to hadamard.json, MLX model directory, or GGUF file containing "
            "explicit Hadamard sign vectors. When None or 'builtin', uses the built-in "
            "canonical Ternary-Bonsai-2-{2B,4B,9B,27B} sign vectors."
        ),
    )
    bs2_qat_signs_seed: int = Field(
        default=0,
        description="Deterministic fallback RNG seed for generating +/-1 signs on custom widths.",
    )
    bs2_qat_store_rotated: bool = Field(
        default=False,
        description=(
            "If False (default), master weights are stored in standard unfolded Qwen3.5 space "
            "(compatible with tie_word_embeddings=True and zero-Hadamard backward pass via "
            "collapsed Identity Jacobian). If True, master weights are stored in the rotated "
            "Hadamard domain and only input activations are rotated during forward."
        ),
    )
    bs2_qat_lora_mode: Literal["full_weight", "internal_qat", "external_peft"] = Field(
        default="full_weight",
        description=(
            "QAT/LoRA composition mode:\n"
            "- 'full_weight': Full-parameter QAT on Bonsai2LinearFakeQuant.weight "
            "(requires `adapter:` unset).\n"
            "- 'internal_qat': Freezes base weights and trains internal lora_A/lora_B "
            "inside the Hadamard + PQ2_0 quantizer as T^-1(Q(T(W + (alpha/r)*B*A))) "
            "(requires `adapter:` unset).\n"
            "- 'external_peft': Keeps PEFT LoRA outside the Hadamard fake-quantized layer "
            "(requires `adapter: lora`)."
        ),
    )
    bs2_qat_lora_r: Optional[int] = Field(
        default=None,
        description="Internal-QAT LoRA rank. Falls back to lora_r (or 16) when unset.",
    )
    bs2_qat_lora_alpha: Optional[float] = Field(
        default=None,
        description="Internal-QAT LoRA alpha. Falls back to lora_alpha (or 32.0) when unset.",
    )
    bs2_qat_lora_dropout: Optional[float] = Field(
        default=None,
        description="Internal-QAT LoRA dropout. Falls back to lora_dropout (or 0.0) when unset.",
    )
    bs2_qat_freeze_non_lora: bool = Field(
        default=True,
        description="For internal_qat mode, freeze all parameters except internal lora_A/lora_B.",
    )
    bs2_qat_snap_on_finish: bool = Field(
        default=False,
        description=(
            "When True, at post_train snaps all Bonsai2LinearFakeQuant layers onto the exact "
            "PQ2_0 ternary grid in the Hadamard domain and unfolds back to standard nn.Linear."
        ),
    )
    bs2_qat_target_names: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of Linear child-module names to patch. Defaults to the 10 "
            "Ternary-Bonsai-2 rotated projections (q/k/v/o_proj, in_proj_qkv/z, out_proj, "
            "gate/up/down_proj)."
        ),
    )
    bs2_qat_skip_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BS2_SKIP_NAMES),
        description="Child-module names to leave as standard nn.Linear.",
    )
