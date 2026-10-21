"""Plugin config schema for OneBitLLMs llama.cpp QAT."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LlamaCppQATArgs(BaseModel):
    llama_cpp_qat: bool = Field(
        default=False,
        description="Enable OneBitLLMs llama.cpp-compatible fake quant layers.",
    )
    llama_cpp_qat_quant_type: str = Field(
        default="Q4_0",
        description="llama.cpp fake quant type, for example Q4_0.",
    )
    llama_cpp_qat_activation_quant: Optional[str] = Field(
        default=None,
        description="Optional activation fake quant type. Keep null for weight-only QAT.",
    )
    llama_cpp_qat_backend: str = Field(
        default="torch",
        description="Fake quant backend: torch, triton, or auto.",
    )
    llama_cpp_qat_accumulator_dtype: Optional[str] = Field(
        default="float32",
        description="Accumulator dtype used inside fake quant linear forward.",
    )
    llama_cpp_qat_target_names: Optional[list[str]] = Field(
        default=None,
        description="Optional child-module name substrings to patch.",
    )
    llama_cpp_qat_skip_names: list[str] = Field(
        default_factory=lambda: ["lm_head"],
        description="Child-module names to leave as nn.Linear.",
    )
