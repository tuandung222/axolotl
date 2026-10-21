# Bonsai Internal-QAT Non-Reasoning Run

## Objective

Train `prism-ml/Ternary-Bonsai-4B-unpacked` for one epoch with BS ternary internal-QAT LoRA at 32K context on four H200 GPUs.

## QAT and LoRA

- Mode: ternary QAT, group size 128
- LoRA mode: `internal_qat`
- Rank: 128
- Alpha: 256
- Non-LoRA weights frozen
- Targets: q/k/v/o and gate/up/down projections
- PEFT `adapter` remains unset

## Corpus

- `custom-org/planning-sft-dataset-v07022026`: `1.v2`, `2.v01`, and `3.bucket_b`, each sampled twice
- `tuandunghcmut/vi-if-dataset`: `pass`
- `tuandunghcmut/nemotron-rl-completed-if_free_form_formatting`: `pass`
- `tuandunghcmut/nemotron-rl-completed-if_calendar_v2`: `pass`
- `tuandunghcmut/vi-translation-task`: `pass`
- `tuandunghcmut/nemotron-rl-completed-structured_outputs`: `pass`
- `tuandunghcmut/Nemotron-SFT-Agentic-v2-search-toolcalling-parquet`: `tool_calling` only
- `tuandunghcmut/Nemotron-SFT-Agentic-v2-parquet`: `interactive_agent`, with reasoning fields omitted and visible-reasoning rows filtered
- `tuandunghcmut/nvidia_instruction_following_if_split_v3_non_thinking`: `train`
- `tuandunghcmut/merge-2-split-nemotron-instruction-following`: normalized locally from JSON-string columns and filtered for reasoning markers

## Excluded Corpus

- `tuandunghcmut/Nemotron-Agentic-v1-interactive-agent`: reasoning leakage
- `tuandunghcmut/vi-agentic-search-toolcalling`: reasoning leakage
- `tuandunghcmut/doc-slide-agent-sft-trajectories`: reasoning leakage, irregular prefix roles, and `tool_schemas` mismatch
- `tuandunghcmut/vi-agentic-search-toolcalling-pairs`: explicit Thought/Suy nghĩ text
- Nemotron v2 `search`: its system prompt explicitly requires visible Thought/Search/Observation output

## Validation Policy

- Reject non-empty `reasoning_content`.
- Reject `<think>` and `</think>` in content.
- Reject explicit `Thought:` or `Suy nghĩ:` sections.
- Require conversations to start with system/user and end with assistant.
- Require tool-call and tool-response counts to match.
- Preserve top-level tools so the tokenizer chat template injects tool metadata.

## Runtime

- Config: `examples/bs_bonsai_internal_qat_lora_non_reasoning.yaml`
- Preparation script: `scripts/bonsai_internal_qat/prepare_dataset.py`
- Runner: `scripts/bonsai_internal_qat/run_tmux.sh`
- Tmux session: `bonsai_internal_qat_4xh200`
- Unified dataset: `artifacts/bonsai_internal_qat_non_reasoning/data/train.jsonl`
- Dataset manifest: `artifacts/bonsai_internal_qat_non_reasoning/data/train.manifest.json`
- Prepared cache: `artifacts/bonsai_internal_qat_non_reasoning/prepared`
- Output: `artifacts/bonsai_internal_qat_non_reasoning/checkpoints`
- Preparation workers: 256 (192 parquet row-group tasks)
- Tokenization workers: 256
- Epochs: 1
- Preparation log: `artifacts/bonsai_internal_qat_non_reasoning/logs/prepare.log`
- Tokenization log: `artifacts/bonsai_internal_qat_non_reasoning/logs/tokenize.log`
- Train log: `artifacts/bonsai_internal_qat_non_reasoning/logs/train.log`
