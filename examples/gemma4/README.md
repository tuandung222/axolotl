# Finetune Google's Gemma 4 with Axolotl

[Gemma 4](https://huggingface.co/collections/google/gemma-4) is a family of multimodal models from Google. This guide covers how to train them with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
# 26B MoE QLoRA (1x80GB)
axolotl train examples/gemma4/26b-a4b-moe-qlora.yaml

# 31B Dense QLoRA (1x80GB @ ~25.2 GiB)
axolotl train examples/gemma4/31b-qlora.yaml

# E2B vision LoRA (1x80GB @ ~10.4 GiB)
axolotl train examples/gemma4/e2b-vision-lora.yaml

# E2B text-only Q4_0 llama.cpp-compatible QAT
axolotl train examples/gemma4/e2b-q4_0-llama-cpp-qat.yaml
```

## E2B text-only Q4_0 llama.cpp QAT

This fork includes an experimental QAT path for `google/gemma-4-E2B-it` using
OneBitLLMs llama.cpp-compatible fake quantization.

The example is intentionally text-only:

- `force_text_only: true` prevents Axolotl from selecting the multimodal
  collator for Gemma 4 E2B.
- `type_of_model: Gemma4ForConditionalGeneration` keeps the original checkpoint
  architecture so the `model.language_model.*` weights load correctly.
- LoRA targets are restricted to `model.language_model.layers.*`.
- `llama_cpp_qat_quant_type: Q4_0` applies weight fake quantization that follows
  the llama.cpp Q4_0 block layout.

Prepare the non-reasoning tool-calling dataset:

```bash
python examples/gemma4/scripts/prepare_nemotron_agentic_non_reasoning.py \
  --output-dir examples/gemma4/data/nemotron_agentic_non_reasoning
```

Run the QAT smoke or training job:

```bash
axolotl train examples/gemma4/e2b-q4_0-llama-cpp-qat.yaml
```

The dataset preparation script uses both parquet subsets from
`tuandunghcmut/Nemotron-SFT-Agentic-v2-search-toolcalling-parquet`, removes
reasoning fields, strips `<think>...</think>` spans, and normalizes missing
tool-call fields that break the Gemma 4 template renderer.

The QAT integration requires OneBitLLMs to be installed or available on
`PYTHONPATH`:

```bash
pip install onebitllms
```

### MoE Expert Quantization & Expert LoRA (26B-A4B only)

The 26B-A4B config uses ScatterMoE kernels via the transformers `ExpertsInterface` and quantizes expert weights on load. To learn about expert quantization, expert LoRA targeting, and related limitations, see the [MoE Expert Quantization](https://docs.axolotl.ai/docs/expert_quantization.html) docs.

## Limitations

- **Flash Attention**: FA2 (max head_dim=256) and FA4 (max head_dim=128) cannot serve Gemma 4's `global_head_dim=512` on their own. Use `flex_attention`, or `gemma4_hybrid_attn_impl: true` to run the sliding-window layers under FA2 and the global (head_dim=512) layers under `sdpa` (requires `attn_implementation: flash_attention_2` and a flash-attn build for your GPU arch).
- **LoRA kernels**: Not supported for models with KV-sharing layers.
- **lora_target_linear**: Incompatible for multimodal models; use `lora_target_modules` with a regex to restrict LoRA to the text backbone.

### TIPS

- `gemma4_hybrid_attn_impl: true` trains ~2× faster than `flex_attention` on 31B (~25.2 GiB reserved, packing on) and avoids the flex `head_dim=512` kernel, which can exhaust shared memory on Blackwell.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- You can run full finetuning by removing `adapter: qlora`, `load_in_4bit: true`, and `quantize_moe_experts: true` from the config. This is heavy and has not been tested.

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Gemma 4 Blog](https://huggingface.co/blog/gemma4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
