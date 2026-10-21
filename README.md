<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>
  <p align="center">
      <strong>A Free and Open Source LLM Fine-tuning Framework</strong><br>
  </p>

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://codecov.io/gh/axolotl-ai-cloud/axolotl"><img src="https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <a href="https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/docker-e2e.yml/badge.svg" alt="docker-e2e-tests">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>


<details>
<summary><h2 style="display:inline">Fork Additions: BS_QAT Ternary/Binary QAT (BitLinear)</h2></summary>

This fork adds **BS_QAT** (`axolotl.integrations.bs_qat.BSQATPlugin`), a
BitNet-style fake-quant QAT integration targeting ternary (`{-1,0,+1}`,
GGUF `Q2_0`) and binary (`{-1,+1}`, GGUF `Q1_0`) weights, designed for the
`prism-ml/Ternary-Bonsai-4B-unpacked` base model and the
Bonsai 1-bit LLM framework. Full
docs, config reference, and export pipeline are in
[`src/axolotl/integrations/bs_qat/README.md`](src/axolotl/integrations/bs_qat/README.md).

Highlights:

- **Group-wise Fake Quantization**: Default `group_size: 128` straight-through
  estimator (STE) fake quantization of selected `nn.Linear` layers (`q/k/v/o_proj`,
  `gate/up/down_proj` by default); `lm_head` and token embeddings remain in full precision.
- **Three LoRA / QAT Composition Modes** via `bs_qat_lora_mode`:
  - `external_peft` (default): PEFT LoRA stays outside the fake-quant layer —
    $y = \text{QDQ}(W)\cdot x + \text{LoRA}(x)$.
  - `internal_qat`: LoRA lives directly inside `BitLinear` and is fake-quantized
    together with the base weight every forward pass — $y = \text{QDQ}(W + \text{LoRA})\cdot x$,
    matching the single-tensor objective of a baked Q2_0 GGUF export and avoiding train/deploy mismatch.
  - `full_weight`: Full-parameter QAT with no LoRA adapter (`adapter: null`). Every target
    `BitLinear.weight` is directly trainable and fake-quantized as $y = \text{QDQ}(W)\cdot x$.
- **Exact Path Segment Matching**: Module replacement uses whole dot-separated segment matching
  (`_matches_target`), preventing accidental substring matches on fused projections (e.g. `up_proj` will not match `gate_up_proj`).
- **Distributed Rank-Zero Safe Saving**: Checkpoint hooks verify rank zero via process environment
  variables (`_is_rank_zero`) even after distributed process groups are torn down.
- **Export & Baking Pipeline**:
  - `scripts/export_bs_qat_lora_baked_hf.py` bakes trained weights/adapters into a plain Hugging Face model.
  - `bitnet_qat/export/requantize_gguf_v2.py` repacks QAT-trained tensors into Q2_0/Q1_0 GGUF format without corrupting embeddings or `output.weight`.
  - `scripts/eval_qat_lora_ppl.py` performs perplexity evaluation and regression benchmarks across base, baked, and quantized checkpoints.
- **Cluster Runners & Large-Context Configs**:
  - `scripts/bonsai_internal_qat/` contains multi-worker dataset preparation (`prepare_dataset.py`) and multi-GPU tmux runners (`run_tmux.sh`, `run_tmux_full_weight.sh`).
  - `examples/bs_bonsai_internal_qat_lora_non_reasoning.yaml` & `examples/bs_bonsai_full_qat_non_reasoning.yaml` provide 32K context configurations validated on 4x H200 nodes.

</details>

<details>
<summary><h2 style="display:inline">Fork Additions: llama.cpp-Compatible 4-Bit & 8-Bit QAT</h2></summary>

This fork provides a first-class training path for **llama.cpp-compatible fake-quant QAT**
supporting `Q4_0`, `Q4_1`, `Q8_0`, and `Q8_1` block layouts and rounding rules matching
`ggml/src/ggml-quants.c`. The primary validated target is `google/gemma-4-E2B-it` with
Q4_0 weight fake quantization and text-only SFT.

Highlights:

- **`LlamaCppQATPlugin` Integration**: Located at `axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin`.
  Wraps selected decoder `nn.Linear` modules with `onebitllms.layers.LlamaCppFakeQuantLinear`.
- **Bit-Exact Llama.cpp Q4_0 Emulation**:
  - Uses `block_size = 32` matching `block_q4_0`.
  - Simulates fp16 scale quantization (`d = fp16(signed_absmax / -8.0)`).
  - Exact C++ truncation and offset: `q = clamp(trunc(x * id + 8.5), 0, 15)` ($q = \text{clamp}(\text{trunc}(x / d + 8.5), 0, 15)$).
  - Supports Straight-Through Estimator (STE) for backward pass gradients.
- **llama.cpp QAT Config Fields**:
  - `llama_cpp_qat: true`
  - `llama_cpp_qat_quant_type: "Q4_0"` (or `Q4_1`, `Q8_0`, `Q8_1`, `Q1_0`, `Q2_0`)
  - `llama_cpp_qat_activation_quant: null` (or `Q8_0` for dot-product input emulation)
  - `llama_cpp_qat_backend: "triton"` (or `"pytorch"`)
  - `llama_cpp_qat_targets: [...]`
  - `llama_cpp_qat_skip_targets: [...]`
- **Validated Gemma 4 E2B Examples**:
  - `examples/gemma4/e2b-q4_0-llama-cpp-qat.yaml`: LoRA text-only Q4_0 QAT.
  - `examples/gemma4/e2b-q4_0-llama-cpp-full-qat-aligned.yaml`: Full finetuning statically aligned with
    llama.cpp PTQ tensor coverage (patches 275 decoder linears including `per_layer_input_gate` and `per_layer_projection`,
    while preserving `per_layer_model_projection` at F16 as required by llama.cpp).
- **Non-Reasoning Agentic Dataset Pipeline**:
  - `examples/gemma4/scripts/prepare_nemotron_agentic_non_reasoning.py`: Extracts and cleans Nemotron v2 parquet data.
  - `examples/gemma4/gemma4_non_reasoning_tool_safe.jinja`: Gemma 4 tool-safe chat template without reasoning artifacts.

Requirements:

```bash
pip install git+https://github.com/tuandung222/onebitllms.git
# Or include local sibling repo: export PYTHONPATH=/path/to/tuandung222_onebitllms/src:$PYTHONPATH
```

Workflow:

```bash
# 1. Prepare data
python examples/gemma4/scripts/prepare_nemotron_agentic_non_reasoning.py \
  --output-dir examples/gemma4/data/nemotron_agentic_non_reasoning

# 2. Train with Q4_0 QAT
axolotl train examples/gemma4/e2b-q4_0-llama-cpp-full-qat-aligned.yaml

# 3. Export HF checkpoint -> convert to GGUF -> llama-quantize ... Q4_0
```

</details>

<details>
<summary><h2 style="display:inline">Fork Additions: Agentic Dataset Stabilization & Model Support</h2></summary>

- **Heterogeneous Tool Schema Stabilization**:
  Added `_stabilize_tools_json_column()` in `src/axolotl/utils/data/shared.py`. Datasets with
  complex agentic tool calling often have varying nested dictionary structures in `parameters.properties`.
  This utility automatically stringifies heterogeneous tool definitions prior to Arrow type inference,
  preventing `pyarrow` schema mismatch crashes during dataset loading.
- **Extended Architecture Recognition**:
  Added explicit support for `lfm2_moe` and `lfm2_vl` in `ModelLoader._check_model_requirements()`.
- **LFM2.5 8B A1B Agentic Fine-Tuning**:
  Includes training configs (`examples/LiquidAI/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.yaml`),
  data preparation scripts (`scripts/prepare_lfm2_5_non_reasoning_mix.py`), and background execution
  runners for 4x H200 clusters.
- **Synchronized with Upstream Axolotl (v0.18.0+)**:
  Fully merged with upstream `axolotl-ai-cloud/axolotl` (commit `0e1443758`), featuring native Flash Attention 4
  (SM90 Hopper and SM100 Blackwell with `quack-kernels >= 0.6.0`), FSDP2 compile enhancements, and configurable
  FP8 scaling recipes.

</details>

## 🎉 Latest Updates

- 2026/10:
  - Reflection's [Beam](https://docs.axolotl.ai/docs/models/beam.html) is now supported in Axolotl, with QLoRA and expert-parallel LoRA configs including NVFP4.
  - [Distributed training improvements](https://github.com/axolotl-ai-cloud/axolotl/pull/4091) keep EP groups node-local when the topology permits, add DeepSpeed Ulysses support through `context_parallel_size`, and fix expert-LoRA resume and FSDP2 activation checkpointing with LoRA.
- 2026/09:
  - New model support has been added in Axolotl for [MiniCPM5](https://docs.axolotl.ai/docs/models/minicpm5.html), [Nex-N2.5-mini](https://docs.axolotl.ai/docs/models/nex-n2.5.html), [Qwen3.6](https://docs.axolotl.ai/docs/models/qwen3.5.html), [Qwen3.8-Flash-Next](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html), and [K2 Horizon](https://docs.axolotl.ai/docs/models/k2-horizon.html).
  - [CPU-staged FSDP2 QLoRA loading](https://docs.axolotl.ai/docs/expert_quantization.html#cpu-staged-fsdp2-qlora-loading) lets you load models larger than a single GPU's memory without materializing a full quantized replica on each GPU.
  - [GGUF export](https://docs.axolotl.ai/docs/export.html) is now available via `axolotl export`, with optional quantization for deployment with llama.cpp and Ollama.
  - [NVFP4 LoRA](https://docs.axolotl.ai/docs/nvfp4_lora.html) now supports native TorchAO dense weights and merge-aware training, optimizing against the quantized merged weights for deployment.
  - New training options: [MixLoRA](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/mixlora) adds routed LoRA experts to dense models, and [Dynamic Fine-Tuning (DFT)](https://github.com/axolotl-ai-cloud/axolotl/pull/4008) weights the SFT loss by token probability.
  - [Ringmaster context parallelism](https://docs.axolotl.ai/docs/sequence_parallelism.html) adds Ulysses and combined Ulysses/Ring (USP) support, sample packing, and CP fixes for GDN, KDA, and Mamba/Mamba2 models.
  - [Torch all-to-all Expert Parallelism](https://docs.axolotl.ai/docs/custom_integrations.html#expert-parallelism-integration) enables MoE training over NVLink or PCIe without requiring DeepEP. Expanded [N-D parallelism](https://docs.axolotl.ai/docs/nd_parallelism.html) supports EP with FSDP2, HSDP, and context parallelism, plus FSDP2 + tensor parallelism.
  - [Selective activation checkpointing](https://docs.axolotl.ai/docs/gradient_checkpointing.html#selective-activation-checkpointing-sac) adds rules to save specific projections or large matrix multiplications, with optional CPU offload to tune the memory/recomputation tradeoff.
  - Improved [plugin integration](https://docs.axolotl.ai/docs/custom_integrations.html) includes new cloud provider interfaces for launching training jobs, with support for [Nebius Serverless Jobs](https://docs.axolotl.ai/docs/nebius-serverless.html).
- 2026/08:
  - New model support has been added in Axolotl for [Ling 3.0](https://docs.axolotl.ai/docs/models/ling3.html), [Muse Glimmer](https://docs.axolotl.ai/docs/models/muse-glimmer.html), [North Micro Vision Instruct](https://docs.axolotl.ai/docs/models/cohere-north-micro-vision-instruct.html) and [Shieldstral](https://docs.axolotl.ai/docs/models/shieldstral.html).
- 2026/07:
  - [NVFP4 (4-bit) MoE LoRA training](https://docs.axolotl.ai/docs/nvfp4_lora.html) is now supported via ScatterMoE (W4A16) and SonicMoE (W4A4), including adapter merge back into a plain NVFP4 checkpoint.
- 2026/06:
  - [Expert Parallelism (EP)](https://docs.axolotl.ai/docs/nd_parallelism.html) for distributed MoE training via DeepEP, remote training through [Tinker-compatible APIs](https://github.com/axolotl-ai-cloud/axolotl/pull/3614), [Context Parallelism for hybrid SSM models](https://github.com/axolotl-ai-cloud/axolotl/pull/3572) (Nemotron-H, Falcon-H1, Bamba), [BitNet 1.58-bit](https://github.com/axolotl-ai-cloud/axolotl/pull/3634) fine-tuning, and a [multimodal assistant-only loss-masking fix](https://github.com/axolotl-ai-cloud/axolotl/pull/3625).

<details>

<summary>Expand older updates</summary>

- 2026/04:
  - New model support has been added in Axolotl for [Mistral Medium 3.5](https://docs.axolotl.ai/docs/models/mistral-medium-3_5.html) and [Gemma 4](https://docs.axolotl.ai/docs/models/gemma4.html).
  - New RL and kernels: [Async GRPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3486) (up to 58% faster steps), [Flash Attention 4](https://docs.axolotl.ai/docs/attention.html#flash-attention), [NeMo Gym](https://github.com/axolotl-ai-cloud/axolotl/pull/3516), and [EBFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3527).
  - Axolotl is now [uv-first](https://github.com/axolotl-ai-cloud/axolotl/pull/3545) and has [SonicMoE fused LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3519) support.
- 2026/03:
  - New model support has been added in Axolotl for [Mistral Small 4](https://docs.axolotl.ai/docs/models/mistral4.html), [Qwen3.5, Qwen3.5 MoE](https://docs.axolotl.ai/docs/models/qwen3.5.html), [GLM-4.7-Flash](https://docs.axolotl.ai/docs/models/glm47-flash.html), [GLM-4.6V](https://docs.axolotl.ai/docs/models/glm46v.html), and [GLM-4.5-Air](https://docs.axolotl.ai/docs/models/glm45.html).
  - [MoE expert quantization](https://docs.axolotl.ai/docs/expert_quantization.html) support (via `quantize_moe_experts: true`) greatly reduces VRAM when training MoE models (FSDP2 compat).
- 2026/02:
  - [ScatterMoE LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3410) support. LoRA fine-tuning directly on MoE expert weights using custom Triton kernels.
  - Axolotl now has support for [SageAttention](https://github.com/axolotl-ai-cloud/axolotl/pull/2823) and [GDPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3353) (Generalized DPO).
- 2026/01:
  - New integration for [EAFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3366) (Entropy-Aware Focal Training), weights loss by entropy of the top-k logit distribution, and [Scalable Softmax](https://github.com/axolotl-ai-cloud/axolotl/pull/3338), improves long context in attention.
- 2025/12:
  - Axolotl now includes support for [Kimi-Linear](https://docs.axolotl.ai/docs/models/kimi-linear.html), [Plano-Orchestrator](https://docs.axolotl.ai/docs/models/plano.html), [MiMo](https://docs.axolotl.ai/docs/models/mimo.html), [InternVL 3.5](https://docs.axolotl.ai/docs/models/internvl3_5.html), [Olmo3](https://docs.axolotl.ai/docs/models/olmo3.html), [Trinity](https://docs.axolotl.ai/docs/models/trinity.html), and [Ministral3](https://docs.axolotl.ai/docs/models/ministral3.html).
  - [Distributed Muon Optimizer](https://github.com/axolotl-ai-cloud/axolotl/pull/3264) support has been added for FSDP2 pretraining.
- 2025/10: New model support has been added in Axolotl for: [Qwen3 Next](https://docs.axolotl.ai/docs/models/qwen3-next.html), [Qwen2.5-vl, Qwen3-vl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen2_5-vl), [Qwen3, Qwen3MoE](https://docs.axolotl.ai/docs/models/qwen3.html), [Granite 4](https://docs.axolotl.ai/docs/models/granite4.html), [HunYuan](https://docs.axolotl.ai/docs/models/hunyuan.html), [Magistral 2509](https://docs.axolotl.ai/docs/models/magistral/vision.html), [Apertus](https://docs.axolotl.ai/docs/models/apertus.html), and [Seed-OSS](https://docs.axolotl.ai/docs/models/seed-oss.html).
- 2025/09: Axolotl now has text diffusion training. Read more [here](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/diffusion).
- 2025/08: QAT has been updated to include NVFP4 support. See [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/3107).
- 2025/07:
  - ND Parallelism support has been added into Axolotl. Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. Check out the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more info.
  - Axolotl adds more models: [GPT-OSS](https://docs.axolotl.ai/docs/models/gpt-oss.html), [Gemma 3n](https://docs.axolotl.ai/docs/models/gemma3n.html), [Liquid Foundation Model 2 (LFM2)](https://docs.axolotl.ai/docs/models/LiquidAI.html), and [Arcee Foundation Models (AFM)](https://docs.axolotl.ai/docs/models/arcee.html).
  - FP8 finetuning with fp8 gather op is now possible in Axolotl via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
  - [Voxtral](https://docs.axolotl.ai/docs/models/voxtral.html), [Magistral 1.1](https://docs.axolotl.ai/docs/models/magistral.html), and [Devstral](https://docs.axolotl.ai/docs/models/devstral.html) with mistral-common tokenizer support has been integrated in Axolotl!
  - TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support has been added to support Arctic Long Sequence Training. (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for using ALST with Axolotl!
- 2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [docs](https://docs.axolotl.ai/docs/models/magistral.html) to start training your own Magistral models with Axolotl!
- 2025/05: Quantization Aware Training (QAT) support has been added to Axolotl. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
- 2025/04: Llama 4 support has been added in Axolotl. See [docs](https://docs.axolotl.ai/docs/models/llama-4.html) to start training your own Llama 4 models with Axolotl's linearized version!
- 2025/03: Axolotl has implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.
- 2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
- 2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
- 2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
- 2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## ✨ Overview

Axolotl is a free and open-source tool designed to streamline post-training and fine-tuning for the latest large language models (LLMs).

Features:

- **Multiple Model Support**: Train various models like GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and many more models available on the Hugging Face Hub.
- **Multimodal Training**: Fine-tune vision-language models (VLMs) including LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, GLM-4.6V, InternVL 3.5, Gemma 3n, PaddleOCR-VL, Muse Glimmer, and audio models like Voxtral with image, video, and audio support.
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, GPTQ, QAT (int8/int4/FP8/NVFP4/MXFP4), FP8 mixed-precision training, NVFP4/MXFP4 MoE LoRA, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO, GDPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
- **Easy Configuration**: Re-use a single YAML configuration file across the full fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference.
- **Performance Optimizations**: [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention 2/3/4](https://docs.axolotl.ai/docs/attention.html#flash-attention), [Xformers](https://docs.axolotl.ai/docs/attention.html#xformers), [Flex Attention](https://docs.axolotl.ai/docs/attention.html#flex-attention), [SageAttention](https://docs.axolotl.ai/docs/attention.html#sageattention), [Liger Kernel](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels), [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy), [ScatterMoE](https://docs.axolotl.ai/docs/custom_integrations.html#kernels-integration), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and many more!
- **Flexible Dataset Handling**: Load from local, HuggingFace, and cloud (S3, Azure, GCP, OCI) datasets.
- **Cloud Ready**: We ship [Docker images](https://hub.docker.com/u/axolotlai) and also [PyPI packages](https://pypi.org/project/axolotl/) for use on cloud platforms and local hardware.



## 🚀 Quick Start - LLM Fine-tuning in Minutes

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python >=3.12 (3.12 recommended)
- PyTorch ≥2.13.0 (2.13.0 recommended)

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

### Installation

```bash
# install uv if you don't already have it installed (restart shell after)
curl -LsSf https://astral.sh/uv/install.sh | sh

# change depending on system
export UV_TORCH_BACKEND=cu130

# create a new virtual environment
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.14.0 torchvision
uv pip install --no-build-isolation axolotl[deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker

Installing with Docker can be less error prone than installing in your own environment.
```bash
docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

#### Cloud Providers

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.


## 📚 Documentation

- [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
- [Support Matrix](https://docs.axolotl.ai/docs/support-matrix.html) - Feature support, compatibility, and known gaps
- [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
- [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
- [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
- [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
- [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## AI Agent Support

Axolotl ships with built-in documentation optimized for AI coding agents (Claude Code, Cursor, Copilot, etc.). These docs are bundled with the pip package, no repo clone needed.

```bash
# Show overview and available training methods
axolotl agent-docs

# Topic-specific references
axolotl agent-docs sft                 # supervised fine-tuning
axolotl agent-docs grpo                # GRPO online RL
axolotl agent-docs preference_tuning   # DPO, KTO, ORPO, SimPO
axolotl agent-docs reward_modelling    # outcome and process reward models
axolotl agent-docs pretraining         # continual pretraining
axolotl agent-docs --list              # list all topics

# Dump config schema for programmatic use
axolotl config-schema
axolotl config-schema --field adapter
```

If you're working with the source repo, agent docs are also available at `docs/agents/` and the project overview is in `AGENTS.md`.

## 🤝 Getting Help

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
- Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## 📈 Telemetry

Axolotl has opt-out telemetry that helps us understand how the project is being used
and prioritize improvements. We collect basic system information, model types, and
error rates, never personal data or file paths. Telemetry is enabled by default. To
disable it, set AXOLOTL_DO_NOT_TRACK=1. For more details, see our [telemetry documentation](https://docs.axolotl.ai/docs/telemetry.html).

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

If you use Axolotl in your research or projects, please cite it as follows:

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
