# Low-Bit Quantization-Aware Training (QAT) & Distillation (QAD) Plugins

This directory documents the four low-bit Quantization-Aware Training (QAT) and Quantization-Aware Distillation (QAD) plugins added in this Axolotl fork (`tuandung222/axolotl`), along with the forward and backward GPU computation optimizations implemented across all four integrations.

---

## 1. Plugin Comparison Matrix

| Plugin | Module Path | Target Quantization Formats | Primary Target Models | Supported Training Modes | Export Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[`bs2_qat`](./01_bs2_qat_bonsai2_hadamard.md)** | `axolotl.integrations.bs2_qat.BS2QATPlugin` | Block-wise Sylvester Walsh-Hadamard rotation + `PQ2_0` (`group_size=128`, least-squares ternary $\{-1, 0, +1\}$, FP16 scale) | `Ternary-Bonsai-2-{2B, 4B, 9B, 27B}` (`Qwen3.5`, `tuandunghcmut/QW-3.5-BS-2-*`) | `full_weight`, `internal_qat`, `external_peft` | Unfolded HF `nn.Linear`, `PQ2_0` GGUF, MLX |
| **[`bs_qat`](./02_bs_qat_bitlinear.md)** | `axolotl.integrations.bs_qat.BSQATPlugin` | `ternary` (`Q2_0`, $\{-1, 0, +1\}$), `binary` (`Q1_0`, $\{-1, +1\}$), `int4` (`W4A16`, $[-8, 7]$) | `prism-ml/Ternary-Bonsai-4B-unpacked` (`Qwen3`), general decoder LLMs | `external_peft`, `internal_qat`, `full_weight` | Baked HF `nn.Linear` $\rightarrow$ `Q2_0`/`Q1_0` GGUF |
| **[`qad`](./03_qad_distillation.md)** | `axolotl.integrations.qad.QADPlugin` | Fixed-scale or LSQ symmetric `INT4` (`W4A16`, `group_size=128`, $q \in [-8, 7]$) initialized from PTQ/GPTQ | `Qwen3.5-{4B, 9B}` PTQ/GPTQ (`compressed-tensors`) | Full-weight fixed-scale QAD, LSQ (`qad_learn_scale`), Internal LoRA (`qad_internal_lora`) | `compressed-tensors` `pack-quantized` W4A16 (vLLM Machete/Marlin) |
| **[`llama_cpp_qat`](./04_llama_cpp_qat_onebitllms.md)** | `axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin` | Exact `ggml-quants.c` block layouts: `Q1_0` (g128), `Q2_0` (g128), `Q4_0` (g32), `Q4_1` (g32), `Q8_0` (g32), `Q8_1` (g32) | `google/gemma-4-E2B-it`, general HF CausalLMs | Full fine-tuning or PEFT LoRA (`post_lora_load`) | HF checkpoint $\rightarrow$ `convert_hf_to_gguf.py` $\rightarrow$ `llama-quantize` |

---

## 2. Forward & Backward GPU Computation Audit Summary

All four plugins were audited for redundant memory allocations, unnecessary dtype conversions (`BF16 -> FP32 -> INT8 -> FP32 -> BF16`), autograd graph retention inside non-differentiable rounding steps, floating-point cancellation in Straight-Through Estimators (STE), and dense weight-gradient GEMMs during internal-LoRA training.

### 2.1. Core Systems Optimizations Applied Across Plugins

1. **Rank-$r$ Factored Backward for Internal-LoRA STE (`bs2_qat`, `bs_qat`, `qad`):**
   - In `internal_qat` mode, a linear layer computes $Y = X \hat{W}^\top$ where $\hat{W} = Q(W_{\text{base}} + \frac{\alpha}{r} B A)$ and $W_{\text{base}}$ is frozen (`requires_grad=False`).
   - Standard PyTorch autograd through `F.linear(x, w_hat)` materializes the dense weight gradient $\nabla_{\hat{W}} \mathcal{L} = (\nabla_Y \mathcal{L})^\top X \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (`24 MB` to `136 MB` per layer) before multiplying by $A^\top$ and $B^\top$.
   - Custom `torch.autograd.Function` classes (`_BS2InternalLoRALinearSTE`, `_BitLinearInternalLoRASTE`, `_FQLoRALinearSTE`) factor the LoRA parameter gradients directly through the low-rank bottleneck $r \ll \min(d_{\text{out}}, d_{\text{in}})$ on BF16 Tensor Cores:
     $$G_{\text{proj}} = (\nabla_Y \mathcal{L}) B \in \mathbb{R}^{T \times r}, \qquad \nabla_A \mathcal{L} = \frac{\alpha}{r} G_{\text{proj}}^\top X_{\text{drop}}, \qquad \nabla_B \mathcal{L} = \frac{\alpha}{r} (\nabla_Y \mathcal{L})^\top (X_{\text{drop}} A^\top)$$
   - For $T=4096, d_{\text{out}}=6144, d_{\text{in}}=2048, r=16$, this reduces the LoRA weight-gradient compute from **104.9 GFLOPs** to **2.15 GFLOPs** (a **48.8x** reduction in weight-gradient FLOPs) and eliminates the `[d_out, d_in]` gradient buffer allocation.
   - Forward effective-weight construction uses a single fused cuBLAS `torch.addmm(W_fp32, B, A, alpha=scaling)` call followed by in-place fake quantization.

2. **Zero-Allocation `_STEGate` & Elimination of `bfloat16` Sub/Add Cancellation (`bs2_qat`, `qad`):**
   - Implementing STE via `w_q = w + (w_hard - w).detach()` in `bfloat16` launches two elementwise CUDA kernels, allocates two `[d_out, d_in]` temporary tensors, and suffers from IEEE-754 7-bit mantissa cancellation (`w + round_bf16(w_hard - w) != w_hard` when `|w|` and `|w_hard|` have different exponents). On `Bonsai2LinearFakeQuant`, this caused **93.78%** of BF16 forward outputs to differ by 1 ULP from the true quantized grid used at inference.
   - Replacing this expression with `_STEGate.apply(w, w_hard)` returns `w_hard` directly in `forward` and passes `grad_output` straight to `w` in `backward` (`return grad_output, None`), using **0 elementwise kernels**, **0 bytes of extra memory**, and **100.0000% bit-exact parity** with inference weights.

3. **Elimination of `float32 -> int8 -> float32` Roundtrips & Out-of-Place Quantizer Allocations (`bs2_qat`, `bs_qat`, `qad`, `llama_cpp_qat`):**
   - Training-time fake quantizers (`_ternary_dequant_pq2_0_no_grad`, `_fake_quantize_dequantize_no_grad`, `fake_quant`, `_fake_quant_q4_0_no_grad`) now execute entirely under `@torch.no_grad()` using in-place rounding/clamping/scaling (`.round_().clamp_().mul_()`), avoiding unused `int8` tensor materializations and saving 4–7 full-matrix allocations per forward pass.

4. **Top-$k$ `renorm` KL Divergence Without Full-Vocabulary `log_softmax` (`qad`):**
   - In `src/axolotl/integrations/qad/kl.py`, when `topk_mode == "renorm"`, softmax shift/scale invariance implies that `log_softmax(topk(logits))` is mathematically identical to `log_softmax(log_softmax(full_logits)[topk_idx])`.
   - Skipping the two `[chunk_tokens, vocab_size]` full-vocabulary `log_softmax` calls saves **885.0 MB of peak VRAM** at `V = 151,936` (`chunk_tokens = 1024`) and over **1.4 GB** at `V = 248,320`.

5. **CUDA BF16/FP16 Tensor Core Preservation (`llama_cpp_qat` / `onebitllms`):**
   - Previously, `LlamaCppFakeQuantLinear.forward` cast `x` and `w_q` from `bfloat16` to `float32` whenever `accumulator_dtype="float32"` (the default), forcing cuBLAS off BF16 Tensor Cores onto slow FP32 SGEMM/TF32 paths and allocating 236 MB of FP32 activation/gradient buffers.
   - Because NVIDIA Tensor Cores natively accumulate `BF16` and `FP16` matrix multiplications in `FP32` in hardware, keeping `F.linear(x, w_q, bias)` in `bfloat16`/`float16` on CUDA yields a **6.73x end-to-end speedup** (`30.824 ms -> 4.583 ms`) and saves **377.9 MB of peak VRAM** per layer with `cosine_similarity = 1.0000000`.

---

## 3. End-to-End Benchmark Results on NVIDIA H100 PCIe

Measured on a single decoder MLP projection (`T = 4096` tokens, `d_out = 6144`, `d_in = 2048`, `lora_r = 16`, `bfloat16`, `forward + backward` pass) on an **NVIDIA H100 PCIe (80 GB)**:

| Plugin / Component | Before Latency | After Latency | Speedup | Before Peak $\Delta$VRAM | After Peak $\Delta$VRAM | VRAM Saved | Numerical Parity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bs2_qat`** (`Bonsai2LinearFakeQuant`, `B=512`, `internal_qat`) | `8.723 ms` | **`7.113 ms`** | **1.23x** | `285.4 MB` | **`260.9 MB`** | **-24.5 MB** | **100.00% exact inference grid match** |
| **`bs_qat`** (`BitLinear`, `ternary`, `internal_qat`) | `6.304 ms` | **`4.741 ms`** | **1.33x** | `285.1 MB` | **`236.6 MB`** | **-48.5 MB** | `exact_equal=True` (`y`, `grad_x`), `cos(grad_A/B)=0.999994` |
| **`qad`** (`FQLoRALinear`, `W4A16 g128`) | `4.777 ms` | **`3.284 ms`** | **1.45x** | `272.6 MB` | **`104.1 MB`** | **-168.5 MB** | `exact_equal=True` (`y`, `grad_x`), `cos(grad_A/B)=0.999995` |
| **`qad`** (`chunked_kl_loss`, `renorm`, $V=151936, k=64$) | `67.851 ms` | **`67.061 ms`** | **1.01x** | `1779.7 MB` | **`894.8 MB`** | **-885.0 MB** | `cos(loss)=1.0000000`, `cos(grad)=1.0000000` |
| **`llama_cpp_qat`** (`LlamaCppFakeQuantLinear`, `Q4_0`, `BF16`) | `30.824 ms` | **`4.583 ms`** | **6.73x** | `514.9 MB` | **`137.0 MB`** | **-377.9 MB** | `cos(y)=1.0000000`, `cos(grad_W)=1.0000000` |

---

## 4. Detailed Plugin Documentation

1. **[`01_bs2_qat_bonsai2_hadamard.md`](./01_bs2_qat_bonsai2_hadamard.md)** — Bonsai-2 Sylvester Walsh-Hadamard + `PQ2_0` Group-128 Least-Squares Ternary QAT (`BS2QATPlugin`)
2. **[`02_bs_qat_bitlinear.md`](./02_bs_qat_bitlinear.md)** — Bonsai-1 `BitLinear` `Q2_0`/`Q1_0`/`INT4` QAT (`BSQATPlugin`)
3. **[`03_qad_distillation.md`](./03_qad_distillation.md)** — W4A16 Fixed-Scale & LSQ Quantization-Aware Distillation (`QADPlugin`)
4. **[`04_llama_cpp_qat_onebitllms.md`](./04_llama_cpp_qat_onebitllms.md)** — `llama.cpp` Fake-Quant QAT via `OneBitLLMs` (`LlamaCppQATPlugin`)
