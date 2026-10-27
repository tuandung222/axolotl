# 4. `LLAMA_CPP_QAT`: `llama.cpp`-Compatible Fake-Quant QAT (`OneBitLLMs`)

- **Plugin Class:** `axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin`
- **Config Schema:** `axolotl.integrations.llama_cpp_qat.args.LlamaCppQATArgs`
- **Source Directory:** `src/axolotl/integrations/llama_cpp_qat/` + `onebitllms` (`https://github.com/tuandung222/onebitllms`)
- **Primary Validated Target:** `google/gemma-4-E2B-it` (`Q4_0` full fine-tuning & LoRA QAT) and general GGUF deployment targets

---

## 1. Overview & Motivation

Standard PyTorch symmetric/asymmetric fake quantizers use textbook min-max or absmax scaling with IEEE-754 round-to-even (`torch.round`), which diverges from the actual C structs and quantization routines in `llama.cpp` (`ggml/src/ggml-quants.c`). Specifically, `ggml` quantizers use:
- Signed-absmax scale definitions (e.g., `d = max_signed / -8` in `Q4_0`),
- Explicit `FP16` (`ggml_fp32_to_fp16`) scale and min rounding before computing inverse scales,
- C truncation (`(int)(x * id + 8.5f)` in `Q4_0` or `(int)(x * id + 2.5f)` in `Q2_0`) rather than round-half-to-even.

`LlamaCppQATPlugin` bridges Axolotl with `onebitllms.layers.LlamaCppFakeQuantLinear`, emulating the exact `ggml-quants.c` block sizes, `FP16` header fields, and rounding/truncation rules during forward passes while propagating gradients via the Straight-Through Estimator (STE).

---

## 2. Supported `llama.cpp` Quantization Formats & Exact Formulas

Each weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is reshaped into contiguous blocks of size $B \in \{32, 128\}$ along `in_features`:

| Format | Block Size ($B$) | Scale Representation | Exact `ggml-quants.c` Quantization & Dequantization Rule |
| :--- | :---: | :--- | :--- |
| **`Q1_0`** | `128` | `d = fp16(mean(\|x\|))` | $q_i \in \{-1, +1\}$ via $\operatorname{where}(x_i \ge 0, 1, -1)$; $\hat{x}_i = d \cdot q_i$ |
| **`Q2_0`** | `128` | `d = fp16(1.5 * mean(\|x\|))` | $q_i = \operatorname{clamp}(\operatorname{trunc}(x_i / d + 2.5), 0, 3) - 2 \in \{-2, -1, 0, 1\}$; $\hat{x}_i = d \cdot q_i$ |
| **`Q4_0`** | `32` | `d = fp16(x[argmax(\|x\|)] / -8.0)` | $u_i = \operatorname{clamp}(\operatorname{trunc}(x_i / d + 8.5), 0, 15) \in [0, 15]$; $\hat{x}_i = d \cdot (u_i - 8)$ |
| **`Q4_1`** | `32` | `m = fp16(min(x))`, `d = fp16((max(x) - min(x)) / 15.0)` | $u_i = \operatorname{clamp}(\operatorname{trunc}((x_i - m) / d + 0.5), 0, 15)$; $\hat{x}_i = d \cdot u_i + m$ |
| **`Q8_0`** | `32` | `d = fp16(max(\|x\|) / 127.0)` | $q_i = \operatorname{clamp}(\operatorname{round}(x_i / d), -127, 127)$; $\hat{x}_i = d \cdot q_i$ |
| **`Q8_1`** | `32` | `d = fp16(max(\|x\|) / 127.0)`, `s = fp16(sum(q) * d)` | Same grid values as `Q8_0` (`block_q8_1` stores block sum `s` for fast dot-product with `Q4_1`) |

Optional activation fake quantization (`llama_cpp_qat_activation_quant: "Q8_0"`) emulates `llama.cpp`'s runtime `vec_dot_q4_0_q8_0` / `vec_dot_q2_0_q8_0` integer dot products by quantizing input activations to `Q8_0` (`block_size = 32`) prior to `F.linear`.

---

## 3. GPU Forward & Backward Computational Optimizations

### 3.1. `@torch.no_grad()` + Zero-Allocation `_STEGate` + In-Place Block Quantization

Previously, the inner PyTorch quantizers (`_fake_quant_q1_0`, `_fake_quant_q2_0`, `_fake_quant_q4_0`, `_fake_quant_q4_1`, `_fake_quant_q8_0_like`) executed **with PyTorch autograd enabled** even though their outputs were immediately detached in `x + (x_q - x).detach()`. Because operations like `torch.gather` (to find the signed absmax element in `Q4_0`), division by `d`, and multiplication by `d` are differentiable in PyTorch, autograd built an entire computation graph and retained 6–8 full-matrix `[d_out, d_in]` FP32 intermediate buffers per layer during the forward pass, only to discard the graph when `.detach()` was called.

In the optimized implementation (`onebitllms/kernels/llama_cpp_quant.py`):
- Every inner quantizer (`_fake_quant_q4_0_no_grad`, etc.) is decorated with `@torch.no_grad()`.
- Quantization and dequantization steps execute **in-place** on a single FP32 block buffer:
  ```python
  q = blocks * inv_d
  q.add_(8.5).trunc_().clamp_(0.0, 15.0).sub_(8.0).mul_(d)
  ```
- The STE identity gradient is attached via `_STEGate.apply(x, x_q)` with zero elementwise kernels and zero extra tensor allocations.

### 3.2. Preserving CUDA `BF16` / `FP16` Tensor Cores in `LlamaCppFakeQuantLinear.forward`

Previously, `LlamaCppFakeQuantLinear.forward` contained:
```python
if self.accumulator_dtype is not None:
    x_q = x_q.to(self.accumulator_dtype)
    w_q = w_q.to(self.accumulator_dtype)
    output = F.linear(x_q, w_q, bias)
    return output.to(x.dtype)
```
Because `llama_cpp_qat_accumulator_dtype` defaults to `"float32"`, every patched linear layer cast `x` (`[B, T, d_in]`) and `w_q` (`[d_out, d_in]`) from `bfloat16` to `float32` before calling `F.linear`. On NVIDIA GPUs:
1. `F.linear` in `float32` cannot use `BF16`/`FP16` Tensor Cores (falling back to FP32 SGEMM or TF32, which has $4\times$ to $8\times$ lower throughput).
2. `F.linear` in `bfloat16` or `float16` on CUDA **already accumulates in `FP32` in hardware** inside cuBLAS Tensor Core MMA instructions (`MMA.m16n8k16.F32.BF16.BF16.F32`).
3. Casting `x` and `w_q` to `float32` forced autograd to save `x_q` and `w_q` in `float32` for backward (`236 MB` vs `118 MB` per layer).

The optimized forward pass keeps `x_q` and `w_q` in their native `bfloat16`/`float16` dtype when running on CUDA with `accumulator_dtype == torch.float32`, only upcasting on CPU or for integer/FP8 inputs:
```python
use_native_tc = (
    x.is_cuda
    and x.dtype in (torch.bfloat16, torch.float16)
    and self.accumulator_dtype == torch.float32
)
```

### 3.3. H100 Benchmark Impact (`T=4096, 6144x2048, Q4_0, BF16`)

| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :---: | :---: | :---: |
| **Forward + Backward Latency** | `30.824 ms` | **`4.583 ms`** | **6.73x Faster** |
| **Peak Delta VRAM per Layer** | `514.9 MB` | **`137.0 MB`** | **-377.9 MB (-73.4%)** |
| **Output & Gradient Cosine Similarity** | `1.0000000` | **`1.0000000`** | Bit-level Tensor Core parity |

---

## 4. Configuration Reference (`LlamaCppQATArgs`)

| YAML Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `llama_cpp_qat` | `bool` | `false` | Enable `LlamaCppQATPlugin`. |
| `llama_cpp_qat_quant_type` | `str` | `"Q4_0"` | Weight fake-quant format: `"Q1_0"`, `"Q2_0"`, `"Q4_0"`, `"Q4_1"`, `"Q8_0"`, or `"Q8_1"`. |
| `llama_cpp_qat_activation_quant` | `Optional[str]` | `null` | Optional activation fake-quant format (`null` for weight-only QAT, or `"Q8_0"`). |
| `llama_cpp_qat_backend` | `str` | `"torch"` | Backend selection: `"torch"`, `"triton"` (fused Triton kernel for `Q8_0`), or `"auto"`. |
| `llama_cpp_qat_accumulator_dtype` | `Optional[str]` | `"float32"` | Accumulator dtype (`"float32"`, `"bfloat16"`, `"float16"`, or `"none"`). On CUDA `BF16`/`FP16`, `"float32"` automatically uses hardware FP32 accumulation on Tensor Cores. |
| `llama_cpp_qat_target_names` | `Optional[list[str]]` | `null` | Optional list of module name substrings to patch (when `null`, patches all eligible decoder `nn.Linear` layers whose `in_features % block_size == 0`, excluding LoRA adapters). |
| `llama_cpp_qat_skip_names` | `list[str]` | `["lm_head"]` | Child-module names to leave as unquantized `nn.Linear`. |

---

## 5. Example YAML Configurations & GGUF Workflow

### 5.1. Full-Weight `Q4_0` QAT Aligned with `llama.cpp` PTQ (`examples/gemma4/e2b-q4_0-llama-cpp-full-qat-aligned.yaml`)

```yaml
base_model: google/gemma-4-E2B-it

plugins:
  - axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin

llama_cpp_qat: true
llama_cpp_qat_quant_type: Q4_0
llama_cpp_qat_activation_quant: null
llama_cpp_qat_backend: torch
llama_cpp_qat_accumulator_dtype: float32
llama_cpp_qat_skip_names:
  - lm_head
  - per_layer_model_projection

bf16: true
tf32: true
gradient_checkpointing: true
```

### 5.2. Training & Exporting to `llama.cpp` GGUF

```bash
# 1. Train model with Q4_0 fake quantization
axolotl train examples/gemma4/e2b-q4_0-llama-cpp-full-qat-aligned.yaml

# 2. Convert trained Hugging Face checkpoint to F16 GGUF
python convert_hf_to_gguf.py outputs/gemma4-e2b-q4_0-qat \
  --outtype f16 \
  --outfile outputs/gemma4-e2b-qat-f16.gguf

# 3. Quantize to Q4_0 with llama-quantize (weights are already aligned to the Q4_0 grid)
./llama-quantize outputs/gemma4-e2b-qat-f16.gguf outputs/gemma4-e2b-qat-Q4_0.gguf Q4_0
```

### 5.3. Running Unit Tests

```bash
pytest --noconftest -o addopts="" \
  tests/test_llama_cpp_fake_quant.py \
  tests/test_kernels.py
```
