# `LlamaCppQATPlugin`: `llama.cpp`-Compatible Fake-Quant QAT (`OneBitLLMs`)

`axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin` patches decoder `nn.Linear` layers with `onebitllms.layers.LlamaCppFakeQuantLinear`, emulating the exact block sizes, `FP16` scale headers, and C truncation rules (`trunc(x * id + offset)`) from `ggml/src/ggml-quants.c` (`Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q8_0`, `Q8_1`).

For full mathematical formulas, configuration reference, and benchmark details, see [`docs/qat_plugins/04_llama_cpp_qat_onebitllms.md`](../../../../docs/qat_plugins/04_llama_cpp_qat_onebitllms.md).

---

## 1. Supported Formats

| Format | Block Size | Scale Header | Rounding / Truncation Rule |
| :--- | :---: | :--- | :--- |
| `Q1_0` | `128` | `d = fp16(mean(\|x\|))` | Binary $\{-1, +1\}$ via `where(x >= 0, 1, -1)` |
| `Q2_0` | `128` | `d = fp16(1.5 * mean(\|x\|))` | Asymmetric 2-bit $\{-2, -1, 0, 1\}$ via `clamp(trunc(x / d + 2.5), 0, 3) - 2` |
| `Q4_0` | `32` | `d = fp16(signed_absmax / -8.0)` | Symmetric 4-bit $[-8, 7]$ via `clamp(trunc(x / d + 8.5), 0, 15) - 8` |
| `Q4_1` | `32` | `m = fp16(min(x))`, `d = fp16((max - min) / 15.0)` | Asymmetric 4-bit $[0, 15]$ via `clamp(trunc((x - m) / d + 0.5), 0, 15)` |
| `Q8_0` | `32` | `d = fp16(max(\|x\|) / 127.0)` | Symmetric 8-bit $[-127, 127]$ via `clamp(round(x / d), -127, 127)` |
| `Q8_1` | `32` | `d = fp16(max(\|x\|) / 127.0)`, `s = fp16(sum(q) * d)` | Same quantization grid as `Q8_0` |

---

## 2. GPU Computational Optimizations

1. **`@torch.no_grad()` + In-Place Quantization + Zero-Allocation `_STEGate`**:
   - All inner block quantizers (`_fake_quant_q1_0_no_grad`, `_fake_quant_q2_0_no_grad`, `_fake_quant_q4_0_no_grad`, `_fake_quant_q4_1_no_grad`, `_fake_quant_q8_0_like`) run inside `@torch.no_grad()` using in-place `.add_().trunc_().clamp_().sub_().mul_()` operations and attach the STE identity gradient via `_STEGate.apply(x, x_q)`.
2. **CUDA `BF16` / `FP16` Tensor Core Preservation**:
   - On CUDA GPUs, `F.linear` in `bfloat16` or `float16` natively accumulates dot products in `FP32` inside Tensor Core MMA instructions. `LlamaCppFakeQuantLinear.forward` avoids upcasting `x` and `w_q` to `float32` on CUDA when `accumulator_dtype == torch.float32`, keeping `F.linear` on `BF16`/`FP16` Tensor Cores (**6.73x speedup** from `30.824 ms` to `4.583 ms` and **-377.9 MB peak VRAM** per layer on NVIDIA H100 PCIe).

---

## 3. Configuration (`LlamaCppQATArgs`)

```yaml
plugins:
  - axolotl.integrations.llama_cpp_qat.LlamaCppQATPlugin

llama_cpp_qat: true
llama_cpp_qat_quant_type: Q4_0           # Q1_0 | Q2_0 | Q4_0 | Q4_1 | Q8_0 | Q8_1
llama_cpp_qat_activation_quant: null     # null or Q8_0
llama_cpp_qat_backend: torch             # torch | triton | auto
llama_cpp_qat_accumulator_dtype: float32 # float32 | bfloat16 | float16 | none
llama_cpp_qat_target_names: null         # optional list of target substrings
llama_cpp_qat_skip_names:
  - lm_head
```
