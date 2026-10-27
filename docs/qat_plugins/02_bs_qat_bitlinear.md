# 2. `BS_QAT`: Bonsai-1 `BitLinear` (`Q2_0` / `Q1_0` / `INT4`) QAT

- **Plugin Class:** `axolotl.integrations.bs_qat.BSQATPlugin`
- **Config Schema:** `axolotl.integrations.bs_qat.args.BSQATArgs`
- **Source Directory:** `src/axolotl/integrations/bs_qat/`
- **Target Model Family:** `prism-ml/Ternary-Bonsai-4B-unpacked` (`Qwen3` architecture) and general Hugging Face decoder models

---

## 1. Overview & Architecture

`BSQATPlugin` replaces selected `nn.Linear` projections with `BitLinear` (`src/axolotl/integrations/bs_qat/bitnet_qat/bitlinear.py`), performing group-wise Straight-Through Estimator (STE) fake quantization across three supported weight grids:

1. **`ternary` (`Q2_0`, default `group_size = 128`):**
   - Matches the `Q2_0` block format in Prism ML's `llama.cpp` fork (`128` ternary weights $\{-1, 0, +1\}$ + `1` `FP16` scale per block).
2. **`binary` (`Q1_0`, default `group_size = 128`):**
   - Matches the `Q1_0` block format (`128` binary signs $\{-1, +1\}$ + `1` `FP16` scale per block).
3. **`int4` (`W4A16`, default `group_size = 128`):**
   - Symmetric 4-bit integer grid $q \in [-8, 7]$ with per-group `FP16` scale, bit-compatible with `compressed-tensors` `pack-quantized` exports.

Optional per-token 8-bit dynamic activation quantization (`bs_qat_act_quant: true`) is also supported via `activation_quant_ste`.

---

## 2. Mathematical Formulation

Given a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ reshaped into groups $\mathbf{w}_g \in \mathbb{R}^G$ ($G = \text{bs\_qat\_group\_size} = 128$):

### 2.1. `ternary` (`Q2_0`) Grid

1. **Group Scale (`FP16`-quantized mean abs $\times 1.5$):**
   $$s_{\text{raw}} = \frac{1.5}{G} \sum_{i=1}^G |w_{g,i}|, \qquad s_g = \operatorname{float32}\!\big(\operatorname{float16}(\max(s_{\text{raw}}, 10^{-8}))\big)$$
   When a group's maximum absolute value is zero, $s_g = 0$.
2. **C++ `std::round` (Half-Away-From-Zero) vs IEEE-754 Round-Half-To-Even:**
   - Standard `ggml` C++ quantizers use half-away-from-zero (`std::round(u) = sign(u) * floor(|u| + 0.5)`), whereas `torch.round` uses round-half-to-even.
   - `_round_half_away_from_zero(u)` is used for `ternary` (`Q2_0`) so that exact half-integers ($\pm 0.5$) round identically to the C++ GGUF packer:
     $$q_{g,i} = \operatorname{clamp}\!\left(\operatorname{sign}(u_{g,i}) \lfloor |u_{g,i}| + 0.5 \rfloor, -1, +1\right), \qquad u_{g,i} = \frac{w_{g,i}}{s_g}$$
     $$\hat{\mathbf{w}}_g = s_g \cdot \mathbf{q}_g$$

### 2.2. `binary` (`Q1_0`) Grid

$$s_g = \operatorname{float32}\!\left(\operatorname{float16}\!\left(\frac{1}{G} \sum_{i=1}^G |w_{g,i}|\right)\right), \qquad q_{g,i} = \begin{cases} +1 & \text{if } w_{g,i} \ge 0 \\ -1 & \text{if } w_{g,i} < 0 \end{cases}, \qquad \hat{\mathbf{w}}_g = s_g \cdot \mathbf{q}_g$$

### 2.3. `int4` (`W4A16` Symmetric) Grid

$$s_g = \operatorname{float32}\!\left(\operatorname{float16}\!\left(\frac{\max_i |w_{g,i}|}{7.5}\right)\right), \qquad q_{g,i} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w_{g,i}}{s_g}\right), -8, 7\right), \qquad \hat{\mathbf{w}}_g = s_g \cdot \mathbf{q}_g$$
Note that `int4` uses `torch.round` (round-half-to-even) to match `compressed-tensors` / `llm-compressor`.

---

## 3. GPU Forward & Backward Computational Optimizations

### 3.1. Fused FP32 Quantize-Dequantize Without `int8` Roundtrips (`_fake_quantize_dequantize_no_grad`)

Previously, `FakeQuantSTE.forward` called `dequantize_weights(quantize_weights(w))`, which cast the rounded FP32 values to `torch.int8` and immediately cast `q.to(torch.float32)` back on the next line, allocating an unnecessary `[d_out, d_in]` `int8` tensor on every forward pass.

`_fake_quantize_dequantize_no_grad(w, group_size, mode)` fuses the quantize and dequantize stages inside `@torch.no_grad()` and performs rounding, clamping, and scale multiplication in-place on the FP32 buffer:
```python
q_dequant = _round_half_away_from_zero(w_groups / safe_scale).clamp_(-1.0, 1.0).mul_(scale)
```
The discrete `int8` path (`quantize_weights`) is retained exclusively for checkpoint export and GGUF packing.

### 3.2. Fused `torch.addmm` + Rank-$r$ Factored Backward (`_BitLinearInternalLoRASTE`)

In `internal_qat` mode (`BitLinear(use_lora=True)` with `weight.requires_grad=False`):
- **Forward:** `_BitLinearInternalLoRASTE.forward` constructs the merged FP32 shadow weight via a single fused cuBLAS call:
  ```python
  w_eff = torch.addmm(weight.to(torch.float32), lora_B, lora_a, alpha=lora_scaling)
  ```
  quantizes `w_eff` via `_fake_quantize_dequantize_no_grad`, and runs `F.linear(x, w_q, bias)` in `x.dtype` (`bfloat16`).
- **Backward:** Instead of letting `F.linear` materialize the dense `[d_out, d_in]` weight gradient `grad_W = grad_out.T @ x` (`24 MB` to `112 MB`), `_BitLinearInternalLoRASTE.backward` factors the gradient through rank $r$ directly on BF16 Tensor Cores:
  ```python
  grad_proj = torch.matmul(grad_out_2d, b)                          # [T, r]
  grad_lora_A = torch.matmul(grad_proj.t(), x_drop_2d).mul_(scale)  # [r, d_in]
  x_a = torch.matmul(x_drop_2d, a.t())                              # [T, r]
  grad_lora_B = torch.matmul(grad_out_2d.t(), x_a).mul_(scale)      # [d_out, r]
  ```
  On an NVIDIA H100 PCIe (`T=4096, 6144x2048, r=16`), this reduces layer forward+backward latency from `6.304 ms` to **`4.741 ms` (1.33x speedup)** and saves **48.5 MB of peak VRAM** per layer.

### 3.3. Version-Tracked Frozen-Weight Cache (`_cached_weight`)

When `use_lora=False` and `not self.weight.requires_grad` (such as during `external_peft` training or `eval()`), `BitLinear` caches the fake-quantized weight tensor keyed by `(weight._version, weight.dtype, weight.device)`. Subsequent forward passes reuse `_cached_weight` with zero quantization overhead until `weight` is modified or `clear_weight_cache()` is called.

---

## 4. Three Training / LoRA Composition Modes (`bs_qat_lora_mode`)

| Property | `external_peft` (Default) | `internal_qat` | `full_weight` |
| :--- | :--- | :--- | :--- |
| **Forward Formula** | $Y = X Q(W)^\top + \frac{\alpha}{r} X A^\top B^\top$ | $Y = X Q\!\big(W + \frac{\alpha}{r} B A\big)^\top$ | $Y = X Q(W)^\top$ |
| **Trainable Parameters** | External PEFT `lora_A`, `lora_B` | Internal `BitLinear.lora_A`, `lora_B` | `BitLinear.weight` (all patched layers) |
| **Axolotl `adapter:` Key** | `adapter: lora` | Must be unset (`null`) | Must be unset (`null`) |
| **Lifecycle Hook Used** | `post_lora_load` | `post_model_build` | `post_model_load` |
| **Saved Artifacts** | Standard PEFT `adapter_model.safetensors` | Full Trainer checkpoint + `internal_qat_lora/bs_internal_qat_lora.safetensors` | Full Trainer checkpoint |
| **Single-Tensor GGUF Parity** | Approximate (merges $W + \Delta W$ post-training) | **Exact** (optimizes directly on quantized merged grid) | **Exact** |

---

## 5. Configuration Reference (`BSQATArgs`)

| YAML Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bs_qat` | `bool` | `false` | Enable `BSQATPlugin`. |
| `bs_qat_mode` | `Literal["ternary", "binary", "int4"]` | `"ternary"` | Quantization grid (`ternary` `Q2_0`, `binary` `Q1_0`, or `int4` `W4A16`). |
| `bs_qat_group_size` | `int` | `128` | Group size $G$ along `in_features`. |
| `bs_qat_act_quant` | `bool` | `false` | Enable 8-bit per-token activation fake quantization. |
| `bs_qat_lora_mode` | `Literal["external_peft", "internal_qat", "full_weight"]` | `"external_peft"` | LoRA and fake-quant composition mode. |
| `bs_qat_lora_r` | `Optional[int]` | `null` | Rank $r$ for `internal_qat` (falls back to top-level `lora_r`). |
| `bs_qat_lora_alpha` | `Optional[int]` | `null` | Alpha scaling for `internal_qat` (falls back to top-level `lora_alpha`). |
| `bs_qat_lora_dropout` | `Optional[float]` | `null` | Dropout probability for `internal_qat` (falls back to `lora_dropout`). |
| `bs_qat_freeze_non_lora` | `bool` | `true` | In `internal_qat` mode, freeze all parameters except `BitLinear.lora_A`/`lora_B`. |
| `bs_qat_target_names` | `Optional[list[str]]` | `null` | Target linear module names (defaults to `q/k/v/o_proj`, `gate/up/down_proj`). Uses exact dot-segment matching (`_matches_target`). |
| `bs_qat_skip_names` | `list[str]` | `["lm_head"]` | Linear module names to skip (never fake-quantized). |

---

## 6. Example YAML Configs & Export Pipeline

### 6.1. Internal-QAT LoRA (`examples/bs_bonsai_internal_qat_lora_non_reasoning.yaml`)

```yaml
base_model: prism-ml/Ternary-Bonsai-4B-unpacked

plugins:
  - axolotl.integrations.bs_qat.BSQATPlugin

# Leave `adapter:` unset for internal_qat
bs_qat: true
bs_qat_mode: ternary
bs_qat_group_size: 128
bs_qat_act_quant: false
bs_qat_lora_mode: internal_qat
bs_qat_lora_r: 64
bs_qat_lora_alpha: 128
bs_qat_lora_dropout: 0.0
bs_qat_freeze_non_lora: true
bs_qat_target_names:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bs_qat_skip_names:
  - lm_head
```

### 6.2. Baking & Exporting to `Q2_0` GGUF

```bash
# Step 1: Bake trained checkpoint (or internal_qat sidecar) onto the exact ternary grid as a standard HF model
python scripts/export_bs_qat_lora_baked_hf.py \
  --base-model prism-ml/Ternary-Bonsai-4B-unpacked \
  --checkpoint outputs/bs-bonsai-internal-qat/checkpoint-500 \
  --lora-mode internal_qat \
  --output-dir outputs/baked_hf

# Step 2: Convert baked HF directory to F16 GGUF using Prism ML's llama.cpp fork
python ../llama.cpp/convert_hf_to_gguf.py \
  outputs/baked_hf \
  --outtype f16 \
  --outfile outputs/model-f16.gguf

# Step 3: Repack target projections into Q2_0 (keeping output.weight / embeddings in F16 by default)
python -m axolotl.integrations.bs_qat.bitnet_qat.export.requantize_gguf_v2 \
  outputs/model-f16.gguf \
  outputs/model-Q2_0.gguf \
  --manifest outputs/requantize_manifest.json
```

### 6.3. Running Unit Tests

```bash
pytest --noconftest -o addopts="" tests/integrations/test_bs_qat.py
```
