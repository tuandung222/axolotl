# 1. `BS2_QAT`: Bonsai-2 Sylvester Walsh-Hadamard + `PQ2_0` Ternary QAT

- **Plugin Class:** `axolotl.integrations.bs2_qat.BS2QATPlugin`
- **Config Schema:** `axolotl.integrations.bs2_qat.args.BS2QATArgs`
- **Source Directory:** `src/axolotl/integrations/bs2_qat/`
- **Target Model Family:** `Ternary-Bonsai-2-{2B, 4B, 9B, 27B}` (`Qwen3.5` architecture; unfolded BF16 checkpoints at `tuandunghcmut/QW-3.5-BS-2-{2B,4B,9B}-bf16`)

---

## 1. Overview & Motivation

Unlike Bonsai v1 (`bs_qat`), which applies group-wise ternary quantization directly to raw weight matrices, **Bonsai-2 (`Ternary-Bonsai-2`)** suppresses activation and weight outliers by rotating each linear projection along its input dimension $d_{\text{in}}$ using a randomized **block-wise Sylvester Walsh-Hadamard Transform** ($\mathcal{T}$) prior to **Group-128 Least-Squares Ternary Quantization (`PQ2_0`)**.

In the official `Ternary-Bonsai-2` GGUF and MLX checkpoints, weights are stored in the rotated Hadamard domain $\tilde{W} = \mathcal{T}(W)$, and the inference engine applies $\mathcal{T}(\mathbf{x})$ to the input activation vector before each matrix multiplication. However, in `Qwen3.5-{2B, 4B, 9B}`, `tie_word_embeddings=true` ties `model.language_model.embed_tokens.weight` to `lm_head.weight`:
- `embed_tokens` performs token lookups in the **unfolded (unrotated) embedding space**.
- Rotating `embed_tokens` directly breaks token lookup unless `lm_head` is untied (adding 0.5B–1.0B parameters) or rotated at every logit projection.

`BS2QATPlugin` solves this by storing master weights in the **unfolded standard `Qwen3.5` space** by default (`bs2_qat_store_rotated: false`). During each forward pass, `Bonsai2LinearFakeQuant` rotates weights into the Hadamard domain ($\mathcal{T}$), applies bit-exact `PQ2_0` ternary quantization ($Q_{\text{PQ2\_0}}$), and unfolds the quantized weights back ($\mathcal{T}^{-1}$) before computing `F.linear(x, w_q, bias)`.

---

## 2. Mathematical Formulation

### 2.1. Randomized Block-wise Sylvester Walsh-Hadamard Transform ($\mathcal{T}$ and $\mathcal{T}^{-1}$)

Let $B \in \{512, 1024, 4096\}$ be a power-of-2 block size dividing $d_{\text{in}}$, and let $\mathbf{s} \in \{-1, +1\}^{d_{\text{in}}}$ be the canonical per-input-channel sign vector for width $d_{\text{in}}$. Let $H_B \in \{-1, +1\}^{B \times B}$ denote the Sylvester-construction Hadamard matrix:
$$H_1 = [1], \qquad H_{2k} = \begin{bmatrix} H_k & H_k \\ H_k & -H_k \end{bmatrix}, \qquad \hat{H}_B = \frac{1}{\sqrt{B}} H_B$$
Since $\hat{H}_B^\top = \hat{H}_B$ and $\hat{H}_B^2 = I_B$, $\hat{H}_B$ is symmetric and orthogonal. Writing $\mathcal{H}_B = I_{d_{\text{in}}/B} \otimes \hat{H}_B$, the forward and inverse transforms on a row vector $\mathbf{w} \in \mathbb{R}^{d_{\text{in}}}$ are:
$$\tilde{\mathbf{w}} = \mathcal{T}(\mathbf{w}) = (\mathbf{w} \odot \mathbf{s}) \mathcal{H}_B, \qquad \mathcal{T}^{-1}(\tilde{\mathbf{w}}) = (\tilde{\mathbf{w}} \mathcal{H}_B) \odot \mathbf{s}$$
Because $\mathcal{H}_B^2 = I_{d_{\text{in}}}$ and $\mathbf{s} \odot \mathbf{s} = \mathbf{1}$, $\mathcal{T}^{-1}(\mathcal{T}(\mathbf{w})) = \mathbf{w}$ and $\|\mathcal{T}(\mathbf{w})\|_2 = \|\mathbf{w}\|_2$.

### 2.2. `PQ2_0` 2-Stage Boolean-Mask Least-Squares Ternary Quantizer (`group_size = 128`)

In the rotated domain, each row of $\tilde{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is partitioned into contiguous groups $\mathbf{w}_g \in \mathbb{R}^{128}$ (`G = 128`). Unlike `Q2_0` (`round(mean(|w|) * 1.5)`), `PQ2_0` computes the exact least-squares scale over active ternary elements:

1. **Initial Absmax Thresholding:**
   $$a_i = |w_{g,i}|, \qquad s_{\text{init}} = \max_{1 \le i \le 128} a_i, \qquad m_i = \mathbb{I}\!\left(a_i \ge \tfrac{1}{2} s_{\text{init}}\right) \in \{0, 1\}$$
   The Boolean condition $a_i \ge 0.5 s_{\text{init}}$ is algebraically identical to $\operatorname{round}(a_i / s_{\text{init}}) \ne 0$ for all $s_{\text{init}} > 0$ and avoids division by zero on all-zero groups.
2. **Least-Squares Scale & FP16 Cast:**
   $$s^* = \underset{s}{\arg\min} \sum_{i=1}^{128} \big(w_{g,i} - s \cdot \operatorname{sign}(w_{g,i}) m_i\big)^2 = \frac{\sum_{i=1}^{128} a_i m_i}{\max\!\big(1, \sum_{i=1}^{128} m_i\big)}, \qquad s_{\text{fp16}} = \operatorname{float16}(s^*)$$
3. **Hard Ternary Assignment & Dequantization:**
   $$q_{g,i} = \begin{cases} \operatorname{clamp}\!\left(\operatorname{round}\!\left(\dfrac{w_{g,i}}{s_{\text{fp16}}}\right), -1, +1\right) & \text{if } s_{\text{fp16}} > 0 \\ 0 & \text{if } s_{\text{fp16}} = 0 \end{cases}, \qquad \hat{\mathbf{w}}_g = s_{\text{fp16}} \cdot \mathbf{q}_g$$

Empirical verification against the official `prism-ml/Ternary-Bonsai-2-{2B, 4B, 9B, 27B}` GGUF and MLX checkpoints confirms that this formula matches **100.0000%** of all discrete trits $\{-1, 0, +1\}$ and FP16 group scales across all 240 linear layers.

---

## 3. GPU Forward & Backward Computational Optimizations

### 3.1. Collapsed Identity Jacobian STE ($\mathcal{T}^{-1} \circ \mathcal{I}_{\text{STE}} \circ \mathcal{T} = \mathcal{I}$)

In unfolded mode (`bs2_qat_store_rotated: false`), the forward operator is:
$$\hat{W} = \mathcal{T}^{-1}\!\Big(Q_{\text{PQ2\_0}}\big(\mathcal{T}(W_{\text{eff}})\big)\Big)$$
Under the Straight-Through Estimator (STE), the Jacobian of $Q_{\text{PQ2\_0}}$ is approximated by the identity matrix $\mathcal{I}$. By the chain rule, the composite Jacobian with respect to $W_{\text{eff}}$ is:
$$\frac{\partial \hat{W}}{\partial W_{\text{eff}}} = \mathcal{T}^{-1} \circ \mathcal{I} \circ \mathcal{T} = \mathcal{I}$$
Consequently:
- `Bonsai2LinearFakeQuant` runs the entire sequence $\mathcal{T} \rightarrow Q_{\text{PQ2\_0}} \rightarrow \mathcal{T}^{-1}$ inside `@torch.no_grad()` (`_fake_quant_weight_no_grad`).
- This eliminates **100% of Hadamard transforms from the backward pass** and frees all intermediate Hadamard activation buffers (~180 MB per layer) immediately during forward execution.

### 3.2. Zero-Allocation `_STEGate` (Fixing `bfloat16` Cancellation Drift)

Instead of attaching STE via `w + (wq - w).detach()` (which suffers from `bfloat16` subtraction/addition cancellation on **93.78%** of output elements and allocates two `[d_out, d_in]` tensors), `bs2_qat` uses `_STEGate.apply(w, wq)`:
```python
class _STEGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
        return w_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None
```
This guarantees **100.0000% bit-identical outputs** between QAT training and snapped `nn.Linear` inference while launching zero CUDA kernels.

### 3.3. L2-Cached Block GEMM ($B \le 512$) & 2D Kronecker Factorization ($B > 512$)

Instead of executing $\log_2(B) = 9$ to $12$ sequential memory-bound butterfly kernels in PyTorch:
- For **$B \le 512$** (`2B` and `4B` models): `_get_normalized_hadamard(512)` caches $\hat{H}_{512}$ in `float32` (`1.0 MB`, fitting inside the H100 50 MB L2 cache) and computes `torch.matmul(x_blocks, H_512)` in a single GEMM.
- For **$B > 512$** (`9B` with $B=4096 = 64 \times 64$, `27B` with $B=1024 = 32 \times 32$): `_split_kronecker_factors(B)` factors $\hat{H}_B = \hat{H}_{B_1} \otimes \hat{H}_{B_2}$ and executes two small GEMMs (`torch.matmul(H_1, x_2d)` and `torch.matmul(..., H_2)`). At $B=4096$, this reduces arithmetic operations by **$32\times$** compared to a dense $4096 \times 4096$ matmul and avoids the `64 MB` matrix allocation.
- Input sign multiplication `x * s_f32` promotes `bfloat16` to `float32` in a single fused elementwise kernel, and inverse sign scaling `out_f32.mul_(s_f32)` runs in-place before casting back to `x.dtype`.

### 3.4. In-Place Training Dequantizer (`_ternary_dequant_pq2_0_no_grad`)

During training, the discrete `int8` tensor `q_hard` and `scales_fp16` do not need to be returned separately. `_ternary_dequant_pq2_0_no_grad` computes `inv_scales = torch.where(scales_f32 > 0, 1.0 / scales_f32, 0.0)` at the `[rows, num_groups, 1]` level (384 KB instead of 48 MB) and executes `(w_g * inv_scales).round_().clamp_(-1.0, 1.0).mul_(scales_f32)` in-place, eliminating four `[d_out, d_in]` FP32 temporary buffers and the `float32 -> int8` cast.

### 3.5. Rank-$r$ Factored Backward (`_BS2InternalLoRALinearSTE`) & Frozen-Weight Cache

- In `internal_qat` mode (`use_lora=True`, `weight.requires_grad=False`), `_BS2InternalLoRALinearSTE` computes $W_{\text{eff}}$ via fused `torch.addmm` inside `torch.no_grad()`, quantizes it via `_fake_quant_weight_no_grad`, and in `backward` projects `grad_output` through `lora_B` (`[T, r]`) to compute `grad_lora_A` and `grad_lora_B` on BF16 Tensor Cores without ever materializing `grad_W` (`[d_out, d_in]`).
- In `external_peft` or `eval()` mode (`use_lora=False`, `weight.requires_grad=False`), `Bonsai2LinearFakeQuant` caches the quantized weight tensor keyed by `(weight._version, signs._version, weight.dtype, weight.device)`, reducing per-step Hadamard + quantization overhead to zero.

---

## 4. Built-in Canonical Sign Vectors & Model Architecture Support

Across all four official `Ternary-Bonsai-2` models (`2B`, `4B`, `9B`, `27B`), every rotated projection of a given input width $d_{\text{in}}$ shares the exact same canonical sign vector $\mathbf{s}_{d_{\text{in}}}$. All 8 canonical widths (`2048, 2560, 4096, 5120, 6144, 9216, 12288, 17408`) are packed in `src/axolotl/integrations/bs2_qat/builtin_signs.py`.

| Model | `hidden_size` | `intermediate_size` | Auto `block_size` ($B$) | Canonical Input Widths ($d_{\text{in}}$) |
| :--- | :---: | :---: | :---: | :--- |
| **`Ternary-Bonsai-2-2B`** | `2048` | `6144` | `512` | `2048, 6144` |
| **`Ternary-Bonsai-2-4B`** | `2560` | `9216` | `512` | `2560, 4096, 9216` |
| **`Ternary-Bonsai-2-9B`** | `4096` | `12288` | `4096` | `4096, 12288` |
| **`Ternary-Bonsai-2-27B`** | `5120` | `17408` | `1024` | `5120, 6144, 17408` |

### Default Target & Skip Rules on `Qwen3.5` Hybrid Gated-DeltaNet + Attention

- **Patched (`DEFAULT_BS2_TARGET_NAMES`):**
  - Full Attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Linear Attention (Gated DeltaNet) layers: `in_proj_qkv`, `in_proj_z`, `out_proj`
  - SwiGLU MLP layers: `gate_proj`, `up_proj`, `down_proj`
- **Skipped (`DEFAULT_BS2_SKIP_NAMES` & Scope Filter):**
  - `lm_head`, `embed_tokens` (kept in BF16; supports `tie_word_embeddings=True`)
  - `in_proj_a`, `in_proj_b` (`out_features = 16` Gated DeltaNet decay/beta gates; kept in BF16 matching official checkpoints, including zero-initialized `in_proj_qkv` rows)
  - Vision tower (`model.visual.*`) is automatically excluded by `_language_model_submodule(model)`.

---

## 5. Configuration Reference (`BS2QATArgs`)

| YAML Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bs2_qat` | `bool` | `false` | Enable the `BS2QATPlugin` integration. |
| `bs2_qat_lora_mode` | `Literal["full_weight", "internal_qat", "external_peft"]` | `"full_weight"` | Training composition mode (`full_weight`, `internal_qat`, or `external_peft`). |
| `bs2_qat_group_size` | `int` | `128` | Group size $G$ for `PQ2_0` least-squares ternary quantization. |
| `bs2_qat_block_size` | `Optional[int]` | `null` | Sylvester Walsh-Hadamard block size $B$. Auto-inferred from `hidden_size` when `null` (`2B/4B -> 512`, `9B -> 4096`, `27B -> 1024`). |
| `bs2_qat_signs_path` | `Optional[str]` | `null` | Path to `hadamard.json`, MLX directory, or GGUF file. Uses built-in canonical sign table when `null` or `"builtin"`. |
| `bs2_qat_signs_seed` | `int` | `0` | Fallback RNG seed for non-standard layer widths not in the sign table. |
| `bs2_qat_store_rotated` | `bool` | `false` | Keep `false` to store unfolded weights (enables Collapsed Identity Jacobian backward). Set `true` only when loading pre-rotated weights directly. |
| `bs2_qat_lora_r` | `Optional[int]` | `null` | Rank $r$ for `internal_qat` LoRA (falls back to `lora_r` or `16`). |
| `bs2_qat_lora_alpha` | `Optional[float]` | `null` | Alpha scaling for `internal_qat` LoRA (falls back to `lora_alpha` or `32.0`). |
| `bs2_qat_lora_dropout` | `Optional[float]` | `null` | Dropout probability for `internal_qat` LoRA (falls back to `lora_dropout` or `0.0`). |
| `bs2_qat_freeze_non_lora` | `bool` | `true` | In `internal_qat` mode, freeze all parameters except `lora_A` and `lora_B`. |
| `bs2_qat_snap_on_finish` | `bool` | `false` | At `post_train`, snap all `Bonsai2LinearFakeQuant` layers onto the exact `PQ2_0` grid in the Hadamard domain, unfold via $\mathcal{T}^{-1}$, and replace with standard `nn.Linear`. |
| `bs2_qat_target_names` | `Optional[list[str]]` | `null` | Override list of linear child-module names to patch. |
| `bs2_qat_skip_names` | `list[str]` | `["lm_head", "in_proj_a", "in_proj_b"]` | Child-module names to keep as unquantized `nn.Linear`. |

---

## 6. Example YAML Configurations & Workflows

### 6.1. Internal-QAT LoRA (`examples/bs2_bonsai_2b_internal_qat_lora.yaml`)

```yaml
base_model: tuandunghcmut/QW-3.5-BS-2-2B-bf16
model_type: Qwen3_5ForConditionalGeneration
trust_remote_code: true

plugins:
  - axolotl.integrations.bs2_qat.BS2QATPlugin

# Do NOT set `adapter: lora` in internal_qat mode
bs2_qat: true
bs2_qat_lora_mode: internal_qat
bs2_qat_lora_r: 32
bs2_qat_lora_alpha: 64
bs2_qat_lora_dropout: 0.0
bs2_qat_group_size: 128
bs2_qat_block_size: 512
bs2_qat_signs_path: builtin
bs2_qat_store_rotated: false
bs2_qat_snap_on_finish: true

bf16: true
tf32: true
gradient_checkpointing: true
```

During training, `BS2InternalQATLoRASaveCallback` automatically saves `bs2_internal_qat_lora/bs2_internal_qat_lora.safetensors` alongside every Trainer checkpoint and into `output_dir` at training completion. When `bs2_qat_snap_on_finish: true`, the final saved model in `output_dir` contains standard `nn.Linear` weights that are already snapped onto the Hadamard `PQ2_0` ternary grid.

### 6.2. Full-Weight Hadamard QAT (`examples/bs2_bonsai_2b_full_qat.yaml`)

```yaml
base_model: tuandunghcmut/QW-3.5-BS-2-2B-bf16
model_type: Qwen3_5ForConditionalGeneration
trust_remote_code: true

plugins:
  - axolotl.integrations.bs2_qat.BS2QATPlugin

bs2_qat: true
bs2_qat_lora_mode: full_weight
bs2_qat_group_size: 128
bs2_qat_block_size: 512
bs2_qat_signs_path: builtin
bs2_qat_store_rotated: false
bs2_qat_snap_on_finish: true
```

### 6.3. Running Unit Tests

```bash
pytest --noconftest -o addopts="" tests/integrations/test_bs2_qat.py
```
