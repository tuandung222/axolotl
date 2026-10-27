# Bonsai-2 (`BS2_QAT`) Hadamard + `PQ2_0` Ternary QAT Plugin for Axolotl

`axolotl.integrations.bs2_qat.BS2QATPlugin` implements Quantization-Aware Training (QAT) specifically designed for the **`Ternary-Bonsai-2-{2B, 4B, 9B, 27B}`** (`Qwen3.5`) family (`tuandunghcmut/QW-3.5-BS-2-{2B,4B,9B}-bf16`).

Unlike Bonsai v1 (`bs_qat`), **Bonsai v2** rotates every linear projection along its input dimension using a block-wise **Sylvester Walsh-Hadamard Transform** ($\mathcal{H}_B$) combined with a deterministic per-channel sign vector ($\mathbf{s} \in \{-1, +1\}^{d_{\text{in}}}$) before applying **Group-128 Least-Squares Ternary Quantization (`PQ2_0`)**.

---

## 1. Mathematical Formulation & Computational Optimizations

For an unfolded weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (or effective LoRA-merged weight $W_{\text{eff}} = W + \frac{\alpha}{r} B A$):

1. **Forward Hadamard Rotation ($\mathcal{T}$):**
   $$\tilde{W} = \mathcal{T}(W_{\text{eff}}) = (W_{\text{eff}} \odot \mathbf{s}) \mathcal{H}_B, \qquad \mathcal{H}_B = I_{d_{\text{in}}/B} \otimes \left(\frac{1}{\sqrt{B}} H_B\right)$$
   - For $B \le 512$ (`2B`, `4B`): Computed via a single L2-cache-resident GEMM (`1.0 MB` matrix $\hat{H}_{512}$).
   - For $B > 512$ (`9B` with $B=4096$, `27B` with $B=1024$): Computed via 2D Kronecker factorization $\hat{H}_{B_1 B_2} = \hat{H}_{B_1} \otimes \hat{H}_{B_2}$ (`64x64` or `32x32`), reducing FLOPs by $32\times$ at $B=4096$.

2. **Boolean-Mask 2-Stage Least-Squares Ternary Quantization (`PQ2_0`, `group_size = 128`):**
   For each group $\mathbf{w}_g \in \mathbb{R}^{128}$ of $\tilde{W}$, let $a_i = |w_{g,i}|$, $s_{\text{init}} = \max_i a_i$, and Boolean mask $m_i = \mathbb{I}(a_i \ge 0.5 s_{\text{init}})$:
   $$s^* = \frac{\sum_{i=1}^{128} a_i m_i}{\max\!\big(1, \sum_{i=1}^{128} m_i\big)}, \qquad s_{\text{fp16}} = \operatorname{float16}(s^*), \qquad \mathbf{q}_{\text{hard}} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{\mathbf{w}_g}{s_{\text{fp16}}}\right), -1, +1\right)$$
   This reproduces **100.0000%** of all discrete trits $\{-1, 0, +1\}$ and `FP16` scales in the official `Ternary-Bonsai-2` checkpoints.

3. **Inverse Hadamard Unfolding ($\mathcal{T}^{-1}$) & Collapsed Identity Jacobian STE:**
   $$\hat{W}_{\text{hard}} = \mathcal{T}^{-1}\big(s_{\text{fp16}} \cdot \mathbf{q}_{\text{hard}}\big) = \big((s_{\text{fp16}} \cdot \mathbf{q}_{\text{hard}}) \mathcal{H}_B\big) \odot \mathbf{s}$$
   Because $\mathcal{T}$ and $\mathcal{T}^{-1}$ are orthogonal inverses ($\mathcal{T}^{-1} \circ \mathcal{I}_{\text{STE}} \circ \mathcal{T} = \mathcal{I}$), the entire forward rotation $\to$ quantization $\to$ inverse rotation runs inside `torch.no_grad()`, and the Straight-Through Estimator is attached in one outer operation:
   $$\hat{W} = W_{\text{eff}} + \operatorname{stop\_grad}\!\big(\hat{W}_{\text{hard}} - W_{\text{eff}}\big)$$
   This eliminates **100% of Hadamard transforms from the backward pass** and saves **~180 MB of Autograd VRAM per layer**.

---

## 2. Built-in Canonical Sign Vectors (`builtin_signs.py`)

Across all four official `Ternary-Bonsai-2` models (`2B`, `4B`, `9B`, `27B`), every linear layer of a given input width $d_{\text{in}} \in \{2048, 2560, 4096, 5120, 6144, 9216, 12288, 17408\}$ shares the exact same canonical sign vector $\mathbf{s}_{d_{\text{in}}}$.
All 8 canonical sign vectors are bit-packed directly inside `builtin_signs.py`, so training `tuandunghcmut/QW-3.5-BS-2-{2B,4B,9B}-bf16` requires **no external GGUF or MLX files**.

| Model | `hidden_size` | `intermediate_size` | Auto `block_size` ($B$) | Canonical Sign Widths |
| :--- | :---: | :---: | :---: | :--- |
| **`QW-3.5-BS-2-2B-bf16`** | `2048` | `6144` | `512` | `2048, 6144` |
| **`QW-3.5-BS-2-4B-bf16`** | `2560` | `9216` | `512` | `2560, 4096, 9216` |
| **`QW-3.5-BS-2-9B-bf16`** | `4096` | `12288` | `4096` | `4096, 12288` |
| **`QW-3.5-BS-2-27B-bf16`** | `5120` | `17408` | `1024` | `5120, 6144, 17408` |

---

## 3. Supported Training Modes (`bs2_qat_lora_mode`)

1. **`full_weight` (Full-Parameter Hadamard QAT):**
   Trains all target `Bonsai2LinearFakeQuant.weight` tensors directly (`adapter:` must be unset).
2. **`internal_qat` (Internal LoRA inside Hadamard + `PQ2_0` Quantizer):**
   Freezes base weights and trains `lora_A` / `lora_B` *inside* the Hadamard rotation and ternary quantizer:
   $$\hat{W} = \mathcal{T}^{-1}\!\left(Q_{\text{PQ2\_0}}\!\left(\mathcal{T}\!\left(W + \frac{\alpha}{r} B A\right)\right)\right)$$
   Automatically saves lightweight sidecars (`bs2_internal_qat_lora/bs2_internal_qat_lora.safetensors`) on every checkpoint and at the end of training.
3. **`external_peft` (Standard PEFT LoRA outside Hadamard QAT):**
   Quantizes base weights via `Bonsai2LinearFakeQuant` while Axolotl attaches standard PEFT LoRA adapters on top (`adapter: lora`).

---

## 4. Example YAML Configurations

### 4.1. Internal-QAT LoRA on `QW-3.5-BS-2-2B-bf16`
```yaml
base_model: tuandunghcmut/QW-3.5-BS-2-2B-bf16
model_type: Qwen3_5ForConditionalGeneration

plugins:
  - axolotl.integrations.bs2_qat.BS2QATPlugin

bs2_qat: true
bs2_qat_lora_mode: internal_qat
bs2_qat_lora_r: 32
bs2_qat_lora_alpha: 64
bs2_qat_group_size: 128
bs2_qat_snap_on_finish: true
```

### 4.2. Full-Weight Hadamard QAT on `QW-3.5-BS-2-2B-bf16`
```yaml
base_model: tuandunghcmut/QW-3.5-BS-2-2B-bf16
model_type: Qwen3_5ForConditionalGeneration

plugins:
  - axolotl.integrations.bs2_qat.BS2QATPlugin

bs2_qat: true
bs2_qat_lora_mode: full_weight
bs2_qat_group_size: 128
bs2_qat_snap_on_finish: true
```
