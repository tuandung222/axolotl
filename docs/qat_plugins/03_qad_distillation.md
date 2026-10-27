# 3. `QAD`: Fixed-Scale & LSQ `W4A16` Quantization-Aware Distillation

- **Plugin Class:** `axolotl.integrations.qad.QADPlugin`
- **Trainer Class:** `axolotl.integrations.qad.trainer.AxolotlQADTrainer`
- **Config Schema:** `axolotl.integrations.qad.args.QADArgs`
- **Source Directory:** `src/axolotl/integrations/qad/`
- **Companion Scripts:** `scripts/qad/{prepare_init.py,prepare_init_rtn.py,export_w4a16.py,verify_export.py,to_vllm_layout.py}`

---

## 1. Overview & Motivation

When initializing a 4-bit student model from a calibrated Post-Training Quantization (PTQ) checkpoint (such as GPTQ `W4A16` `group_size=128` symmetric in `compressed-tensors` format), standard dynamic QAT (e.g., TorchAO `scale = max(|w|) / 7.5`) recomputes group scales from the clipped weights on step 0. Because GPTQ clips outliers when optimizing the Hessian-weighted reconstruction error, recomputing `max(|w|) / 7.5` shifts 100% of the group scales (by up to 20.5%) and causes **~85% of the weights to jump to a different integer bin at step 0** before any gradient update occurs.

`QADPlugin` solves this by:
1. Extracting the exact PTQ per-group scales (`qad_scales.safetensors`) and unpacked BF16 student weights (`prepare_init.py`) so that **`fake_quant(W_ptq) == W_ptq` bit-identically across 100% of tensors at step 0**.
2. Supporting optional **Learned Step Size Quantization (`qad_learn_scale: true`)** via a zero-initialized log-multiplier `fq_logs` ($s_{\text{eff}} = s_{\text{ptq}} \cdot \exp(\text{fq\_logs})$) and **Internal LoRA (`qad_internal_lora: true`)** inside the fixed-scale quantizer (`FQLoRALinear`).
3. Training the student with **in-process BF16 teacher distillation** (`AxolotlQADTrainer`) using memory-efficient **chunked KL divergence (`chunked_kl_loss`)** with activation checkpointing over large vocabularies ($V = 151,936$ to $248,320$).

---

## 2. Mathematical Formulation

### 2.1. Fixed-Scale & LSQ Symmetric `INT4` Fake Quantization (`FQLinear`)

For a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ partitioned into groups $\mathbf{w}_g \in \mathbb{R}^G$ ($G = 128$) and fixed PTQ group scale $s_g > 0$:
- **Fixed-Scale Mode (`qad_learn_scale: false`):**
  $$q_{g,i} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w_{g,i}}{s_g}\right), -8, 7\right), \qquad \hat{w}_{g,i} = q_{g,i} \cdot s_g$$
  where $\operatorname{round}(\cdot)$ is IEEE-754 round-half-to-even (`torch.round`), matching `compressed-tensors` `pack-quantized` format.
- **Learned-Scale LSQ Mode (`qad_learn_scale: true`):**
  Each group has a trainable parameter $\ell_g = \text{fq\_logs}_g$ initialized to $0$:
  $$s_{\text{eff},g} = s_g \cdot e^{\ell_g}, \qquad \hat{w}_{g,i} = \operatorname{STE}\!\left(\frac{w_{g,i}}{s_{\text{eff},g}}\right) \cdot s_{\text{eff},g}$$
  So gradients flow both to $w_{g,i}$ (with unit gain) and to $\ell_g$ via the standard LSQ residual $(q_{g,i} - w_{g,i}/s_{\text{eff},g}) s_{\text{eff},g}$.

### 2.2. Internal LoRA on Fixed INT4 Grid (`FQLoRALinear`)

When `qad_internal_lora: true`, base weights $W$ and scales $s_g$ are frozen, and low-rank adapters $A \in \mathbb{R}^{r \times d_{\text{in}}}, B \in \mathbb{R}^{d_{\text{out}} \times r}$ are quantized *inside* the fixed INT4 grid on every forward pass:
$$\hat{W} = Q_{s}\!\left(W + \frac{\alpha}{r} B A\right), \qquad Y = X \hat{W}^\top + \mathbf{b}$$
At export time (`export_w4a16.py`), $W + \frac{\alpha}{r} B A$ is quantized onto the exact same PTQ scale grid $s_g$, producing a pure `W4A16` `compressed-tensors` checkpoint with zero runtime LoRA overhead.

### 2.3. Chunked KL Divergence Distillation (`chunked_kl_loss`)

Given student hidden states $H_S \in \mathbb{R}^{B \times T \times d}$ and teacher hidden states $H_T \in \mathbb{R}^{B \times T \times d}$, only supervised assistant tokens ($M_{\text{sup}}$ where `shift_labels != -100`, count $N_{\text{sup}}$) are gathered before projecting through `lm_head`, avoiding computation on prompt/padding tokens. The $N_{\text{sup}}$ active tokens are processed in chunks of size $C = \text{qad\_chunk\_tokens}$ (default `1024`) under `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`:

Let $\mathbf{z}_S = \text{lm\_head}_S(\mathbf{h}_S) / \tau$ and $\mathbf{z}_T = \text{lm\_head}_T(\mathbf{h}_T) / \tau$ at temperature $\tau = \text{qad\_temperature}$:
1. **Full-Vocabulary Mode (`qad_topk: null`):**
   $$\mathcal{L}_{\text{KL}} = \tau^2 \sum_{v=1}^V P_T(v) \big(\log P_T(v) - \log P_S(v)\big)$$
2. **Top-$k$ `renorm` Mode (`qad_topk: k`, `qad_topk_mode: "renorm"`):**
   Let $\mathcal{K}_i = \operatorname{arg\,top-}k(\mathbf{z}_{T,i})$ be the indices of the top-$k$ teacher logits for token $i$. Renormalizing over $\mathcal{K}_i$:
   $$\tilde{P}_T(v) = \operatorname{softmax}_{v \in \mathcal{K}_i}(\mathbf{z}_{T,i,v}), \qquad \tilde{P}_S(v) = \operatorname{softmax}_{v \in \mathcal{K}_i}(\mathbf{z}_{S,i,v}), \qquad \mathcal{L}_{\text{KL}}^{\text{renorm}} = \tau^2 \sum_{v \in \mathcal{K}_i} \tilde{P}_T(v) \big(\log \tilde{P}_T(v) - \log \tilde{P}_S(v)\big)$$
3. **Top-$k$ `truncated` Mode (`qad_topk: k`, `qad_topk_mode: "truncated"`):**
   Uses un-renormalized full-vocabulary log-probabilities restricted to $v \in \mathcal{K}_i$ (note: empirically, `truncated` top-$k$ allows student probability mass to leak outside $\mathcal{K}_i$; full-vocabulary KL or `renorm` is recommended).

Total loss combines KL and optional ground-truth Cross-Entropy:
$$\mathcal{L} = \alpha_{\text{KL}} \mathcal{L}_{\text{KL}} + \alpha_{\text{CE}} \mathcal{L}_{\text{CE}}$$

---

## 3. GPU Forward & Backward Computational Optimizations

1. **Zero-Allocation `_STEGate` & `@torch.no_grad()` Fixed-Scale Quantizer:**
   - `fake_quant(w, scale, group_size)` runs the round/clamp/dequantize steps inside `@torch.no_grad()` and attaches the gradient via `_STEGate.apply(w, w_q)`, eliminating 5 full-matrix intermediate autograd buffers and preventing `bfloat16` `w + (wq - w).detach()` cancellation.
   - `apply_fixed_scale_fq` detaches `scale` (`scale.detach().to(...)`) before registering `fq_scale`, preventing stale autograd graph errors if `scale` was derived from a tensor with `requires_grad=True`.
2. **Fused `torch.addmm` + 100% In-Place Quantization + Rank-$r$ Backward in `_FQLoRALinearSTE`:**
   - Forward computes `v = torch.addmm(weight.float(), lora_B.float(), lora_A.float(), alpha=lora_scale)` in a single cuBLAS call, then quantizes `v` **100% in-place** (`vg.div_(s).round_().clamp_(QMIN, QMAX).mul_(s)`), reducing full-matrix FP32 allocations from 8 down to 1.
   - Backward computes `grad_lora_A` and `grad_lora_B` via rank-$r$ factored matmuls on BF16 Tensor Cores without materializing `grad_W`.
   - On an NVIDIA H100 PCIe (`T=4096, 6144x2048, r=16`), this achieves a **1.45x speedup** (`4.777 ms -> 3.284 ms`) and reduces peak delta VRAM from `272.6 MB` to **`104.1 MB` (-168.5 MB saved per layer)**.
3. **Elimination of Full-Vocabulary `log_softmax` in Top-$k$ `renorm` KL Loss (`kl.py`):**
   - Previously, `_kl_chunk` computed `F.log_softmax` over the entire vocabulary `[chunk_tokens, V]` for both teacher and student *before* branching on `topk`, and then called `F.log_softmax` a second time on the gathered `[chunk_tokens, k]` slice when `topk_mode == "renorm"`.
   - By softmax shift invariance ($\operatorname{log\_softmax}(\mathbf{z}_{\mathcal{K}} - \log\sum_v e^{z_v}) = \operatorname{log\_softmax}(\mathbf{z}_{\mathcal{K}})$), the two `[chunk_tokens, V]` `log_softmax` tensors were 100% redundant in `renorm` mode.
   - Skipping full-vocab `log_softmax` in `renorm` mode and using `F.kl_div(s_lp, t_lp, reduction="none", log_target=True)` saves **885.0 MB of peak VRAM** (`1779.7 MB -> 894.8 MB`) at $V = 151,936$ (`chunk_tokens = 1024`).

---

## 4. Configuration Reference (`QADArgs`)

| YAML Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qad_trainer` | `Optional[bool]` | `null` | When `true`, swaps the trainer class to `AxolotlQADTrainer`. |
| `qad_scales_path` | `Optional[str]` | `null` | Path to `qad_scales.safetensors` produced by `scripts/qad/prepare_init.py`. |
| `qad_group_size` | `Optional[int]` | `128` | Quantization group size $G$ along `in_features`. |
| `qad_teacher_path` | `Optional[str]` | `null` | Path to the dense BF16 teacher model directory loaded inside `AxolotlQADTrainer`. |
| `qad_kl_alpha` | `Optional[float]` | `1.0` | Weight $\alpha_{\text{KL}}$ on the KL distillation loss. |
| `qad_ce_alpha` | `Optional[float]` | `0.0` | Weight $\alpha_{\text{CE}}$ on the ground-truth token Cross-Entropy loss. |
| `qad_temperature` | `Optional[float]` | `1.0` | Distillation temperature $\tau$. |
| `qad_chunk_tokens` | `Optional[int]` | `1024` | Number of active assistant tokens per activation-checkpointed KL chunk. |
| `qad_learn_scale` | `Optional[bool]` | `false` | Enable per-group trainable LSQ scale multiplier (`fq_logs`). |
| `qad_internal_lora` | `Optional[bool]` | `false` | Wrap `FQLinear` in `FQLoRALinear` (`Q(W + (alpha/r) * B @ A)`) and freeze base weights. |
| `qad_lora_r` | `Optional[int]` | `64` | Internal LoRA rank $r$ when `qad_internal_lora: true`. |
| `qad_lora_alpha` | `Optional[float]` | `128.0` | Internal LoRA alpha when `qad_internal_lora: true`. |
| `qad_topk` | `Optional[int]` | `null` | `null` for full-vocab KL, or integer $k$ for top-$k$ teacher logit KL. |
| `qad_topk_mode` | `Optional[str]` | `"truncated"` | `"renorm"` (recommended for top-$k$, avoids full-vocab softmax) or `"truncated"`. |
| `qad_freeze_head_and_embed` | `Optional[bool]` | `true` | Freeze `lm_head.weight` and `embed_tokens` during full-weight QAD so output embeddings stay aligned with the teacher and PTQ grid. |

---

## 5. End-to-End Workflow (PTQ Init $\rightarrow$ QAD Training $\rightarrow$ vLLM W4A16 Export)

### Step 1: Unpack PTQ Checkpoint into Dense Student + Fixed Scales

```bash
python scripts/qad/prepare_init.py \
  --ptq-dir /path/to/qwen35_4b_gptq_w4a16 \
  --untied-dir /path/to/qwen35_4b_bf16_untied_teacher \
  --out-dir artifacts/student_init_4b
```
This writes `model.safetensors` (unpacked BF16 weights on the exact INT4 grid), `qad_scales.safetensors`, and `qad_meta.json`, verifying zero grid-parity error (`fake_quant(W) == W`).

### Step 2: Train with `QADPlugin`

```yaml
base_model: artifacts/student_init_4b
plugins:
  - axolotl.integrations.qad.QADPlugin

qad_trainer: true
qad_scales_path: artifacts/student_init_4b/qad_scales.safetensors
qad_teacher_path: /path/to/qwen35_4b_bf16_untied_teacher
qad_group_size: 128
qad_kl_alpha: 1.0
qad_ce_alpha: 0.0
qad_temperature: 1.0
qad_chunk_tokens: 1024
qad_learn_scale: false
qad_internal_lora: false
qad_freeze_head_and_embed: true

learning_rate: 5e-5
bf16: true
gradient_checkpointing: true
```

### Step 3: Repack into `compressed-tensors` W4A16 & Verify for vLLM

```bash
# Export trained checkpoint back to compressed-tensors pack-quantized W4A16
python scripts/qad/export_w4a16.py \
  --ckpt-dir outputs/qad_run/checkpoint-300 \
  --ptq-ref-dir /path/to/qwen35_4b_gptq_w4a16 \
  --scales-path artifacts/student_init_4b/qad_scales.safetensors \
  --out-dir artifacts/exported_w4a16

# Verify bit-exactness of packed weights
python scripts/qad/verify_export.py \
  --ckpt-dir outputs/qad_run/checkpoint-300 \
  --export-dir artifacts/exported_w4a16 \
  --scales-path artifacts/student_init_4b/qad_scales.safetensors
```

### Step 4: Running Unit Tests

```bash
pytest --noconftest -o addopts="" tests/integrations/test_qad.py
```
