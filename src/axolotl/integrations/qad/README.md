# QAD W4A16 cho Qwen3.5-4B v7 r64 trên Axolotl — quy trình, kết quả, bài học

**Ngày giờ lập tài liệu:** 2026-10-07 05:15:30 (EDT)
**Repo:** `/home/dungvpt/qad_v7_axolotl` (git) · **Trạng thái:** huấn luyện 2 thí nghiệm đã xong (307/307 bước); đánh giá checkpoint cuối đang chạy khi viết; các mục "chưa kiểm chứng" ở §8.

---

## 1. Tóm tắt

**Mục tiêu:** QAD (quantization-aware distillation) đưa model BF16 xuống INT4 W4A16 (đối xứng, group 128, chuẩn compressed-tensors/vLLM), **khởi tạo student từ chính checkpoint PTQ (GPTQ)**, dùng Axolotl (không dùng ModelOpt; chỉ tham khảo quy trình 3 bước của họ).

| | |
|---|---|
| Student khởi tạo | `outputs/quantized/qwen35_4b_v7_r64_gptq_w4a16_calA_fixed_1024` (GPTQ W4A16 g128 sym; calibrate bằng bộ 1024 đã sửa lỗi chỉ số dòng) |
| Teacher | `outputs/merged/qwen35_4b_v7_r64_untied` (BF16, text-only, `lm_head` tách riêng) |
| Dữ liệu | `data/train/train_smart_editor_agent_thinking_v7_20260924.jsonl`: 16.652 mẫu, 152,7 triệu token (trung vị 8.783, p99 19.097, tối đa 32.237 token/mẫu); en 60%, vi 14%, zh 7%, còn lại es/fr/de/ru/pt/ar |
| Loss | KL(teacher ‖ student) trên token của assistant (forward KL, mode-covering), tính chunked trên toàn vocab 248k |
| Hai thí nghiệm | **A:** KL toàn vocab · **B:** KL chỉ top-10 token của teacher (`truncated`); cùng LR 5e-5, cùng dữ liệu, mỗi cái 1 GPU |

**Kết quả chính (số liệu ở §6, mọi chênh lệch cần đọc kèm §8):**
- A (KL toàn vocab) cải thiện val PPL so với PTQ: 2,4520 (PTQ) → 2,4455 (bước 15) → **2,4266 (bước 195)** → **2,4328 (bước 307, cuối)**. Điểm tốt nhất là bước 195, không phải bước cuối; PPL xấu đi 0,0062 từ 195 sang 307 trong khi loss huấn luyện vẫn giảm (chưa rõ là nhiễu hay overfit, xem §8).
- B (top-10) **làm tệ hơn** PTQ và tệ dần: 2,4645 (bước 15) → 2,4965 (195) → 2,5016 (307), dù loss của chính nó giảm mạnh hơn.
- Tool-calling (bộ agent-fidelity, 300 prompt) gần như không đổi giữa PTQ và QAD bước 195; chênh lệch nằm trong ngưỡng nhiễu.
- Export → vLLM W4A16 chạy được (kernel `MacheteLinearKernel`), cả layout text-only lẫn layout VLM.

---

## 2. Cấu trúc repo

```
src/qad/
  fixed_scale_fq.py   fake-quant INT4 trên scale CỐ ĐỊNH (STE); FQLinear; apply_fixed_scale_fq
  ptq_io.py           đọc checkpoint compressed-tensors W4A16 -> trọng số dense BF16 + scale
  kl.py               KL chunked (toàn vocab hoặc top-k; mode truncated|renorm)
  trainer.py          AxolotlQADTrainer: teacher chạy trong tiến trình, loss = chunked KL
  plugin.py           QADPlugin cho Axolotl (hook post_model_load, get_trainer_cls, args)
configs/              qad_4b_r64_w4a16*.yml (2 GPU), qad_4b_1gpu_{fullkl-gpu0,top10-gpu1}.yml (1 GPU), smoke, deepspeed_zero2.json
scripts/
  00_prepare_init.py      PTQ ckpt -> student dense + qad_scales.safetensors (kiểm tra parity lưới)
  10_train.sh / 11_train_1gpu.sh   huấn luyện 2 GPU / 1 GPU
  40_inspect_ckpt.py      đo tỉ lệ trọng số đổi mức INT4 so với khởi tạo (CPU)
  30_export_w4a16.py      checkpoint dense -> compressed-tensors W4A16 trên cùng lưới
  50_export_and_eval.sh   export + val PPL (cùng giao thức PTQ)
  60_to_vllm_layout.py    đổi tên tensor sang layout VLM (model.language_model.*) + config VLM
  serve_checkpoint.sh     dựng server vLLM (xem §5.6)       62_smoke_request.py   thử nhanh endpoint
  70/71/72_*              replay agent-fidelity, báo cáo, chạy cả bộ
  73_results_table.py     in bảng val PPL (markdown)
  keep_open_perms*.sh     giữ quyền a+rwX cho outputs/ (AGENTS.md)
tests/test_qad_basics.py  test CPU (pack/unpack, parity lưới, STE, KL chunked, top-k)
docs/QAD_PIPELINE.md      tài liệu này
```

## 3. Môi trường

- **Huấn luyện/đánh giá:** conda `axolotl_cu13` (torch 2.13, transformers 5.9.0 bản tùy chỉnh, deepspeed 0.18.9, flash-linear-attention 0.4.1, compressed-tensors 0.19.0). Truy cập: `PYTHONPATH=src`.
- **Serving:** conda `vllm_qwen38_cuda130` (vLLM 0.28.0).
- **Phần cứng:** 2× H200 NVL (143 GB). RAM container giới hạn 405 GB.
- **Quy ước dự án (AGENTS.md):** `umask 000`, `chmod -R a+rwX` outputs; `chat_template: tokenizer_default`; arguments của tool call là dict.

## 4. Thiết kế và lý do

1. **Scale cố định thay vì QAT động của torchao.** QAT của torchao tính lại `scale = max|w|/7.5` ở mỗi forward. Trọng số GPTQ nằm sẵn trên lưới `q·s` (q∈[-8,7]) và phần tử lớn nhất bị kẹp, nên scale tính lại khác scale PTQ. Đo trên 12 tensor thật: scale lệch ở 100% group (tối đa 20,5%) và ~85% trọng số bị re-round ngay bước 0. Ở đây scale PTQ là buffer, nên `fake_quant(W_ptq) == W_ptq` chính xác (kiểm trên cả 200 tensor, sai số 0,0). Hướng dẫn QAD của NVIDIA ModelOpt cũng nói scale từ PTQ tĩnh kiểu MSE nên giữ cố định.
2. **Module được fake-quant (200 Linear):** `mlp.{up,gate,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}_proj`. **Không quantize (BF16):** `lm_head`, `embed_tokens`, `linear_attn.in_proj_a/b`, các norm, `A_log`, `dt_bias`, `conv1d`. Danh sách do checkpoint PTQ quyết định (49 mục ignore).
3. **`lm_head` và `embed_tokens` bị đóng băng** (luôn bằng teacher; export không cần tie).
4. **Teacher chạy trong tiến trình** (BF16, đóng băng) và loss KL toàn vocab chia chunk (checkpoint theo chunk) để không dựng tensor [token × 248k].
5. **LR phải đủ lớn để trọng số vượt rào làm tròn.** Trọng số khởi tạo nằm đúng tâm ô; muốn đổi mức INT4 cần dịch ≥ ½ bước lưới (trung vị bước lưới 3,7e-3). Xem §7.1.

## 5. Quy trình từng bước

### 5.0 Điều kiện đầu vào
- Checkpoint PTQ W4A16 (GPTQ) của model, và model BF16 untied text-only làm teacher (tạo bằng `/home/dungvpt/axolotl_planning_v2/scripts/merge/merge_and_untie_v7.py --size 4b --rank 64`).

### 5.1 Chuẩn bị student (CPU, vài phút)
```bash
python scripts/00_prepare_init.py \
  --ptq-dir /home/dungvpt/axolotl_planning_v2/outputs/quantized/qwen35_4b_v7_r64_gptq_w4a16_calA_fixed_1024 \
  --untied-dir /home/dungvpt/axolotl_planning_v2/outputs/merged/qwen35_4b_v7_r64_untied \
  --out-dir artifacts/student_init_4b_r64
```
Ghi `model.safetensors` (dense BF16, 9,1 GB), `qad_scales.safetensors`, `qad_meta.json`. Script **dừng với assert** nếu: parity lưới ≠ 0, `lm_head`/`embed` khác nguồn BF16, hoặc thiếu/thừa tensor.

### 5.2 Test CPU
`PYTHONPATH=src python tests/test_qad_basics.py` (5 test, đều pass lúc viết).

### 5.3 Huấn luyện
- **1 GPU mỗi thí nghiệm (cách đã dùng):**
```bash
setsid nohup bash scripts/11_train_1gpu.sh 0 configs/qad_4b_1gpu_fullkl-gpu0.yml 29501 > logs/11_fullkl_gpu0.log 2>&1 < /dev/null &
setsid nohup bash scripts/11_train_1gpu.sh 1 configs/qad_4b_1gpu_top10-gpu1.yml 29502 > logs/11_top10_gpu1.log 2>&1 < /dev/null &
setsid nohup bash scripts/keep_open_perms2.sh >/dev/null 2>&1 < /dev/null &   # giữ quyền checkpoint
```
- **2 GPU cho một thí nghiệm:** `QAD_CONFIG=configs/qad_4b_r64_w4a16_lr5e-5.yml bash scripts/10_train.sh`.
- **Siêu tham số:** seq 32768 + sample packing (hiệu suất gộp 94,5%), global batch 16 (1 GPU: GA 16; 2 GPU: GA 8), 1 epoch = 307 bước, LR 5e-5 cosine, warmup 3% (≈9 bước, bước 1 có LR = 0), AdamW fused, weight decay 0, clip 1,0, bf16, gradient checkpointing, sdpa, DeepSpeed ZeRO-2, `save_steps: 15`, `save_total_limit: 40`.
- **Thời gian đo được:** ~112 s/bước ở 1 GPU (≈9,6 giờ cho 307 bước); ~57 s/bước ở 2 GPU. Khi 2 run chạy song song trên 2 GPU mỗi run đều dùng ~112 GB VRAM dự trữ (cache allocator; đỉnh active chưa đo ở chế độ 1 GPU, smoke 2-GPU là 66,8 GiB).
- **Quyền truy cập:** DeepSpeed ghi `model.safetensors` mode 0600; `keep_open_perms2.sh` chmod định kỳ (hoặc `chmod -R a+rwX outputs/`).

### 5.4 Kiểm tra checkpoint đã "thật sự đổi" chưa (CPU)
`python scripts/40_inspect_ckpt.py outputs/<run>/checkpoint-N` → tỉ lệ trọng số đổi mức INT4 và độ dịch/scale. Đây là chỉ số chẩn đoán LR (xem §6.2).

### 5.5 Export + đánh giá PPL
```bash
bash scripts/50_export_and_eval.sh outputs/<run>/checkpoint-N <tag> <gpu> [limit=400]   # kết quả: logs/eval-<tag>-val.json
python scripts/73_results_table.py                                                        # bảng markdown
```
Export dùng lại scale của PTQ, **không calibrate lại**; `lm_head` xuất rõ ràng (`tie_word_embeddings=false`). Đã có lần **đối chứng**: export chính student chưa train cho val PPL **2,4520 = đúng số của PTQ**, nên đường export không làm lệch số đo.

### 5.6 Serve bằng vLLM
Một lệnh (nhận checkpoint dense *hoặc* thư mục đã export):
```bash
scripts/serve_checkpoint.sh outputs/qad-4b-r64-lr5e-5-fullkl-gpu0/checkpoint-307 --gpu 0 --port 8101 --name qad-w4a16
scripts/serve_checkpoint.sh --stop 8101
```
Tuỳ chọn chính: `--mem` (mặc định 0,85; dùng ~0,17–0,2 nếu GPU đang có job khác), `--max-len` (32768), `--max-num-seqs` (64), `--layout text|vlm` (mặc định text), `--dry-run`, `--foreground`, `--no-smoke`. Script tự: export nếu là checkpoint dense → (tuỳ chọn) đổi layout VLM → kiểm tra bộ nhớ GPU trống → khởi động vLLM nền → chờ API → chạy `62_smoke_request.py` (chat thường + một prompt agent thật có tools, kiểm tra tool call được parse).
- Lệnh cốt lõi nằm trong file base `scripts/serve_vllm_base.sh` (`vllm serve <MODEL_đã_export> ...`, cấu hình bằng biến môi trường `MODEL GPU PORT NAME MEM LEN SEQS`); `serve_checkpoint.sh` dùng cùng dạng lệnh `vllm serve`. File base đã được chạy thử thật (smoke OK, `MEM=0.30`); `serve_checkpoint.sh` sau khi đổi sang `vllm serve` mới chỉ kiểm bằng `--dry-run`.
- vLLM chạy với `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` (tool call dạng XML theo AGENTS.md).
- **Layout:** text-only (`Qwen3_5ForCausalLM`, tensor `model.*`) vLLM 0.28.0 nạp trực tiếp, không cần mapping (đã chạy trong bộ fidelity). Layout VLM (`Qwen3_5ForConditionalGeneration`, `model.language_model.*`) cũng chạy, cần thêm `processor_config.json`.
- **Lỗi hay gặp:** (a) `max_num_seqs (1024) exceeds available Mamba cache blocks` khi `--mem` nhỏ → giảm `--max-num-seqs`; (b) `Free memory ... less than desired GPU memory utilization` → GPU đang bị job khác chiếm; (c) `Can't load image processor` với layout VLM → thiếu `processor_config.json`; (d) cảnh báo `deep_gemm ImportError` ở mức WARNING không chặn server (chưa xác nhận vô hại hoàn toàn).

**Đã kiểm chứng trên checkpoint bước 307 (cấu hình mặc định `--mem 0.85`, H200, vLLM 0.28.0; `scripts/74_verify_vllm.py`):**
- Smoke: prompt agent 8.601 token → tool call parse đúng.
- Parity vLLM so với PyTorch dựng từ chính checkpoint W4A16 đã export: 30/30 prompt val (≤6k token) cùng token top-1; |chênh logprob| trung bình 0,0048, lớn nhất 0,0229. Chỉ so một token sinh ra.
- Prompt dài tổng hợp: 29.821 và 32.321 token đều được xử lý (`finish=length` với max_tokens=16); lần thứ hai có thể nhờ prefix cache nên thời gian không đại diện.
- 32 yêu cầu agent đồng thời (max_tokens 512): 32/32 thành công, 25/32 có tool call được parse (số còn lại chưa xem nguyên nhân), 1.258 token/s đầu ra, p50 7,4 s.
- **Chưa thử:** GPU khác H200 (A100/Marlin), tensor parallel, tải kéo dài, so khớp cả chuỗi sinh. Chat không có tools có thể chạm giới hạn `max_tokens` khi model còn đang suy luận.

### 5.7 Agent-fidelity (tool-calling)
`bash scripts/72_run_fidelity_suite.sh` (BF16, PTQ, QAD; mỗi lượt ~95 s sinh + vài phút khởi động) rồi `python scripts/71_fidelity_report.py bf16_r64 ptq_r64 qad_fullkl_cp195`. Dùng cùng 300 prompt, greedy, tool call đầu tiên.

## 6. Kết quả

### 6.1 Val PPL (400 mẫu, `data/val/val.jsonl`; cùng giao thức với PTQ)
| Checkpoint | val PPL | top-1 vs BF16 | NLL/token |
|---|---|---|---|
| BF16 teacher (measured earlier) | 2.4350 | - | 0.88997 |
| PTQ 1024 (measured earlier) | 2.4520 | 0.9530 | 0.89692 |
| fullkl_cp15 | 2.4455 | 0.9532 | 0.89424 |
| fullkl_cp105 | 2.4430 | 0.9538 | 0.89323 |
| fullkl_cp195 | 2.4266 | 0.9545 | 0.88649 |
| fullkl_cp255 | 2.4374 | 0.9544 | 0.89094 |
| fullkl_cp307 | 2.4328 | 0.9548 | 0.88906 |
| init_control | 2.4520 | 0.9530 | 0.89692 |
| top10_cp15 | 2.4645 | 0.9536 | 0.90198 |
| top10_cp195 | 2.4965 | 0.9539 | 0.91489 |
| top10_cp307 | 2.5016 | 0.9538 | 0.91694 |

(Bảng sinh bằng `python scripts/73_results_table.py` lúc viết. Đường cong full-KL không đơn điệu (15: 2,4455 · 105: 2,4430 · 195: 2,4266 · 255: 2,4374 · 307: 2,4328): từ bước 195 trở đi dao động ~0,005 PPL, cỡ mức nhiễu chưa đo, nên chưa thể coi bước 195 là tốt hơn bước 307 một cách chắc chắn.)

### 6.2 Tỉ lệ trọng số đổi mức INT4 so với khởi tạo (3.565.158.400 trọng số quantize)

| Run / checkpoint | Đổi mức | Độ dịch TB / scale (cần > 0,5 để đổi) |
|---|---|---|
| LR 5e-6, bước 77 (lần chạy cũ, đã bỏ) | 899 (0,00003%) | 0,0139 |
| LR 5e-5 KL toàn vocab, bước 15 | 24.574 (0,0007%) | 0,0341 |
| LR 5e-5 KL toàn vocab, bước 195 | 9.731.526 (0,273%) | 0,1023 |
| LR 5e-5 KL toàn vocab, bước 307 | 9.858.209 (0,2765%) | 0,1026 |
| LR 5e-5 top-10, bước 15 / 195 / 307 | 31.408 (0,0009%) / 10.097.945 (0,283%) / 10.243.699 (0,2873%) | 0,0363 / 0,1036 / 0,1039 |

Tỉ lệ đổi mức gần như không tăng từ bước 195 đến 307 (LR cosine đã gần 0).

### 6.3 Loss KL (trung bình theo 20 bước)
Full-KL: 0,0207 (bước 1–20) → 0,0194 (61–80) → 0,0184 (121–140) → 0,0179 (181–200); 20 bước cuối ≈ 0,0167. Top-10: 0,0199 → 0,0161 → 0,0150 → 0,0141; 20 bước cuối ≈ 0,0127 (hai loss không so trực tiếp được vì top-10 bỏ phần đuôi). Loss khởi đầu ~0,02 nat/token khớp với khoảng cách ln(PPL) PTQ–BF16 trên tập train (0,0188). Trường `ppl` trong log Axolotl chỉ là `exp(loss)`, **không** phải perplexity thật.

### 6.4 Agent-fidelity (300 prompt, tool call đầu, greedy) — QAD bước 195

| Model | Có tool call | Tên tool đầu = BF16 | Mọi lời gọi = BF16 | Tên tool đầu = tham chiếu | Mọi lời gọi = tham chiếu |
|---|---|---|---|---|---|
| BF16 r64 | 98,7% | – | – | 80,0% | 52,0% |
| PTQ r64 | 98,0% | 84,7% | 59,0% | 80,7% | 52,3% |
| QAD full-KL, bước 195 | 97,3% | 88,3% | 59,7% | 79,7% | 50,3% |

Chạm giới hạn 1024 token: BF16 4, PTQ 5, QAD 7. Chưa chạy fidelity cho bước 307.

## 7. Bài học và sự cố đã gặp

1. **LR 5e-6 vô hiệu:** loss gần như không đổi và chỉ 0,00003% trọng số đổi mức sau 77 bước. Trọng số khởi tạo nằm giữa ô lượng tử hóa nên cần dịch ≥ ½ bước lưới; Adam dịch cỡ LR mỗi bước nên LR 5e-6 chỉ đi được ~5% quãng đường. Tăng lên 5e-5 mới có đổi mức (§6.2). Không phải do "loss thấp": Adam chuẩn hóa gradient, và grad_norm ~0,6 ≫ eps.
2. **KL top-10 thuần có hại:** loss giảm nhưng val PPL tệ dần (2,4645 → 2,4965). Giả thuyết (chưa kiểm chứng): student rò xác suất ra ngoài top-10 mà không bị phạt trực tiếp. Chế độ `renorm` chưa thử.
3. **Quyền 0600** của `model.safetensors` do DeepSpeed → dùng `keep_open_perms2.sh`.
4. **`pkill -f "<pattern>"` có thể giết chính shell** nếu chuỗi lệnh chứa pattern (đã xảy ra 3 lần). Dùng `pgrep -f "[p]attern"` chỉ trong lệnh không chứa lại pattern, hoặc kill theo pid.
5. **Cache Triton nguội** làm đánh giá chậm bất thường (>25 phút/150 mẫu) vì kernel fla (linear attention) compile + autotune lại; lần sau nhanh.
6. **GPU:** `CUDA driver initialization failed` và `nvidia-smi` timeout xuất hiện chập chờn (chưa rõ nguyên nhân); `gpu_idle_watchdog` tự bật stress khi GPU rảnh ≥13 phút và chiếm ~43 GB + 100% compute — dừng stress trước khi chạy job nặng.
7. **Dùng chung GPU với huấn luyện:** VRAM dự trữ của job dao động (112–126 GB); vLLM cần `--mem` nhỏ và `--max-num-seqs` nhỏ, vẫn có nguy cơ OOM chéo.
8. **Dung lượng:** mỗi checkpoint ~9,1 GB; 2 run × 21 checkpoint ≈ 380 GB, ổ chung ~93% đầy.
9. **Monitor + `tr`:** `tr '\r' '\n'` làm đệm đầu ra, Monitor không nhận được dòng; dùng `stdbuf -oL` hoặc kiểm tra log trực tiếp.
10. **Script cũ `build_calib_v7_1024.py` dùng `splitlines()`** làm lệch chỉ số dòng sau dòng 4485 (bộ calibrate 1024 gốc có khoảng 740 mẫu sai); bản đã sửa dùng `split("\n")`. Các bản PTQ `rand512` cũ còn quantize cả `lm_head` — không dùng để so sánh sạch.

## 8. Giới hạn và việc chưa kiểm chứng

- **Nhiễu của đánh giá chưa đo.** Mọi val PPL là một lần đo trên 400 mẫu; hai bộ calibrate PTQ từng lệch 0,054 PPL (một lần chạy mỗi bộ), lớn hơn khoảng cách PTQ–BF16 (0,017). Cần NLL theo từng mẫu để có khoảng tin cậy. Vì vậy chưa nên tuyên bố QAD "vượt BF16" dù số val PPL ở bước 195 thấp hơn BF16 (2,4350).
- **Fidelity n=300**, sai số chuẩn ~2 điểm; chưa kiểm định ghép cặp; chưa chạy cho checkpoint cuối.
- **Chưa kiểm tra rò rỉ gần đúng** giữa train và val (chỉ kiểm khớp chính xác lượt user đầu: 0/400).
- **Chưa so logits** giữa vLLM và PyTorch trên cùng prompt; vLLM mới được kiểm tra bằng smoke test và fidelity.
- **Chưa làm:** scale học được (LSQ), 9B / r32 / r128, phân tích theo ngôn ngữ, LR >5e-5, chế độ `renorm`.
- **Chưa xác nhận** `optimizer` giữ bản master fp32 dưới DeepSpeed bf16 (chỉ thấy `bf16: auto`).

## 9. Dọn dẹp
**Đã thực hiện** (2026-10-07, theo cho phép của chủ thí nghiệm): xóa 35 checkpoint thừa, giải phóng ~340 GB (ổ chung từ 433 GB lên 753 GB trống). **Còn giữ:** full-KL `checkpoint-15/105/195/255/307`; top-10 `checkpoint-195/307`; `outputs/qad-4b-r64-w4a16/checkpoint-77` (run LR 5e-6 đã bỏ, ~9 GB, dùng làm bằng chứng cho §7.1); `artifacts/student_init_4b_r64` và `artifacts/export/*` (bản W4A16 đã xuất, ~4,4 GB mỗi cái). Đã xóa thêm `artifacts/vllm/fullkl_cp195` (sinh lại được bằng `60_to_vllm_layout.py`). Các kết quả đánh giá đã lưu trong `logs/eval-*.json` nên không cần giữ lại checkpoint đã xóa để tra số.

## 10. Bước tiếp theo đề xuất
0. Làm rõ vì sao PPL xấu đi từ bước 195 sang 307 (đo thêm 105/255 đang chạy; nhiễu hay overfit?).
1. Đo khoảng tin cậy PPL (NLL theo mẫu) và chạy fidelity cho bước 307.
2. Thử scale học được (LSQ, khởi tạo từ scale PTQ) và LR 1e-4.
3. Thử top-10 `renorm` hoặc KL toàn vocab + trọng số tăng cho token khó.
4. Nhân rộng sang 9B / r32 / r128 nếu 4B r64 xác nhận có ích.
