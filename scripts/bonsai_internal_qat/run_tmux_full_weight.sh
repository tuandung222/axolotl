#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dungvpt/low_bit_research/axolotl_tuandunghcmut"
EXPERIMENT="$ROOT/artifacts/bonsai_full_qat_non_reasoning"
CONFIG="examples/bs_bonsai_full_qat_non_reasoning.yaml"
SESSION="bonsai_full_qat_4xh200"

mkdir -p "$EXPERIMENT/logs" "$ROOT/tmp"
cd "$ROOT"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" bash
tmux send-keys -t "$SESSION" "set -euo pipefail" C-m
tmux send-keys -t "$SESSION" "source /home/dungvpt/anaconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION" "conda activate axolotl_cu13" C-m
tmux send-keys -t "$SESSION" "export TMPDIR=$ROOT/tmp" C-m
tmux send-keys -t "$SESSION" "export HF_HOME=/home/dungvpt/hf_home" C-m
tmux send-keys -t "$SESSION" "unset HF_TOKEN" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3" C-m
tmux send-keys -t "$SESSION" "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t "$SESSION" "export TOKENIZERS_PARALLELISM=false" C-m
tmux send-keys -t "$SESSION" "cd $ROOT" C-m
# Reuses artifacts/bonsai_internal_qat_non_reasoning_v2/data/train.jsonl (same
# corpus as the v2 LoRA run) -- no prepare_dataset step needed here.
tmux send-keys -t "$SESSION" "axolotl preprocess $CONFIG --debug --debug-num-examples 2 2>&1 | tee $EXPERIMENT/logs/tokenize.log" C-m

echo "tmux session started: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "NOTE: this only queues the preprocess sanity check. Training is started"
echo "separately after preprocess is confirmed to succeed."
