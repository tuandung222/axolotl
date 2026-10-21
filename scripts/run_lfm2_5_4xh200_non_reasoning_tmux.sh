#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dungvpt/low_bit_research/axolotl_tuandunghcmut"
SESSION="lfm2_5_4xh200"
CONFIG="examples/LiquidAI/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.yaml"
PREP_LOG="$ROOT/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.prepare.log"
TRAIN_LOG="$ROOT/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.train.log"

mkdir -p "$ROOT/runs"
cd "$ROOT"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" bash
tmux send-keys -t "$SESSION" "set -euo pipefail" C-m
tmux send-keys -t "$SESSION" "source /home/dungvpt/anaconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION" "conda activate axolotl_cu13" C-m
tmux send-keys -t "$SESSION" "export HF_HOME=/home/dungvpt/hf_home" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3" C-m
tmux send-keys -t "$SESSION" "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t "$SESSION" "export TOKENIZERS_PARALLELISM=false" C-m
tmux send-keys -t "$SESSION" "cd $ROOT" C-m
tmux send-keys -t "$SESSION" "python scripts/prepare_lfm2_5_non_reasoning_mix.py 2>&1 | tee $PREP_LOG" C-m
tmux send-keys -t "$SESSION" "axolotl preprocess $CONFIG --debug --debug-num-examples 2 2>&1 | tee -a $PREP_LOG" C-m
tmux send-keys -t "$SESSION" "axolotl train $CONFIG --launcher torchrun -- --nnodes=1 --nproc_per_node=4 2>&1 | tee $TRAIN_LOG" C-m

echo "tmux session started: $SESSION"
echo "attach: tmux attach -t $SESSION"
