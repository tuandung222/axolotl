#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dungvpt/low_bit_research/axolotl_tuandunghcmut"
LOG_PATH="${ROOT_DIR}/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.train.log"
PID_PATH="${ROOT_DIR}/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.train.pid"
CONFIG_PATH="${ROOT_DIR}/examples/LiquidAI/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.yaml"

mkdir -p "${ROOT_DIR}/runs"
rm -f "${PID_PATH}" "${LOG_PATH}"
cd "${ROOT_DIR}"

export HF_HOME=/home/dungvpt/hf_home
export HF_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32

setsid /home/dungvpt/anaconda3/envs/axolotl_cu13/bin/torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  -m axolotl.cli.train \
  "${CONFIG_PATH}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo $! > "${PID_PATH}"
echo "pid=$(cat "${PID_PATH}")"
echo "log=${LOG_PATH}"
