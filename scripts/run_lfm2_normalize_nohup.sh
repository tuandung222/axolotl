#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dungvpt/low_bit_research/axolotl_tuandunghcmut"
LOG_PATH="${ROOT_DIR}/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.normalize.log"
PID_PATH="${ROOT_DIR}/runs/lfm2.5-8b-a1b-32k-4xh200-non-reasoning-mix.normalize.pid"

mkdir -p "${ROOT_DIR}/runs"
rm -f "${PID_PATH}" "${LOG_PATH}"
cd "${ROOT_DIR}"

setsid /home/dungvpt/anaconda3/envs/axolotl_cu13/bin/python \
  scripts/prepare_lfm2_5_non_reasoning_mix.py \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo $! > "${PID_PATH}"
echo "pid=$(cat "${PID_PATH}")"
echo "log=${LOG_PATH}"
