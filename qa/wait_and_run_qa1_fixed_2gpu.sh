#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
LOG=${ROOT}/runs/qa1_fixed_2gpu_wait.log
mkdir -p "${ROOT}/runs"

echo "[$(date '+%F %T')] queued for CUDA devices 0,1" >> "${LOG}"
while true; do
  if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' '
        $1 ~ /^[[:space:]]*0[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); g0=$2}
        $1 ~ /^[[:space:]]*1[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); g1=$2}
        END {exit !(g0 != "" && g1 != "" && g0+0 < 2000 && g1+0 < 2000)}'; then
    echo "[$(date '+%F %T')] CUDA 0,1 available; starting 2-GPU run" >> "${LOG}"
    export CUDA_VISIBLE_DEVICES=0,1
    exec "${ROOT}/bin/qa/run_qa1_fixed_full.sh" >> "${LOG}" 2>&1
  fi
  echo "[$(date '+%F %T')] waiting for both CUDA 0,1 (threshold 2 GiB each)" >> "${LOG}"
  sleep 30
done
