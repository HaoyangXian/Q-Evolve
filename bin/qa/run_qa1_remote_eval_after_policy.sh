#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
TRAIN_PID=${1:?training wrapper PID is required}
LOG=${ROOT}/runs/qa1_fixed_remote_logs/eval_wait.log
ADAPTER=${ROOT}/runs/qa1_fixed_policy_4096/adapter_model.safetensors

echo "[$(date '+%F %T')] waiting for policy PID ${TRAIN_PID} and adapter" >> "${LOG}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 30
done

if [ ! -f "${ADAPTER}" ]; then
  echo "[$(date '+%F %T')] policy exited without adapter; evaluation not started" >> "${LOG}"
  exit 1
fi

echo "[$(date '+%F %T')] adapter ready; evaluation start" >> "${LOG}"
exec "${ROOT}/bin/qa/run_qa1_remote_eval_4gpu.sh" >> "${LOG}" 2>&1
