#!/usr/bin/env bash
set -u

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
LOG=${ROOT}/runs/paper_repro/queue.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "[$(date '+%F %T')] queued; waiting for 8 free GPUs"
while true; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    rows=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    count=$(printf '%s\n' "$rows" | awk 'NF==3 {n++} END {print n+0}')
    busy=$(printf '%s\n' "$rows" | awk 'NF==3 && ($2 > 1024 || $3 <= 0) {n++} END {print n+0}')
    if [[ "$count" -ge 8 && "$busy" -eq 0 ]]; then
      echo "[$(date '+%F %T')] 8 GPUs available"
      break
    fi
  fi
  sleep 60
done

cd "$ROOT"
set +e
bash bin/run_paper_repro_local.sh
rc=$?
echo "[$(date '+%F %T')] reproduction exit=$rc; starting requested reservation"
/data/xhy/miniconda3/envs/marti/bin/python /data/xhy/codes/gpu_reserve.py \
  --gpus 0,1,2,3,4,5,6,7 --fraction 0.9
exit "$rc"
