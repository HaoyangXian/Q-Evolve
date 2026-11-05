#!/usr/bin/env bash
# 等 Solver-A critic 打分(pid 648107)退出后，用 gpu_reserve.py 占住卡3
set -uo pipefail
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
TARGET_PID=${1:-648107}
RESERVE_GPU=3
FRACTION=${FRACTION:-0.85}

log() { echo "$(date '+%F %T') [reserve-critic] $*"; }

# 等 critic 进程退出
log "等待 critic 打分进程 pid=$TARGET_PID 退出..."
while kill -0 "$TARGET_PID" 2>/dev/null; do sleep 60; done
log "critic 进程已退出，开始占住卡$RESERVE_GPU"

# 占卡
nohup "$PY" /data3/xhy/codes/gpu_reserve.py --gpus "$RESERVE_GPU" --fraction "$FRACTION" \
  > /data3/xhy/codes/gpu_reserve_gpu${RESERVE_GPU}.log 2>&1 < /dev/null &
log "占卡已启动(卡$RESERVE_GPU, fraction=$FRACTION)"
