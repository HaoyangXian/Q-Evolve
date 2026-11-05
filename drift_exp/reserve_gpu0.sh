#!/usr/bin/env bash
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
nohup $PY /data3/xhy/codes/gpu_reserve.py --gpus 0 --fraction 0.85 \
  > /data3/xhy/codes/gpu_reserve_gpu0.log 2>&1 < /dev/null &
echo "卡0 reserve 已启动 PID=$!"
sleep 5
echo "=== 当前所有 reserve 进程 ==="
ps aux | grep gpu_reserve | grep -v grep | grep xhy
