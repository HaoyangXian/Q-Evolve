#!/usr/bin/env bash
# 启动实验队列调度器（后台）
cd /data3/xhy/codes/work/paper_repos/Q-Evolve
nohup /data3/xhy/miniconda3/envs/pettingllms/bin/python -u drift_exp/capability_probe/experiments.py \
  > drift_exp/capability_probe/experiments.log 2>&1 < /dev/null &
echo "队列调度器启动 PID=$!"
