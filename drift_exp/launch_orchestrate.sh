#!/usr/bin/env bash
cd /data3/xhy/codes/work/paper_repos/Q-Evolve
nohup /data3/xhy/miniconda3/envs/pettingllms/bin/python -u drift_exp/orchestrate_branching.py \
  > drift_exp/branch_pilot/orchestrate.log 2>&1 < /dev/null &
echo "编排启动 PID=$!"
