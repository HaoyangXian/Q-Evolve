#!/usr/bin/env bash
# 重跑 shard0 critic 打分（卡4 被 AgenTracer 占用，改用卡5）
cd /data3/xhy/codes/work/paper_repos/Q-Evolve
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/data3/xhy/codes/work/paper_repos/Q-Evolve:/data3/xhy/codes/work/paper_repos/Q-Evolve/QEvolve
export TOKENIZERS_PARALLELISM=false
nohup /data3/xhy/miniconda3/envs/pettingllms/bin/python -u drift_exp/score_prefixes.py \
  --traj drift_exp/branch_pilot/traj_shard0.jsonl \
  --prefixes 16 32 \
  --out drift_exp/branch_pilot/prefix_scores_shard0.jsonl \
  > drift_exp/branch_pilot/score_shard0_retry.log 2>&1 < /dev/null &
echo "shard0 重打分启动 PID=$!"
