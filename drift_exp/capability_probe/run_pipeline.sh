#!/usr/bin/env bash
# 多维能力诊断 Pipeline：Probe 构造 -> 能力矩阵 -> 信息缺口 -> Active vs Static
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
OUT=$ROOT/drift_exp/capability_probe
export PYTHONPATH=$ROOT:$ROOT/QEvolve

echo "=== [1/6] 构造 Probe Pool ==="
$PY $OUT/probe_bank.py --outdir $OUT/probe_bank

echo "=== [2/6] 准备评测 input ==="
$PY $OUT/eval_capability.py --prepare --probe $OUT/probe_bank/probe_pool.jsonl

echo "=== [3/6] 3 档 checkpoint 并行跑能力矩阵（卡0/1/2）==="
$PY $OUT/eval_capability.py --run --checkpoint base   --gpu 0 &
$PY $OUT/eval_capability.py --run --checkpoint sft    --gpu 1 &
$PY $OUT/eval_capability.py --run --checkpoint policy --gpu 2 &
wait

echo "=== [4/6] 聚合能力矩阵 ==="
$PY $OUT/eval_capability.py --aggregate

echo "=== [5/6] 信息缺口验证 ==="
$PY $OUT/info_gap.py --matrix $OUT/results/capability_matrix.json

echo "=== [6/6] Active vs Static 模拟 ==="
$PY $OUT/adaptive_sim.py --pool $OUT/probe_bank/probe_pool.jsonl \
  --rollouts $OUT/results/rollout_base.jsonl $OUT/results/rollout_sft.jsonl $OUT/results/rollout_policy.jsonl

echo "=== 完成 ==="
