#!/bin/bash
# 重置能力矩阵实验：杀掉被截断污染/不完整的旧进程，重命名旧文件，用新参数重跑
set -e
cd /data3/xhy/codes/work/paper_repos/Q-Evolve
RES=drift_exp/capability_probe/results
TS=$(date +%F_%H%M%S)

echo "=== 1. 杀旧调度器 + 旧 qa_rollout 进程 ==="
kill 1521175 2>/dev/null || true
sleep 1
kill 1541757 1541758 1591009 1706514 2>/dev/null || true
sleep 2
# 兜底：pkill 精确匹配（仅 xhy 用户 + probe_input 相关）
pkill -f 'qa_rollout.py --input .*probe_input.jsonl' 2>/dev/null || true
sleep 1
echo "残留进程数: $(ps aux | grep -E 'experiments.py|qa_rollout.py' | grep -v grep | grep xhy | wc -l)"

echo
echo "=== 2. 重命名污染/不完整文件（保留已完整的 budget_8）==="
for f in rollout_base.jsonl rollout_sft.jsonl rollout_policy.jsonl rollout_sft_b16.jsonl rollout_sft_b32.jsonl; do
    if [ -f "$RES/$f" ]; then
        sz=$(stat -c%s "$RES/$f")
        mv "$RES/$f" "$RES/$f.old_$TS"
        echo "重命名 $f ($sz bytes) -> $f.old_$TS"
    fi
done

echo
echo "=== 3. 保留的 budget_8（已完整）==="
wc -l "$RES/rollout_sft_b8.jsonl" 2>/dev/null || echo "budget_8 不存在"

echo
echo "=== 4. 重启调度器 ==="
nohup /data3/xhy/miniconda3/envs/pettingllms/bin/python -u drift_exp/capability_probe/experiments.py > drift_exp/capability_probe/experiments.log 2>&1 < /dev/null &
echo "调度器重启 PID=$!"
