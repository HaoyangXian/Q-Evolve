#!/usr/bin/env bash
# 每 60 秒记录 critic(卡3) 和 Solver-B(卡0) 进度到 monitor.log
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
while :; do
  C=$(tail -1 $ROOT/drift_exp/critic_A_ms512.log 2>/dev/null | tr "\r" "\n" | tail -1)
  B=$(wc -l < $ROOT/drift_exp/solver_B_ms64.jsonl 2>/dev/null)
  A=$(wc -l < $ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl 2>/dev/null)
  echo "$(date +%F %T) | Solver-B=$B/200 | critic_out=$A | $C" >> $ROOT/drift_exp/monitor.log
  # critic 和 reserve 都结束后退出
  if ! kill -0 660136 2>/dev/null; then
    echo "$(date +%F %T) | critic 进程退出" >> $ROOT/drift_exp/monitor.log
    break
  fi
  sleep 60
done
