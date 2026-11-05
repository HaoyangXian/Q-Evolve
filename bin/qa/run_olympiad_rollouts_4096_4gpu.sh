#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-4B
INPUT=$ROOT/data/qa/olympiadbench.jsonl
OUT=$ROOT/data/qa/olympiad_rollouts_8_4096
mkdir -p "$OUT"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
wait_gpus(){
 while true; do
  rows=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
  ok=1
  for g in 1 2 3 5; do u=$(echo "$rows" | awk -F',' -v g="$g" '$1+0==g{gsub(/ /,"",$2);print $2}'); [ -n "$u" ] && [ "$u" -lt 1024 ] || ok=0; done
  [ "$ok" -eq 1 ] && break
  sleep 60
 done
}
wait_gpus
pids=(); off=0
GPUS=(1 2 3 5)
for part in 0 1 2 3; do
 lim=169; [ "$part" = 3 ] && lim=167
 CUDA_VISIBLE_DEVICES=${GPUS[$part]} $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUT/part${part}.jsonl" --model "$MODEL" --offset "$off" --limit "$lim" --samples 8 --max_new_tokens 4096 --temperature 0.7 --termination_mode strict >"$OUT/part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim))
done
for p in "${pids[@]}"; do wait "$p"; done
$PY - <<'PY'
from pathlib import Path
p=Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/olympiad_rollouts_8_4096'); out=p/'all.jsonl'
with out.open('w') as f:
 for i in range(4): f.write((p/f'part{i}.jsonl').read_text())
print('merged',sum(1 for _ in out.open()),out)
PY
