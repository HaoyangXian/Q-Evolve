#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-4B
INPUT=$ROOT/data/qa/gsm8k_eval150.jsonl
OUTROOT=$ROOT/data/qa/current_close
mkdir -p "$OUTROOT"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
GPUS=(1 2 3 5)
pids=(); off=0
for part in 0 1 2 3; do
 lim=38; [ "$part" = 3 ] && lim=36
 CUDA_VISIBLE_DEVICES=${GPUS[$part]} $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUTROOT/policy_part${part}.jsonl" --model "$MODEL" --adapter "$ROOT/runs/formal200_qwen3_policy_valid" --offset "$off" --limit "$lim" --samples 1 --max_new_tokens 2048 --temperature 0 --termination_mode strict >"$OUTROOT/policy_part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim))
done
for p in "${pids[@]}"; do wait "$p"; done
$PY - <<'PY'
import json,pathlib
p=pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/current_close'); out=p/'policy_gsm8k_eval150_strict.jsonl'
with out.open('w') as f:
 for i in range(4): f.write((p/f'policy_part{i}.jsonl').read_text())
rs=[json.loads(x) for x in out.open() if x.strip()]
print('policy',len(rs),sum(r.get('meta',{}).get('success',False) for r in rs),sum(r.get('meta',{}).get('finished',False) for r in rs))
PY
