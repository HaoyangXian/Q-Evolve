#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-4B
INPUT=$ROOT/data/qa/gsm8k_eval150.jsonl
OUTROOT=$ROOT/data/qa/current_close
mkdir -p "$OUTROOT"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
run_one(){ local label=$1; local adapter=$2; local part=$3; local off=$4; local lim=$5; local gpu=$6; CUDA_VISIBLE_DEVICES=$gpu $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUTROOT/${label}_part${part}.jsonl" --model "$MODEL" ${adapter:+--adapter "$adapter"} --offset "$off" --limit "$lim" --samples 1 --max_new_tokens 2048 --temperature 0 --termination_mode strict; }
for label in base policy; do
  adapter=""; [ "$label" = policy ] && adapter="$ROOT/runs/formal200_qwen3_policy_valid"
  pids=(); off=0
  for part in 0 1 2 3; do
    lim=38; [ "$part" = 3 ] && lim=36
    run_one "$label" "$adapter" "$part" "$off" "$lim" "$((part+1))" >"$OUTROOT/${label}_part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim))
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
$PY - <<'PY'
import json,pathlib
p=pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/current_close')
for lab in ('base','policy'):
 out=p/f'{lab}_gsm8k_eval150_strict.jsonl'
 with out.open('w') as f:
  for i in range(4):
   f.write((p/f'{lab}_part{i}.jsonl').read_text())
 rs=[json.loads(x) for x in out.open() if x.strip()]
 print(lab,len(rs),sum(r['meta']['success'] for r in rs),sum(r['meta']['finished'] for r in rs))
PY
