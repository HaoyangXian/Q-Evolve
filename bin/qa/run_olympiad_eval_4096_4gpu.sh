#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-4B
INPUT=$ROOT/data/qa/olympiadbench.jsonl
OUT=$ROOT/data/qa/olympiad_eval
mkdir -p "$OUT"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
while [ ! -f "$ROOT/runs/olympiad_qevolve_4096_4096/OLYMPIAD_TRAIN_DONE" ]; do sleep 60; done
run_model(){ local label=$1; local adapter=$2; rm -f "$OUT/${label}_part"*.jsonl; pids=(); off=0; GPUS=(1 2 3 5)
for part in 0 1 2 3; do lim=169; [ "$part" = 3 ] && lim=167; CUDA_VISIBLE_DEVICES=${GPUS[$part]} $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUT/${label}_part${part}.jsonl" --model "$MODEL" ${adapter:+--adapter "$adapter"} --offset "$off" --limit "$lim" --samples 1 --max_new_tokens 2048 --temperature 0 --termination_mode strict >"$OUT/${label}_part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim)); done; for p in "${pids[@]}"; do wait "$p"; done; }
run_model base ""
run_model sft "$ROOT/runs/olympiad_qevolve_4096/sft_success"
run_model policy "$ROOT/runs/olympiad_qevolve_4096/policy"
$PY - <<'PY'
import json,pathlib
p=pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/olympiad_eval')
for lab in ('base','sft','policy'):
 out=p/f'{lab}.jsonl'
 with out.open('w') as f:
  for i in range(4): f.write((p/f'{lab}_part{i}.jsonl').read_text())
 rs=[json.loads(x) for x in out.open() if x.strip()]
 print(lab,len(rs),sum(r.get('meta',{}).get('success',False) for r in rs),sum(r.get('meta',{}).get('finished',False) for r in rs))
PY
