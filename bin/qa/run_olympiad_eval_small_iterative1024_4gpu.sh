#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
INPUT=$ROOT/data/qa/olympiadbench.jsonl
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
OUT=$ROOT/data/qa/olympiad_eval_small_iterative1024
mkdir -p "$OUT"; export PYTHONPATH=$ROOT:$ROOT/QEvolve
run_model(){ local label=$1; local adapter=$2; rm -f "$OUT/${label}_part"*.jsonl; pids=(); off=0; for part in 0 1 2 3; do lim=8; CUDA_VISIBLE_DEVICES=$part $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUT/${label}_part${part}.jsonl" --model "$MODEL" ${adapter:+--adapter "$adapter"} --offset "$off" --limit "$lim" --samples 1 --max_new_tokens 8192 --step_max_new_tokens 1024 --max_steps 256 --context_max_tokens 32768 --temperature 0 --termination_mode strict >"$OUT/${label}_part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim)); done; for p in "${pids[@]}"; do wait "$p"; done; cat "$OUT/${label}_part"*.jsonl > "$OUT/${label}.jsonl"; }
run_model base 
run_model sft "$RUN/sft_success"
run_model policy "$RUN/policy"
$PY - "$OUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); report={}
for label in ('base','sft','policy'):
 rows=[json.loads(x) for x in (p/f'{label}.jsonl').open() if x.strip()]
 report[label]={'n':len(rows),'correct':sum(r.get('meta',{}).get('success',False) for r in rows),'finished':sum(r.get('meta',{}).get('finished',False) for r in rows)}
print(json.dumps(report,ensure_ascii=False,indent=2)); (p/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
PY
