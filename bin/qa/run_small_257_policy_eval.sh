#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
DATA=$RUN/policy_data_len16384.jsonl
QV=$RUN/policy_adv_len16384.jsonl
LOG=$RUN/logs
OUT=$ROOT/data/qa/olympiad_eval_small_iterative1024
mkdir -p "$LOG" "$OUT"
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
rm -f "$RUN/DONE" "$RUN/EVAL_DONE"
CUDA_VISIBLE_DEVICES=2,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29805 "$ROOT/QEvolve/train_policy.py" \
  --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$QV" \
  --policy_model "$MODEL" --lr 1e-5 --max_length 16384 --epochs 2 \
  --output_dir "$RUN/policy" --gradient_accumulation_steps 1 \
  --max_tokens_per_device 8192 --max_examples_per_device 2 --dataloader_num_workers 0 \
  --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 \
  --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora \
  2>&1 | tee "$LOG/policy_16384_gpus2567.log"
printf 'OLYMPIAD_QWEN3_8B_TRAIN_DONE\n' > "$RUN/DONE"

run_model(){
  local label=$1
  local adapter=$2
  rm -f "$OUT/${label}_part"*.jsonl "$OUT/${label}_part"*.log
  local pids=() off=0
  local GPUS=(2 5 6 7)
  for part in 0 1 2 3; do
    local lim=8
    CUDA_VISIBLE_DEVICES=${GPUS[$part]} "$PY" -u "$ROOT/eval_agent/qa_rollout.py" \
      --input "$ROOT/data/qa/olympiadbench.jsonl" \
      --output "$OUT/${label}_part${part}.jsonl" \
      --model "$MODEL" ${adapter:+--adapter "$adapter"} \
      --offset "$off" --limit "$lim" --samples 1 --max_new_tokens 8192 \
      --step_max_new_tokens 1024 --max_steps 256 --context_max_tokens 32768 \
      --temperature 0 --termination_mode strict \
      >"$OUT/${label}_part${part}.log" 2>&1 &
    pids+=("$!")
    off=$((off+lim))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$OUT/${label}_part"*.jsonl > "$OUT/${label}.jsonl"
}
run_model base ""
run_model sft "$RUN/sft_success"
run_model policy "$RUN/policy"
"$PY" - "$OUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); report={}
for label in ('base','sft','policy'):
 rows=[json.loads(x) for x in (p/f'{label}.jsonl').open() if x.strip()]
 report[label]={
  'n':len(rows),
  'correct':sum(bool(r.get('meta',{}).get('success',False)) for r in rows),
  'accuracy':sum(bool(r.get('meta',{}).get('success',False)) for r in rows)/len(rows) if rows else 0.0,
  'finished':sum(bool(r.get('meta',{}).get('finished',False)) for r in rows),
 }
print(json.dumps(report,ensure_ascii=False,indent=2)); (p/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
PY
: > "$RUN/EVAL_DONE"
