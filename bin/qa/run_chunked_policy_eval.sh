#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
LOG=$RUN/logs
ADV_FULL=$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/olympiad_chunked/adv_token_pred-gamma0.95-len32768-olympiad_chunked.jsonl
DATA=$RUN/policy_chunked8192_data.jsonl
QV=$RUN/policy_chunked8192_adv.jsonl
SUM=$RUN/policy_chunked8192_summary.json
OUT=$ROOT/data/qa/olympiad_eval_small_chunked8192
mkdir -p "$LOG" "$OUT"
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec > >(tee -a "$RUN/chunked_policy_queue.log") 2>&1
printf '[Chunked] waiting for full advantage %s\n' "$(date -Is)"
while true; do
  n=0; [[ -f "$ADV_FULL" ]] && n=$(wc -l < "$ADV_FULL")
  if (( n >= 215 )) && ! pgrep -f "[i]nf_critic_chunked_fast.py" >/dev/null; then break; fi
  sleep 60
done
printf '[Chunked] full advantage ready %s\n' "$(date -Is)"
"$PY" /tmp/build_chunked_policy_data.py
"$PY" - "$SUM" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x['dropped_source_rows']:
    raise SystemExit(f"chunk builder dropped {x['dropped_source_rows']} source rows")
print('CHUNK_SUMMARY',json.dumps({k:x[k] for k in ('source_trajectories','chunk_rows','dropped_source_rows','mean_overlap_tokens')},ensure_ascii=False))
PY
printf '[Chunked] waiting for GPUs 2,5,6,7 to be idle\n'
while true; do
  ok=1
  while IFS=, read -r idx mem; do
    idx=${idx//[[:space:]]/}; mem=${mem//[[:space:]]/}
    case "$idx" in 2|5|6|7) if (( mem > 512 )); then ok=0; fi ;; esac
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  (( ok == 1 )) && break
  sleep 60
done
printf '[Chunked] GPUs ready %s; starting chunked policy\n' "$(date -Is)"
rm -f "$RUN/CHUNKED_DONE" "$RUN/CHUNKED_EVAL_DONE"
CUDA_VISIBLE_DEVICES=2,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29806 "$ROOT/QEvolve/train_policy_chunked.py" \
  --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$QV" \
  --policy_model "$MODEL" --lr 1e-5 --max_length 8192 --epochs 2 \
  --output_dir "$RUN/policy_chunked8192" --gradient_accumulation_steps 1 \
  --max_tokens_per_device 4096 --max_examples_per_device 1 --dataloader_num_workers 0 \
  --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 \
  --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora --precomputed_advantage \
  2>&1 | tee "$LOG/policy_chunked8192.log"
printf 'OLYMPIAD_QWEN3_8B_CHUNKED_POLICY_DONE\n' > "$RUN/CHUNKED_DONE"

run_model(){
  local label=$1
  local adapter=$2
  rm -f "$OUT/${label}_part"*.jsonl "$OUT/${label}_part"*.log
  local pids=() off=0
  local GPUS=(2 5 6 7)
  for part in 0 1 2 3; do
    local lim=8
    CUDA_VISIBLE_DEVICES=${GPUS[$part]} "$PY" -u "$ROOT/eval_agent/qa_rollout.py" \
      --input "$ROOT/data/qa/olympiadbench.jsonl" --output "$OUT/${label}_part${part}.jsonl" \
      --model "$MODEL" ${adapter:+--adapter "$adapter"} --offset "$off" --limit "$lim" --samples 1 \
      --max_new_tokens 8192 --step_max_new_tokens 1024 --max_steps 256 --context_max_tokens 32768 \
      --temperature 0 --termination_mode strict >"$OUT/${label}_part${part}.log" 2>&1 &
    pids+=("$!"); off=$((off+lim))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$OUT/${label}_part"*.jsonl > "$OUT/${label}.jsonl"
}
run_model base ""
run_model sft "$RUN/sft_success"
run_model policy "$RUN/policy_chunked8192"
"$PY" - "$OUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); report={}
for label in ('base','sft','policy'):
 rows=[json.loads(x) for x in (p/f'{label}.jsonl').open() if x.strip()]
 report[label]={'n':len(rows),'correct':sum(bool(r.get('meta',{}).get('success',False)) for r in rows),'accuracy':sum(bool(r.get('meta',{}).get('success',False)) for r in rows)/len(rows) if rows else 0.0,'finished':sum(bool(r.get('meta',{}).get('finished',False)) for r in rows)}
print(json.dumps(report,ensure_ascii=False,indent=2)); (p/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
PY
: > "$RUN/CHUNKED_EVAL_DONE"
printf '[Chunked] evaluation done %s\n' "$(date -Is)"
