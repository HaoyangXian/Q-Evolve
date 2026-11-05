#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
LOG=$RUN/logs
FILTER_SUM=$RUN/policy_filter_len16384_summary.json
DATA=$RUN/policy_data_len16384.jsonl
QV=$RUN/policy_adv_len16384.jsonl
mkdir -p "$LOG"
exec > >(tee -a "$RUN/policy_queue_16384.log") 2>&1
printf '[Queue] %s waiting for strict filter output\n' "$(date -Is)"
while [[ ! -s "$FILTER_SUM" || ! -s "$DATA" || ! -s "$QV" ]]; do sleep 30; done
printf '[Queue] %s strict filter ready\n' "$(date -Is)"
cat "$FILTER_SUM"
printf '[Queue] waiting for GPUs 0,1,2,3 to be idle (<512 MiB each)\n'
while true; do
  ok=1
  while IFS=, read -r idx mem; do
    idx=${idx//[[:space:]]/}; mem=${mem//[[:space:]]/}
    case "$idx" in 0|1|2|3) (( mem > 512 )) && ok=0 ;; esac
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if (( ok == 1 )); then break; fi
  sleep 60
done
printf '[Queue] %s GPUs 0-3 idle; starting policy\n' "$(date -Is)"
rm -f "$RUN/DONE" "$RUN/EVAL_DONE"
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29804 "$ROOT/QEvolve/train_policy.py" \
  --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$QV" \
  --policy_model "$MODEL" --lr 1e-5 --max_length 16384 --epochs 2 \
  --output_dir "$RUN/policy" --gradient_accumulation_steps 1 \
  --max_tokens_per_device 8192 --max_examples_per_device 2 --dataloader_num_workers 0 \
  --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 \
  --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora \
  2>&1 | tee "$LOG/policy_16384.log"
printf 'OLYMPIAD_QWEN3_8B_TRAIN_DONE\n' > "$RUN/DONE"
printf '[Queue] %s policy DONE; starting Base/SFT/policy evaluation\n' "$(date -Is)"
"$ROOT/bin/qa/run_olympiad_eval_small_iterative1024_4gpu.sh" 2>&1 | tee "$LOG/eval_small_16384.log"
: > "$RUN/EVAL_DONE"
printf '[Queue] %s evaluation DONE\n' "$(date -Is)"
