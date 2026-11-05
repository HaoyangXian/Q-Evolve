#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RAW_DATA=$ROOT/data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_small/all.jsonl
DATA=$ROOT/data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_small/clean.jsonl
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
LOG=$RUN/logs
mkdir -p "$LOG"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$ROOT"
while [ ! -s "$RAW_DATA" ]; do sleep 60; done
$PY $ROOT/bin/qa/filter_rollouts.py --input "$RAW_DATA" --output "$DATA" | tee "$LOG/filter_rollouts.log"
$PY - "$DATA" "$RUN" <<'PY'
import json,sys,pathlib
src=pathlib.Path(sys.argv[1]); run=pathlib.Path(sys.argv[2]); rows=[json.loads(x) for x in src.open() if x.strip()]
for name,sel in [('success',lambda r:r.get('meta',{}).get('success') is True),('all',lambda r:True)]:
 out=run/f'{name}.json'; out.parent.mkdir(parents=True,exist_ok=True); json.dump([r for r in rows if sel(r)],out.open('w'),ensure_ascii=False)
print('rollouts',len(rows),'success',sum(r.get('meta',{}).get('success') is True for r in rows))
PY
$PY -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29801 $ROOT/eval_agent/train_sft.py \
 --model_name_or_path "$MODEL" --data_path "$RUN/success.json" --bf16 True --output_dir "$RUN/sft_success" \
 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 2 \
 --save_strategy no --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
 --logging_steps 10 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer \
 --tf32 True --model_max_length 16384 --gradient_checkpointing True --lazy_preprocess False 2>&1 | tee "$LOG/sft_success.log"
$PY -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29802 $ROOT/QEvolve/train_critic.py \
 --data_path "$DATA" --model_name "$MODEL" --task qa --no_aux_reward --max_length 16384 \
 --max_tokens_per_device 8192 --batch_size 1 --epochs 5 --lr 1e-5 --output_dir "$RUN/critic" \
 --cache_dir "$RUN/cache" --exp_name olympiad --K 4 --gradient_accumulation_steps 1 2>&1 | tee "$LOG/critic.log"
CUDA_VISIBLE_DEVICES=0 $PY -u $ROOT/QEvolve/inf_critic.py --data_path "$DATA" --task qa --no_aux_reward \
 --advantage_base_model "$MODEL" --advantage_model_path "$RUN/critic/olympiad/Qwen3-8B" --max_length 16384 \
 --exp_name olympiad --strict 2>&1 | tee "$LOG/advantage.log"
ADV=$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/olympiad/adv_token_pred-gamma0.95-len16384-olympiad.jsonl
$PY $ROOT/bin/qa/filter_advantage.py --input "$ADV" --output "$RUN/adv_positive.jsonl" | tee "$LOG/filter_advantage.log"
$PY -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29803 $ROOT/QEvolve/train_policy.py \
 --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$RUN/adv_positive.jsonl" \
 --policy_model "$MODEL" --lr 1e-5 --max_length 16384 --epochs 2 --output_dir "$RUN/policy" \
 --gradient_accumulation_steps 1 --max_tokens_per_device 8192 --max_examples_per_device 2 \
 --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 \
 --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora 2>&1 | tee "$LOG/policy.log"
echo OLYMPIAD_QWEN3_8B_TRAIN_DONE > "$RUN/DONE"
