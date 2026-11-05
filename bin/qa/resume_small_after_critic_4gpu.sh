#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
DATA=$ROOT/data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_small/clean.jsonl
LOG=$RUN/logs
export PYTHONPATH=$ROOT:$ROOT/QEvolve
cd $ROOT
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u $ROOT/QEvolve/inf_critic.py --data_path "$DATA" --task qa --no_aux_reward --advantage_base_model "$MODEL" --advantage_model_path "$RUN/critic/olympiad/Qwen3-8B" --max_length 32768 --exp_name olympiad --strict 2>&1 | tee "$LOG/advantage_4gpu.log"
ADV=$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/olympiad/adv_token_pred-gamma0.95-len32768-olympiad.jsonl
$PY $ROOT/bin/qa/filter_advantage.py --input "$ADV" --output "$RUN/adv_positive.jsonl" | tee "$LOG/filter_advantage_4gpu.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29804 $ROOT/QEvolve/train_policy.py --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$RUN/adv_positive.jsonl" --policy_model "$MODEL" --lr 1e-5 --max_length 32768 --epochs 2 --output_dir "$RUN/policy" --gradient_accumulation_steps 1 --max_tokens_per_device 8192 --max_examples_per_device 2 --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora 2>&1 | tee "$LOG/policy_4gpu.log"
echo OLYMPIAD_QWEN3_8B_TRAIN_DONE > "$RUN/DONE"
$ROOT/bin/qa/run_olympiad_eval_small_iterative1024_4gpu.sh
: > "$RUN/EVAL_DONE"
