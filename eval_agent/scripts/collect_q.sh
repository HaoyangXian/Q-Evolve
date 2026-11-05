task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

python eval_agent/construct_q_data.py --task $task --data_path data/train/${task}/explore_7b_sft/