#!/usr/bin/env bash
set -u
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
LOG=$ROOT/runs/olympiad_monitor_4096.log
mkdir -p "$ROOT/runs"
exec >>"$LOG" 2>&1
while true; do
  echo "===== $(date '+%F %T') ====="
  echo '[processes]'
  ps -eo pid,etime,stat,cmd | grep -E 'olympiad|qa_rollout.py|train_sft.py|train_critic.py|inf_critic.py|train_policy.py' | grep -v grep | head -30 || true
  echo '[rollout metrics]'
  /data3/xhy/miniconda3/envs/pettingllms/bin/python - <<'PY'
import glob,json,os
ps=sorted(glob.glob('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/olympiad_rollouts_8_4096/part*.jsonl'))
n=f=s=0
for p in ps:
 rs=[]
 for line in open(p):
  try: rs.append(json.loads(line))
  except Exception: pass
 nf=sum(bool(x.get('meta',{}).get('finished')) for x in rs); ns=sum(bool(x.get('meta',{}).get('success')) for x in rs)
 n+=len(rs); f+=nf; s+=ns
 print(os.path.basename(p),len(rs),'finished',nf,'correct',ns)
print('TOTAL',n,'/5392','finished',f,'correct',s)
PY
  echo '[training metrics]'
  for f in sft_success critic advantage filter_advantage policy; do
    p="$ROOT/runs/olympiad_qevolve_4096/logs/${f}.log"
    [ -f "$p" ] && { echo "### $f"; grep -E 'train_loss|Qmin_mean|V_mean|A_mean|td_err_mse|steps_used|kept ' "$p" | tail -3; }
  done
  echo '[gpu]'
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  sleep 60
done
