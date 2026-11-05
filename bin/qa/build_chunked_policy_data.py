#!/usr/bin/env python3
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer
sys.path.insert(0, '/data3/xhy/codes/work/paper_repos/Q-Evolve')
from QEvolve.qa_data import _qa_prompt

ROOT=Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
MODEL='/data3/xhy/codes/models/Qwen3-8B'
RUN=ROOT/'runs/olympiad_qwen3_8b_iterative1024_full_4gpu'
SRC=ROOT/'data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_full_4gpu/clean.jsonl'
QV=ROOT/'advantage_inference/qa_traj_tokens/wo_assigned_rewards/olympiad_chunked/adv_token_pred-gamma0.95-len32768-olympiad_chunked.jsonl'
OUT_DATA=RUN/'policy_chunked8192_data.jsonl'
OUT_QV=RUN/'policy_chunked8192_adv.jsonl'
OUT_SUM=RUN/'policy_chunked8192_summary.json'
MAX_LEN=8192
OVERLAP=2048
MAX_STEPS=64

def chunk_conv(conv, s, e):
    # Keep the instruction/OK prefix, then represent each selected original step
    # as a normal user/assistant pair. This preserves recent context overlap.
    out=[conv[0], conv[1]]
    for t in range(s, e+1):
        user=conv[2+2*t]
        assistant=conv[3+2*t]
        out.extend([user, assistant])
    return out

def plen(tok, conv):
    return len(tok(_qa_prompt(tok, MODEL, conv), add_special_tokens=False).input_ids)

def main():
    tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    rows=[json.loads(x) for x in SRC.open() if x.strip()]
    qv={int(x['traj_id']):x for x in (json.loads(y) for y in QV.open() if y.strip())}
    out_rows=[]; out_qv=[]; dropped=[]; overlap_lens=[]
    for orig_idx,row in enumerate(rows):
        conv=row.get('conversations',[])
        T=(len(conv)-3)//2 if len(conv)>=4 else 0
        if T<=0 or orig_idx not in qv:
            dropped.append({'orig_traj_id':orig_idx,'reason':'missing_steps_or_advantage'}); continue
        A=[float(x) for x in qv[orig_idx]['A']]
        if len(A)<T:
            dropped.append({'orig_traj_id':orig_idx,'reason':'advantage_short','steps':T,'A':len(A)}); continue
        s=0; chunks=0
        while s<T:
            best=None
            for e in range(s, min(T, s+MAX_STEPS)):
                c=chunk_conv(conv,s,e)
                L=plen(tok,c)
                if L<=MAX_LEN:
                    best=(e,L)
                else:
                    break
            if best is None:
                dropped.append({'orig_traj_id':orig_idx,'reason':'single_step_over_limit','step':s,'tokens':plen(tok,chunk_conv(conv,s,s))})
                break
            e,L=best
            c=chunk_conv(conv,s,e)
            new_idx=len(out_rows)
            r=dict(row)
            r['id']=f"{row.get('id',orig_idx)}::chunk{s}-{e}"
            r['conversations']=c
            r['assigned_reward']=[0.0]*(e-s+1)
            r['meta']={'reward':0.0,'success':False,'finished':True,'terminate_reason':'chunk_boundary','orig_id':row.get('id'),'orig_traj_id':orig_idx,'orig_step_start':s,'orig_step_end':e}
            out_rows.append(r)
            a=A[s:e+1]; n=len(a)
            out_qv.append({'traj_id':new_idx,'orig_traj_id':orig_idx,'orig_step_start':s,'orig_step_end':e,
                'ep_len':n,'final_done':0,'is_fail':0,'valid_steps':list(range(n)),
                'A':a,'reward':[0.0]*n,'done':[0.0]*n,'V':[0.0]*n,'V_next':[0.0]*n,
                'Q1':[0.0]*n,'Q2':[0.0]*n})
            chunks+=1
            if e>=T-1: break
            # Choose the largest recent suffix whose prompt is <= OVERLAP tokens.
            ns=e
            for cand in range(e, s, -1):
                if plen(tok,chunk_conv(conv,cand,e))<=OVERLAP:
                    ns=cand
            if ns<=s: ns=e
            overlap_lens.append(plen(tok,chunk_conv(conv,ns,e)))
            s=ns

    OUT_DATA.parent.mkdir(parents=True,exist_ok=True)
    with OUT_DATA.open('w') as f:
        for r in out_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with OUT_QV.open('w') as f:
        for r in out_qv: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'source_trajectories':len(rows),'source_advantage_records':len(qv),'chunk_rows':len(out_rows),'dropped_source_rows':len(dropped),'max_length':MAX_LEN,'overlap_budget':OVERLAP,'max_steps_per_chunk':MAX_STEPS,'mean_overlap_tokens':sum(overlap_lens)/len(overlap_lens) if overlap_lens else 0,'dropped':dropped}
    OUT_SUM.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
