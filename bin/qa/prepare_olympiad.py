#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parquet',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    df=pd.read_parquet(args.parquet); rows=[]
    for _,r in df.iterrows():
        sol=r.get('solution',''); gold=r.get('final_answer','')
        if hasattr(sol,'tolist'): sol=sol.tolist()
        if hasattr(gold,'tolist'): gold=gold.tolist()
        if isinstance(sol,list): sol=sol[0] if sol else ''
        if isinstance(gold,list): gold=gold[0] if gold else ''
        rows.append({'id':str(r['id']),'question':str(r['question']),'final_answer':str(gold),'solution':str(sol)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(args.output).open('w') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print('wrote',len(rows),args.output)
if __name__=='__main__': main()
