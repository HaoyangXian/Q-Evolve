#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); args=ap.parse_args(); kept=0; total=0
 with Path(args.output).open('w') as out:
  for line in Path(args.input).open():
   if not line.strip(): continue
   total+=1; x=json.loads(line); a=x.get('A',[])
   if any(abs(float(v))>1e-8 for v in a): out.write(json.dumps(x)+'\n'); kept+=1
 print(f'kept {kept}/{total}')
if __name__=='__main__': main()
