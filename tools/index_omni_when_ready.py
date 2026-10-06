#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['sft_t2a','sft_a2a','sft_i2t']:
    corpus=ROOT/'data/processed'/f'{name}_shards_v1'
    print(f'Waiting for complete shards: {name}',flush=True)
    while not (corpus/'index.json').exists(): time.sleep(10)
    command=[sys.executable,'-u','-B',str(ROOT/'tools/index_omni.py'),'--corpus',str(corpus)]
    if not (corpus/'audit_v1/metadata.json').exists():
        with (ROOT/'logs'/f'index_{name}_full.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'ready':name,'metadata':str(corpus/'audit_v1/metadata.json')}),flush=True)
