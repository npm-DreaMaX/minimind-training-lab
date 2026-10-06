#!/usr/bin/env python3
"""Convert each verified full corpus with bounded reads; keep failures for review."""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
expected={'sft_t2a':1248923,'sft_a2a':414024,'sft_i2t':2904511}
for name,rows in expected.items():
    raw=ROOT/'data/raw/omni'/f'{name}.parquet'
    out=ROOT/'data/processed'/f'{name}_shards_v1'
    print(f'Waiting for verified {raw.name}',flush=True)
    while not Path(str(raw)+'.verified.json').exists(): time.sleep(10)
    if not (out/'index.json').exists():
        command=['/usr/bin/time','-v',sys.executable,'-u','-B',str(ROOT/'tools/repack_omni.py'),'--raw',str(raw),'--out',str(out)]
        with (ROOT/'logs/phase_commands.jsonl').open('a') as f:
            f.write(json.dumps({'command':command,'time':time.time(),'cwd':str(ROOT)})+'\n')
        with (ROOT/'logs'/f'repack_{name}_full.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((out/'index.json').read_text())
    assert meta['rows']==rows and meta['scope']=='full verified corpus',meta
    print(json.dumps({'event':'full_shards_ready','name':name,'rows':rows,'out':str(out)}),flush=True)
