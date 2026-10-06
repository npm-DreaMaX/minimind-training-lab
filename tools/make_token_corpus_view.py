#!/usr/bin/env python3
"""Uniform row-index view of an immutable full cache; large files are hardlinks."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p.add_argument('--train-rows',type=int,default=512); p.add_argument('--val-rows',type=int,default=32)
    p.add_argument('--seq-len',type=int,default=1536); p.add_argument('--seed',type=int,default=20261007)
    a=p.parse_args(); root=Path(a.source).resolve(); out=Path(a.out)
    if out.exists(): raise RuntimeError('Preserve previous sampling evidence')
    source=json.loads((root/'metadata.json').read_text()); assert source['limit']==0 and source['bad_rows']==0
    out.mkdir(parents=True); generator=np.random.default_rng(a.seed); counts={}
    for name in ['tokens.bin','labels.bin','offsets.npy','first_targets.npy','lengths.npy','content_hash64.npy']:
        if (root/name).exists(): os.link(root/name,out/name)
    for split,count in [('train',a.train_rows),('val',a.val_rows)]:
        rows=np.load(root/f'{split}.npy',mmap_mode='r')
        if source['stage']=='sft':
            first=np.load(root/'first_targets.npy',mmap_mode='r'); rows=rows[first[rows]<a.seq_len]
        chosen=np.sort(generator.choice(rows,size=min(len(rows),count),replace=False))
        np.save(out/f'{split}.npy',chosen); counts[split]=len(chosen)
    report=dict(stage=source['stage'],scope='Random full-corpus subset for preflight only; not a formal full epoch',
                source=str(root),source_metadata_sha256=hashlib.sha256((root/'metadata.json').read_bytes()).hexdigest(),
                counts=counts,seed=a.seed,seq_len=a.seq_len,storage='Read-only usage of hardlinked immutable cache arrays; independent row index files')
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
