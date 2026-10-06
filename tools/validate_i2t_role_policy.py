#!/usr/bin/env python3
"""Actual-data before/after contract for role-aware image placeholders."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,SiglipImageProcessor
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.omni_data import ReproducibleOmniDataset

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='models/03_omni_moe/runs/i2t_role_policy_contract_v1')
    p.add_argument('--noise',type=float,default=0.); p.add_argument('--workers',type=int,default=0); args=p.parse_args()
    torch.set_num_threads(2); out=ROOT/args.out
    if out.exists(): raise RuntimeError('Preserve previous contract')
    out.mkdir(); (out/'source_validation.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter(); corpus=ROOT/'data/processed/sft_i2t_shards_v1'
    raw_problem=np.load(corpus/'placeholder_audit_v1/assistant_image_count.npy',mmap_mode='r')
    sampled=[json.loads(l)['row'] for l in (ROOT/'reports/omni_i2t_batches_v1/length_1536_epoch_0.jsonl').read_text().splitlines()]
    extra=[r['row'] for r in json.loads((ROOT/'reports/omni_i2t_placeholder_audit_v1.json').read_text())['examples']]
    rows=np.unique(np.asarray(sampled+extra,dtype='<u8'))
    common=dict(data_path=str(corpus),tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model'),
                vision_processor=SiglipImageProcessor.from_pretrained(ROOT/'data/frozen_models/siglip2-base-p32-256-ve'),
                max_length=1536,rows=rows,seed=20261006,scheduled_sampling=args.noise)
    before=ReproducibleOmniDataset(**common); after=ReproducibleOmniDataset(**common,image_placeholder_policy='user_only')
    records=[]; unchanged=changed=0
    for epoch in [0,1]:
        for i,row in enumerate(rows):
            a,b=before[(epoch,i)],after[(epoch,i)]
            if not raw_problem[row]:
                assert equal(a,b),f'Unrelated clean/text-only row changed: {row}/{epoch}'
                unchanged+=1
            else:
                assert int((b[1]==12).sum())==0,f'Image control tokens remain supervised in assistant: {row}'
                image_count=int((b[0][8]==12).sum())
                assert image_count>=64,f'User image prefix lost: {row}'
                if not args.noise: assert image_count==64,f'Expected exactly one user image: {row}'
                assert equal(a[5],b[5]),'Decoded image/processor changed'
                changed+=1
                records.append(dict(row=int(row),epoch=epoch,before_image_tokens=int((a[0][8]==12).sum()),after_image_tokens=int((b[0][8]==12).sum()),
                                    before_supervised_image_tokens=int((a[1]==12).sum()),after_supervised_image_tokens=int((b[1]==12).sum()),
                                    before_text_labels=int((a[1]!=-100).sum()),after_text_labels=int((b[1]!=-100).sum())))
    worker_draws=0
    if args.workers:
        loader=torch.utils.data.DataLoader(after,batch_size=None,num_workers=args.workers,
                                          generator=torch.Generator().manual_seed(20261006))
        for i,batch in enumerate(loader):
            assert equal(batch,after[(0,i)]),f'Worker/main-process augmentation mismatch at position {i}'
            worker_draws+=1
    report=dict(passed=True,unchanged_clean_draws=unchanged,repaired_draws=changed,worker_equal_draws=worker_draws,
                scheduled_sampling=args.noise,workers=args.workers,records=records,seconds=time.perf_counter()-start,
                scope='256 uniformly sampled rows plus first12 problematic raw rows, two epochs; actual official random replacement probability recorded separately')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
