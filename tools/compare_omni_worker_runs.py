#!/usr/bin/env python3
"""Reproduce worker comparison from retained logs and CPU-mapped checkpoints."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise RuntimeError('Preserve previous report')
    torch.set_num_threads(2)
    paths=[Path(a.before),Path(a.after)]
    series=[[r for r in map(json.loads,(p/'metrics.jsonl').read_text().splitlines()) if r.get('event')=='train'] for p in paths]
    assert len(series[0])==len(series[1])>0
    report={'runs':[str(p) for p in paths],'updates':len(series[0]),'scope':'No GPU allocation; shared source/RNG/counters do not by themselves prove equal numerical trajectories'}
    report['identical_counters']={k:all(x[k]==y[k] for x,y in zip(*series)) for k in ['step','cursor','text_labels','audio_labels','data','skipped_microbatches']}
    report['maximum_absolute_metric_differences']={k:max(abs(x[k]-y[k]) for x,y in zip(*series)) for k in ['text_ce','audio_ce','aux_loss','loss','grad_norm']}
    states=[torch.load(p/'checkpoints/latest_resume.pt',map_location='cpu',mmap=True,weights_only=False) for p in paths]
    report['identical_rng']={k:equal(states[0]['rng'][k],states[1]['rng'][k]) for k in states[0]['rng']}
    report['identical_state_metadata']={k:equal(states[0][k],states[1][k]) for k in ['cursor','step','epoch','fingerprints']}
    report['deterministic']=[s['config']['deterministic'] for s in states]
    report['final_export_sha256']=[digest(p/'checkpoints'/f"model_step_{s['step']:07d}.pth") for p,s in zip(paths,states)]
    report['script_sha256']=digest(Path(__file__))
    assert not torch.cuda.is_initialized()
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
