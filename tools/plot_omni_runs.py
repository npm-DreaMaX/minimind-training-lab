#!/usr/bin/env python3
"""Render mirrored formal Omni metrics locally, using no GPU or remote packages."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def plot(run):
    identifier=hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:16]
    render_lock=(ROOT/f'runs/omni_plot_{identifier}.lock').open('a')
    fcntl.flock(render_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records={};evaluations=[]
    lines=(run/'metrics.jsonl').read_text().splitlines()
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:continue # live append may be incomplete
            raise
        if row.get('event')=='train':records[row['step']]=row
        elif row.get('event')=='evaluation':evaluations.append(row)
    if not records:return
    rows=[records[k] for k in sorted(records)];steps=[r['step'] for r in rows]
    out=run/'plots';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    for key in ['loss','text_ce','audio_ce','aux_loss']:
        axes[0,0].plot(steps,[r[key] for r in rows],label=key,linewidth=.7,alpha=.8)
    for field in ['text_ce','audio_ce']:
        fixed=[r for r in evaluations if not r.get('final')]
        full=[r for r in evaluations if r.get('final')]
        axes[0,1].plot([r['step'] for r in fixed],[r[field] for r in fixed],'.-',label='fixed subset '+field)
        if full:axes[0,1].scatter([r['step'] for r in full],[r[field] for r in full],marker='x',label='full split '+field)
    axes[1,0].plot(steps,[r['learning_rate'] for r in rows],label='learning rate')
    axes[1,1].plot(steps,[r['grad_norm'] for r in rows],label='gradient norm before clipping',linewidth=.7)
    axes[2,0].plot(steps,[r['samples_per_second'] for r in rows],label='samples / second',linewidth=.7)
    for key in ['cuda_allocated','cuda_reserved','cuda_peak_allocated']:
        if key in rows[0]:axes[2,1].plot(steps,[r[key]/1024**3 for r in rows],label=key,linewidth=.7)
    for ax in axes.flat:ax.set_xlabel('optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[2,1].set_ylabel('GiB (process allocator)')
    fig.suptitle(run.name+'\nLatest retained attempt per step; raw attempts remain in metrics.jsonl',fontsize=11)
    temp=out/'training.tmp.png';fig.savefig(temp,dpi=140);plt.close(fig);temp.replace(out/'training.png')
    summary=dict(last_step=steps[-1],plotted_steps=len(rows),total_text_labels=rows[-1]['total_text_labels'],
                 total_audio_labels_across_eight_heads=rows[-1]['total_audio_labels'],
                 last_metric_time=rows[-1]['time'],generated=time.time(),
                 note='Audio label count sums eight codebooks; do not equate it with independent text tokens. GPU allocated/reserved exclude other processes and driver context.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--run');a=p.parse_args()
    if a.watch:
        lock=(ROOT/'runs/omni_curves.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seen={}
    while True:
        runs=[ROOT/a.run] if a.run else list((ROOT/'models/03_omni_moe/runs').glob('omni_0*_full_v1'))
        for run in runs:
            path=run/'metrics.jsonl'
            if path.exists() and seen.get(str(run))!=path.stat().st_mtime_ns:
                plot(run);seen[str(run)]=path.stat().st_mtime_ns
        if not a.watch:return
        time.sleep(300)


if __name__=='__main__':main()
