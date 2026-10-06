#!/usr/bin/env python3
"""Plot measured resident/offloaded AdamW phases from the real T2A diagnostic."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
cases=[('Resident moments; accumulation 2','real_t2a_accum2_diagnostic_v2'),
       ('CPU moments; accumulation 2','real_t2a_offload_v2')]
fig,axes=plt.subplots(2,1,figsize=(15,10),constrained_layout=True)
for ax,(label,name) in zip(axes,cases):
    path=ROOT/'models/03_omni_moe/runs'/name
    rows=[r for line in (path/'metrics.jsonl').open() if (r:=json.loads(line)).get('event')=='memory_phase' and r['step']<2]
    x=list(range(len(rows)))
    ticks=[f"{r['step']+1}.{r['microbatch'] if r['microbatch'] is not None else '-'} {r['phase'].replace('before_','pre ').replace('after_','post ')}" for r in rows]
    ax.plot(x,[r['allocated']/1024**3 for r in rows],marker='o',label='Live CUDA tensors')
    ax.plot(x,[r['reserved']/1024**3 for r in rows],alpha=.65,label='Allocator reserved')
    ax.plot(x,[r['offload_host_bytes']/1024**3 for r in rows],linestyle='--',label='Host moment buffers')
    ax.axhline(5.3,color='red',linestyle=':',label='Explicit allocator budget 5.3 GiB')
    if (path/'failure.json').exists(): ax.annotate('OOM in the following forward',xy=(x[-1],rows[-1]['allocated']/1024**3),xytext=(x[-1]-7,5.6),arrowprops=dict(arrowstyle='->',color='red'),color='red')
    ax.set_xticks(x,ticks,rotation=65,ha='right',fontsize=7); ax.set_ylabel('GiB'); ax.set_ylim(0,6); ax.set_title(label); ax.grid(alpha=.2); ax.legend(loc='upper left',fontsize=8,ncol=2)
fig.suptitle('Full 315M Omni, real T2A, B1 x T1536, BF16 autocast / FP32 AdamW, block recomputation\nPhase labels: optimizer update.microbatch; reserved memory is not live tensors')
out=ROOT/'reports/omni_accumulation_memory.png'; fig.savefig(out,dpi=150); print(out)
