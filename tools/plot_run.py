#!/usr/bin/env python3
"""Render retained training metrics to an offline PNG without external services."""
import argparse,csv,json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('run'); a=p.parse_args(); root=Path(a.run)
rows=[]; evaluations=[]
for line in (root/'metrics.jsonl').open():
    try: r=json.loads(line)
    except json.JSONDecodeError: continue
    if r.get('event')=='train': rows.append(r)
    elif r.get('event')=='evaluation': evaluations.append(r)
if not rows: raise SystemExit('No training metrics yet')
x=[r['step'] for r in rows]
omni='text_ce' in rows[0]
keys=['text_ce','audio_ce','learning_rate','grad_norm','samples_per_second','cuda_peak_allocated'] if omni else ['ce_loss','aux_loss','learning_rate','grad_norm','valid_tokens_per_second','cuda_peak_allocated']
labels=['Text CE','Audio CE (STOP weighted)','Learning rate','Gradient norm before clipping','Samples / second','Peak PyTorch allocated (GiB)'] if omni else ['Token-weighted CE','MoE auxiliary loss','Learning rate','Gradient norm before clipping','Supervised tokens / second','Peak PyTorch allocated (GiB)']
fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
for ax,key,label in zip(axes.flat,keys,labels):
    values=[r.get(key,float('nan'))/(1024**3 if key=='cuda_peak_allocated' else 1) for r in rows]
    ax.plot(x,values,linewidth=1,label=label); ax.set_title(label); ax.set_xlabel('Optimizer step'); ax.grid(alpha=.25)
    if key=='learning_rate':
        # Show the actual LR scale: a tiny early cosine change must not look
        # like a large drop because of an automatic offset and zoomed y-axis.
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
        ax.set_ylim(0,max(values)*1.1 if max(values)>0 else 1)
    if key=='ce_loss' and evaluations:
        ax.plot([r['step'] for r in evaluations],[r['validation_ce'] for r in evaluations],marker='o',label='Held-out CE'); ax.legend()
    if omni and key in ['text_ce','audio_ce'] and evaluations:
        subset=[r for r in evaluations if not r.get('final')]
        ax.plot([r['step'] for r in subset],[r[key] for r in subset],marker='o',label='Held-out CE'); ax.legend()
fig.suptitle(root.name+' — recorded measurements')
(root/'plots').mkdir(exist_ok=True)
temporary=root/'plots/training.tmp.png'; fig.savefig(temporary,dpi=150); temporary.replace(root/'plots/training.png'); plt.close(fig)
if (root/'gpu.csv').exists():
    telemetry=[]
    with (root/'gpu.csv').open() as stream:
        for row in csv.DictReader(stream,skipinitialspace=True):
            try:
                stamp=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp()
                def number(key): return float(row[key].split()[0])
                telemetry.append((stamp,number('memory.used [MiB]')/1024,number('utilization.gpu [%]'),number('temperature.gpu'),number('power.draw [W]')))
            except (KeyError,ValueError,TypeError): continue
    if telemetry:
        fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
        minutes=[(r[0]-telemetry[0][0])/60 for r in telemetry]
        for i,(ax,label) in enumerate(zip(axes.flat,['Device VRAM including desktop (GiB)','GPU utilization (%)','GPU temperature (C)','GPU power (W)']),1):
            ax.plot(minutes,[r[i] for r in telemetry],linewidth=1); ax.set_title(label); ax.set_xlabel('Elapsed wall-clock minutes'); ax.grid(alpha=.25)
        fig.suptitle(root.name+' — nvidia-smi telemetry')
        temporary=root/'plots/system.tmp.png'; fig.savefig(temporary,dpi=150); temporary.replace(root/'plots/system.png'); plt.close(fig)
routed=[r for r in rows if r.get('router_first_microbatch_nonpadding')]
if routed:
    layers=sorted(routed[-1]['router_first_microbatch_nonpadding'],key=int)
    fig,axes=plt.subplots((len(layers)+1)//2,2,figsize=(13,2.6*((len(layers)+1)//2)),squeeze=False,constrained_layout=True)
    summary={'scope':'First microbatch non-padding token routing at recorded steps; sampled diagnostic, not whole-corpus expert load',
             'observations':len(routed),'latest_step':routed[-1]['step'],'recent_window_steps':[r['step'] for r in routed[-10:]],'layers':{}}
    for ax,layer in zip(axes.flat,layers):
        observations=[(r['step'],r['router_first_microbatch_nonpadding'][layer]) for r in routed if layer in r['router_first_microbatch_nonpadding']]
        experts=len(observations[-1][1]['tokens_per_expert'])
        for expert in range(experts):
            ax.plot([s for s,_ in observations],[v['tokens_per_expert'][expert]/sum(v['tokens_per_expert']) for _,v in observations],label=f'Expert {expert}',linewidth=1)
        ax.axhline(1/experts,color='grey',linestyle=':',linewidth=.8); ax.set_ylim(0,1)
        ax.set_title(f'Layer {layer}: fraction of routed tokens'); ax.set_xlabel('Optimizer step'); ax.grid(alpha=.2)
        counts=[sum(v['tokens_per_expert'][e] for _,v in observations[-10:]) for e in range(experts)]
        summary['layers'][layer]=dict(recent_sampled_tokens=sum(counts),recent_token_weighted_share=[c/sum(counts) for c in counts],latest=observations[-1][1])
    for ax in list(axes.flat)[len(layers):]: ax.set_visible(False)
    axes[0,0].legend(ncol=4,fontsize=8)
    fig.suptitle(root.name+' — sampled MoE routing (padding excluded)')
    temp=root/'plots/routing.tmp.png'; fig.savefig(temp,dpi=150); temp.replace(root/'plots/routing.png'); plt.close(fig)
    temp=root/'plots/routing_summary.json.tmp'; temp.write_text(json.dumps(summary,indent=2)+'\n'); temp.replace(root/'plots/routing_summary.json')
