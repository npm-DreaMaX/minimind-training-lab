#!/usr/bin/env python3
"""CPU analysis of direct state-decay initialization; no checkpoint modification."""
import hashlib,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'models/02_hybrid_moe/learning/initial_decay_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes());torch.set_num_threads(2)
    weight=ROOT/'models/02_hybrid_moe/runs/transfer_preflight_early_ar/initialized_fp32.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    generator=torch.Generator().manual_seed(20261012);x=torch.randn(4096,768,generator=generator)
    records=[]
    for layer in [0,1,2,4,5,6]:
        prefix=f'model.layers.{layer}.linear_attn.'
        a=F.linear(x,state[prefix+'in_proj_a.weight']);A=state[prefix+'A_log'].exp()
        bias=state[prefix+'dt_bias'];dt=torch.exp(torch.rand(8,generator=generator)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
        alternative_bias=dt+torch.log(-torch.expm1(-dt))
        row=dict(layer=layer,A=A.tolist(),author_dt_bias=bias.tolist(),candidate_dt=dt.tolist(),candidate_dt_bias=alternative_bias.tolist())
        for name,b in [('author',bias),('fla_dt_distribution_candidate',alternative_bias)]:
            rate=A*F.softplus(a+b);retention=torch.exp(-rate)
            row[name]=dict(mean_rate_per_head=rate.mean(0).tolist(),mean_retention_per_head=retention.mean(0).tolist(),
                direct_decay_efold_positions=(1/rate.mean(0)).tolist(),
                retention_quantiles=torch.quantile(retention,torch.tensor([.01,.5,.99])).tolist())
        records.append(row)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,label,color in [('author','Author dt_bias = 1','tab:orange'),('fla_dt_distribution_candidate','FLA dt distribution candidate','tab:blue')]:
        retention=[v for r in records for v in r[name]['mean_retention_per_head']]
        horizon=[v for r in records for v in r[name]['direct_decay_efold_positions']]
        axes[0].plot(retention,'.-',label=label,color=color);axes[1].plot(horizon,'.-',label=label,color=color)
    axes[0].set_ylabel('Mean exp(g), log scale');axes[0].set_yscale('log')
    axes[1].set_ylabel('1 / mean(-g), log scale');axes[1].set_yscale('log')
    for ax in axes:ax.set_xlabel('6 linear layers x 8 heads');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Direct decay only; Gaussian inputs; not measured model context length')
    fig.savefig(out/'direct_decay.png',dpi=150);plt.close(fig)
    report=dict(scope='CPU mathematical inspection only; same real initialized A and gate projection in both variants, Gaussian unit-scale inputs. Alternative dt sampled with the algorithm in pinned FLA, not adopted or written back. Delta update, learned inputs and Full Attention also determine actual memory.',
        seed=20261012,source_weight=str(weight.relative_to(ROOT)),source_transfer_report='models/02_hybrid_moe/runs/transfer_preflight_early_ar/transfer.json',
        fla_source='sources/flash-linear-attention/fla/layers/gated_deltanet.py',
        fla_source_sha256=hashlib.sha256((ROOT/'sources/flash-linear-attention/fla/layers/gated_deltanet.py').read_bytes()).hexdigest(),records=records)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ['author','fla_dt_distribution_candidate']:
        values=torch.tensor([v for r in records for v in r[name]['direct_decay_efold_positions']])
        print(name,dict(min=float(values.min()),median=float(values.median()),max=float(values.max())),flush=True)

if __name__=='__main__':main()
