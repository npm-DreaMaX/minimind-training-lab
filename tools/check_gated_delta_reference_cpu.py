#!/usr/bin/env python3
"""Validate the independent recurrence using FP64 and central differences.

Optional captured FLA inputs/adjoints isolate its kernel backward from the
surrounding projections. CPU only; never takes the GPU away from formal AR.
"""
import argparse,gc,json,resource,shutil,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]

def recurrence(q,k,v,g,beta):
    b,t,h,d=q.shape; state=q.new_zeros(b,h,d,v.shape[-1]); outputs=[]
    for i in range(t):
        state=state*g[:,i].exp()[...,None,None]
        residual=(v[:,i]-(state*k[:,i,...,None]).sum(-2))*beta[:,i,...,None]
        state=state+k[:,i,...,None]*residual[...,None,:]
        outputs.append((state*(q[:,i]*d**-.5)[...,None]).sum(-2))
    return torch.stack(outputs,dim=1)

def error(a,b):
    a,b=a.double(),b.double(); rms=float(a.square().mean().sqrt()); delta=float((a-b).square().mean().sqrt())
    return dict(reference_rms=rms,error_rms=delta,relative_rms=delta/max(rms,1e-30),max_absolute=float((a-b).abs().max()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--capture')
    args=p.parse_args(); out=ROOT/args.out
    if out.exists(): raise RuntimeError('Preserve previous evidence')
    out.mkdir(parents=True); shutil.copy2(__file__,out/'source_check.py'); torch.set_num_threads(2)
    records=[]; start=time.time()
    if args.capture:
        loaded=torch.load(ROOT/args.capture,map_location='cpu',weights_only=True)
        # Heads are independent in the recurrence. One at a time bounds RAM.
        cases=[(f'captured_head_{h}',{k:v[:,:,h:h+1].clone() for k,v in loaded['inputs'].items()},
                loaded['do'][:,:,h:h+1].clone(),{k:v[:,:,h:h+1].clone() for k,v in loaded['gradients'].items()})
               for h in range(loaded['inputs']['q'].shape[2])]
    else:
        torch.manual_seed(20261009); cases=[]
        for label,t,d,strength in [('mild',65,32,.2),('strong',129,96,24.)]:
            shape=(1,t,1,d)
            inputs=dict(q=F.normalize(torch.randn(shape),dim=-1),k=F.normalize(torch.randn(shape),dim=-1),
                        v=torch.randn(shape),g=-torch.rand(shape[:-1])*strength,beta=torch.randn(shape[:-1]).sigmoid())
            cases.append((label,inputs,torch.randn(shape)/t/d,None))
    try:
        for label,inputs,adjoint,fla in cases:
            results={}
            for dtype in [torch.float64,torch.float32]:
                x={k:v.to(dtype).detach().requires_grad_() for k,v in inputs.items()}
                y=recurrence(**x); (y*adjoint.to(dtype)).sum().backward()
                results[dtype]=dict(output=y.detach(),gradients={k:v.grad.detach() for k,v in x.items()})
                del x,y
            ref=results[torch.float64]; fp=results[torch.float32]
            row=dict(case=label,fp32_vs_fp64=dict(output=error(ref['output'],fp['output']),
                     gradients={k:error(v,fp['gradients'][k]) for k,v in ref['gradients'].items()}))
            if fla: row['fla_vs_fp64']={k:error(v,fla[k]) for k,v in ref['gradients'].items()}
            # Largest-gradient coordinates make the finite-difference signal measurable.
            x={k:v.double() for k,v in inputs.items()}; checks=[]; eps=1e-4
            for idx in ref['gradients']['g'].abs().flatten().topk(4).indices.tolist():
                original=float(x['g'].flatten()[idx]); values=[]
                with torch.no_grad():
                    for delta in [eps,-eps]:
                        x['g'].flatten()[idx]=original+delta
                        values.append(float((recurrence(**x)*adjoint.double()).sum()))
                x['g'].flatten()[idx]=original
                finite=(values[0]-values[1])/(2*eps); analytic=float(ref['gradients']['g'].flatten()[idx])
                passed=abs(finite-analytic) <= 1e-10+abs(analytic)*1e-6
                checks.append(dict(index=idx,finite_difference=finite,analytic=analytic,passed=passed))
            row['finite_differences']=checks
            row['reference_passed']=all(c['passed'] for c in checks) and all(e['relative_rms']<1e-4 for e in row['fp32_vs_fp64']['gradients'].values())
            records.append(row); print(json.dumps(row),flush=True)
            del results,ref,fp,x; gc.collect()
        assert all(r['reference_passed'] for r in records),'Independent recurrence reference needs investigation'
    finally:
        (out/'results.json').write_text(json.dumps(dict(capture=args.capture,records=records,seconds=time.time()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='Checks reference recurrence, not full model convergence'),indent=2)+'\n')

if __name__=='__main__': main()
