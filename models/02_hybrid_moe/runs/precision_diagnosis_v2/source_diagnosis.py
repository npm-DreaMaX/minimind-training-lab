#!/usr/bin/env python3
"""Separate projection rounding, recurrent-kernel dtype and reference gradients.

Runs identical full-width layers through FLA and an independent FP32 sequential
recurrence. This is a numerical diagnostic, never a silent model modification.
"""
import argparse,copy,fcntl,gc,hashlib,importlib.util,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def sequential(q,k,v,g,beta,initial_state=None,output_final_state=False,**_):
    # State [B,H,Dk,Dv], normalized q/k already supplied by the model.
    dtype=v.dtype; q,k,v,g,beta=[t.float() for t in (q,k,v,g,beta)]
    b,t,h,dk=q.shape; dv=v.shape[-1]
    state=q.new_zeros(b,h,dk,dv) if initial_state is None else initial_state.float()
    outputs=[]
    for i in range(t):
        state=state*g[:,i].exp()[...,None,None]
        prediction=(state*k[:,i,...,None]).sum(-2)
        error=(v[:,i]-prediction)*beta[:,i,...,None]
        state=state+k[:,i,...,None]*error[...,None,:]
        outputs.append((state*(q[:,i]*dk**-.5)[...,None]).sum(-2))
    return torch.stack(outputs,dim=1).to(dtype),state if output_final_state else None

def error(a,b):
    a,b=a.float(),b.float(); scale=float(a.square().mean().sqrt()); delta=float((a-b).square().mean().sqrt())
    return dict(reference_rms=scale,error_rms=delta,relative_rms=delta/max(scale,1e-20),max_absolute=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve old precision diagnostics')
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(5.3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    path=ROOT/'models/02_hybrid_moe/src/model_hybrid.py'; spec=importlib.util.spec_from_file_location('diagnosed_hybrid',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); original=module.chunk_gated_delta_rule
    def float_kernel(q,k,v,**kwargs):
        return original(q.float(),k.float(),v.float(),**kwargs)
    all_records=[]
    try:
        for seed in [20261007,20261008]:
            torch.manual_seed(seed)
            template=module.GatedDeltaNet(module.MiniMindConfig(hidden_size=768,require_fla=True),0).cuda()
            x=torch.randn(1,129,768,device='cuda'); probe=torch.randn_like(x); results={}
            for variant,precision,kernel in [('fp32_reference','float32',sequential),('fp32_fla','float32',original),
                    ('bf16_reference','bfloat16',sequential),('bf16_fla','bfloat16',original),('bf16_projections_fp32_kernel','bfloat16',float_kernel)]:
                layer=copy.deepcopy(template); layer.linear_precision=precision; module.chunk_gated_delta_rule=kernel
                inp=x.detach().clone().requires_grad_(); start=time.time(); output,_=layer(inp)
                objective=(output*probe).mean(); objective.backward(); torch.cuda.synchronize()
                results[variant]=dict(output=output.detach().cpu(),input_gradient=inp.grad.detach().cpu(),
                                      gradients={name:param.grad.detach().cpu() for name,param in layer.named_parameters()})
                record=dict(event='variant',seed=seed,variant=variant,objective=float(objective.detach()),seconds=time.time()-start)
                all_records.append(record); print(json.dumps(record),flush=True)
                del layer,inp,output,objective; gc.collect(); torch.cuda.empty_cache()
            pairs=[('fp32_reference','fp32_fla'),('fp32_reference','bf16_reference'),('bf16_reference','bf16_fla'),
                   ('fp32_reference','bf16_fla'),('fp32_reference','bf16_projections_fp32_kernel')]
            for left,right in pairs:
                record=dict(event='comparison',seed=seed,reference=left,candidate=right,
                            output=error(results[left]['output'],results[right]['output']),
                            input_gradient=error(results[left]['input_gradient'],results[right]['input_gradient']),
                            gradients={name:error(value,results[right]['gradients'][name]) for name,value in results[left]['gradients'].items()})
                all_records.append(record); print(json.dumps(record),flush=True)
            del template,x,probe,results; gc.collect(); torch.cuda.empty_cache()
    finally:
        module.chunk_gated_delta_rule=original
        (out/'results.json').write_text(json.dumps(dict(records=all_records,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             scope='Full-width isolated layer, B1 T129, two initialization seeds; independent sequential FP32 recurrence; no formal recipe adopted',time=time.time()),indent=2)+'\n')

if __name__=='__main__': main()
