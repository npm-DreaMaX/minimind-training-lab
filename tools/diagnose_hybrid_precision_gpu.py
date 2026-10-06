#!/usr/bin/env python3
"""Separate projection rounding, recurrent-kernel dtype and reference gradients.

Runs identical full-width layers through FLA and an independent FP32 sequential
recurrence. This is a numerical diagnostic, never a silent model modification.
"""
import argparse,copy,fcntl,gc,hashlib,importlib.util,json,os,shutil,time
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
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--triton-precision',choices=['ieee','tf32','tf32x3'])
    p.add_argument('--fp32-only',action='store_true'); p.add_argument('--capture-kernel',action='store_true')
    p.add_argument('--sequence-length',type=int,default=129)
    p.add_argument('--enforce-fp32-tolerance',action='store_true',help='Require <2%% output and <3%% each gradient vs independent recurrence')
    a=p.parse_args(); out=ROOT/a.out
    if a.triton_precision: os.environ['TRITON_F32_DEFAULT']=a.triton_precision
    if out.exists(): raise RuntimeError('Preserve old precision diagnostics')
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); shutil.copy2(__file__,out/'source_diagnosis.py')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
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
            x=torch.randn(1,a.sequence_length,768,device='cuda'); probe=torch.randn_like(x); results={}
            variants=[('fp32_reference','float32',sequential),('fp32_fla','float32',original)]
            if not a.fp32_only:
                variants += [('bf16_reference','bfloat16',sequential),('bf16_fla','bfloat16',original),('bf16_projections_fp32_kernel','bfloat16',float_kernel)]
            for variant,precision,kernel in variants:
                layer=copy.deepcopy(template); layer.linear_precision=precision; module.chunk_gated_delta_rule=kernel
                captured={}
                if a.capture_kernel and variant=='fp32_fla':
                    def capturing(q,k,v,g,beta,**kwargs):
                        captured['inputs']=dict(q=q,k=k,v=v,g=g,beta=beta)
                        for tensor in captured['inputs'].values(): tensor.retain_grad()
                        output,state=original(q,k,v,g=g,beta=beta,**kwargs)
                        output.register_hook(lambda gradient: captured.update(do=gradient.detach().cpu()))
                        return output,state
                    module.chunk_gated_delta_rule=capturing
                inp=x.detach().clone().requires_grad_(); start=time.perf_counter(); output,_=layer(inp)
                objective=(output*probe).mean(); objective.backward(); torch.cuda.synchronize()
                if captured:
                    torch.save(dict(inputs={k:v.detach().cpu() for k,v in captured['inputs'].items()},
                                    gradients={k:v.grad.detach().cpu() for k,v in captured['inputs'].items()},do=captured['do']),
                               out/f'kernel_seed_{seed}.pt')
                    captured.clear()
                results[variant]=dict(output=output.detach().cpu(),input_gradient=inp.grad.detach().cpu(),
                                      gradients={name:param.grad.detach().cpu() for name,param in layer.named_parameters()})
                record=dict(event='variant',seed=seed,variant=variant,objective=float(objective.detach()),seconds=time.perf_counter()-start)
                all_records.append(record); print(json.dumps(record),flush=True)
                del layer,inp,output,objective; gc.collect(); torch.cuda.empty_cache()
            pairs=[('fp32_reference','fp32_fla'),('fp32_reference','bf16_reference'),('bf16_reference','bf16_fla'),
                   ('fp32_reference','bf16_fla'),('fp32_reference','bf16_projections_fp32_kernel')]
            for left,right in pairs:
                if left not in results or right not in results: continue
                record=dict(event='comparison',seed=seed,reference=left,candidate=right,
                            output=error(results[left]['output'],results[right]['output']),
                            input_gradient=error(results[left]['input_gradient'],results[right]['input_gradient']),
                            gradients={name:error(value,results[right]['gradients'][name]) for name,value in results[left]['gradients'].items()})
                all_records.append(record); print(json.dumps(record),flush=True)
            del template,x,probe,results; gc.collect(); torch.cuda.empty_cache()
        if a.enforce_fp32_tolerance:
            comparisons=[r for r in all_records if r['event']=='comparison' and r['reference']=='fp32_reference' and r['candidate']=='fp32_fla']
            assert len(comparisons)==2
            assert all(r['output']['finite'] and r['output']['relative_rms']<.02 and r['input_gradient']['finite'] and r['input_gradient']['relative_rms']<.03
                       and all(v['finite'] and v['relative_rms']<.03 for v in r['gradients'].values()) for r in comparisons), 'FP32 FLA did not pass the independent reference gate'
    finally:
        module.chunk_gated_delta_rule=original
        (out/'results.json').write_text(json.dumps(dict(records=all_records,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             scope=f'Full-width isolated layer, B1 T{a.sequence_length}, two initialization seeds; independent sequential FP32 recurrence; no formal recipe adopted',
             triton_f32_default=os.environ.get('TRITON_F32_DEFAULT','unset (backend default)'),time=time.time()),indent=2)+'\n')

if __name__=='__main__': main()
