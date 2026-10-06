#!/usr/bin/env python3
"""Validate the author reference, FLA kernels, router fix and full-size hybrid.

Synthetic tensors here are numerical tests, not a training corpus. Kernel errors
must propagate: a silent fallback is not acceptable evidence of GPU feasibility.
"""
import fcntl,importlib.util,json,logging,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/02_hybrid_moe/runs/implementation_validation'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4); torch.manual_seed(20261003)
logging.basicConfig(level=logging.INFO)
records=[]
def record(**row):
    records.append(row)
    with (OUT/'results.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def error(name,ref,actual,tolerance):
    ref=ref.detach().float(); actual=actual.detach().float()
    assert torch.isfinite(actual).all(), name
    rms=float((ref-actual).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12))
    maximum=float((ref-actual).abs().max())
    record(test=name,relative_rms=rms,max_absolute=maximum,tolerance=tolerance,passed=rms<tolerance)
    assert rms<tolerance,(name,rms,tolerance)

def recurrent(q,k,v,g,beta):
    state=q.new_zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3]); out=[]
    for t in range(q.shape[1]):
        state=state*g[:,t,:,None,None].exp()
        prediction=torch.einsum('bhk,bhkv->bhv',k[:,t],state)
        innovation=beta[:,t,:,None]*(v[:,t]-prediction)
        state=state+torch.einsum('bhk,bhv->bhkv',k[:,t],innovation)
        out.append(torch.einsum('bhk,bhkv->bhv',q[:,t]*q.shape[-1]**-0.5,state))
    return torch.stack(out,dim=1),state

def main():
    fixed=module('models/02_hybrid_moe/src/model_hybrid.py','hybrid_fixed')
    author=module('sources/experimental/linear/model_minimind_author.py','hybrid_author')
    mainline=module('upstream/model/model_minimind.py','mainline')
    cfg=dict(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,
             use_moe=True,router_aux_loss_coef=0.)
    modules=[mod.MOEFeedForward(mod.MiniMindConfig(**cfg)) for mod in [author,fixed,mainline]]
    for mod in modules[1:]: mod.load_state_dict(modules[0].state_dict())
    x=torch.randn(2,11,32); target=torch.randn_like(x); outputs=[]; grads=[]
    for mod in modules:
        y=mod(x); (y*target).sum().backward(); outputs.append(y); grads.append(mod.gate.weight.grad)
    torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
    torch.testing.assert_close(outputs[1],outputs[2],atol=0,rtol=0)
    torch.testing.assert_close(grads[1],grads[2],atol=0,rtol=0)
    norms=[float(g.norm()) for g in grads]
    assert norms[1]>1e-3 and norms[0]<norms[1]*1e-5,norms
    record(test='router_task_gradient',author_fixed_mainline_norms=norms,forward_identical=True,passed=True)

    lock=(ROOT/'runs/local_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32=False
    record(event='gpu_lock_acquired',pid=os.getpid(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    B,T,H,K,V=1,33,8,96,96
    values=[F.normalize(torch.randn(B,T,H,K,device='cuda'),dim=-1),
            F.normalize(torch.randn(B,T,H,K,device='cuda'),dim=-1),
            torch.randn(B,T,H,V,device='cuda'),
            -torch.rand(B,T,H,device='cuda')*.2,torch.rand(B,T,H,device='cuda')]
    probe=torch.randn_like(values[2]); results=[]
    methods=[('independent_recurrent',recurrent),
             ('author_chunk',lambda q,k,v,g,b: author.torch_chunk_gated_delta_rule(q,k,v,g,b,chunk_size=16,output_final_state=True)),
             ('fla_chunk_fp32',lambda q,k,v,g,b: fixed.chunk_gated_delta_rule(q,k,v,g=g,beta=b,output_final_state=True))]
    assert fixed.chunk_gated_delta_rule is not None,'FLA import missing'
    for name,fn in methods:
        tensors=[v.detach().clone().requires_grad_() for v in values]; start=time.time()
        out,state=fn(*tensors); (out*probe).sum().backward(); torch.cuda.synchronize()
        results.append((out.detach(),state.detach(),[v.grad.detach() for v in tensors]))
        record(test=name,event='forward_backward',seconds=time.time()-start,passed=True)
    for i,name in [(1,'author_chunk'),(2,'fla_fp32')]:
        tolerance=1e-5 if i==1 else 5e-3
        error(name+'_output',results[0][0],results[i][0],tolerance)
        error(name+'_final_state',results[0][1],results[i][1],tolerance)
        for j,gradient in enumerate(['q','k','v','g','beta']):
            error(name+'_gradient_'+gradient,results[0][2][j],results[i][2][j],tolerance)
    del results,values,tensors; torch.cuda.empty_cache()

    # Instrument the actual model path; preserve exceptions and prohibit fallback.
    calls={'chunk':0,'recurrent':0}
    chunk=fixed.chunk_gated_delta_rule; fused=fixed.fused_recurrent_gated_delta_rule
    def counted_chunk(*args,**kwargs):
        calls['chunk']+=1; return chunk(*args,**kwargs)
    def counted_fused(*args,**kwargs):
        calls['recurrent']+=1; return fused(*args,**kwargs)
    def no_fallback(*args,**kwargs): raise AssertionError('Unexpected slow fallback')
    fixed.chunk_gated_delta_rule=counted_chunk
    fixed.fused_recurrent_gated_delta_rule=counted_fused
    fixed.torch_chunk_gated_delta_rule=no_fallback
    model=fixed.MiniMindForCausalLM(fixed.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True)).cuda()
    count=sum(p.numel() for p in model.parameters()); assert count==205623072,count
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
    ids=torch.randint(3,6400,(1,128),device='cuda')
    torch.cuda.reset_peak_memory_stats(); start=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16): out=model(ids,labels=ids)
    (out.loss+out.aux_loss).backward()
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if 'linear_attn.in_proj_qkv.weight' in n or 'mlp.gate.weight' in n}
    assert norms and all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values()),norms
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    record(test='full_205M_forward_backward_update',parameters=count,ce=float(out.loss),aux=float(out.aux_loss),
           peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.time()-start,kernel_calls=dict(calls),gradient_norms=norms,passed=True)
    del opt,out
    model.eval()
    with torch.no_grad():
        full=model(ids[:,:17]).logits[:,-1]
        prefix=model(ids[:,:16],use_cache=True)
        cached=model(ids[:,16:17],past_key_values=prefix.past_key_values,use_cache=True).logits[:,-1]
    error('full_model_cached_vs_full_logits',full,cached,5e-3)
    assert calls['recurrent']>=6,calls
    record(test='all_tests',kernel_calls=calls,passed=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        record(event='failure',exception=repr(exc),traceback=traceback.format_exc()); raise
