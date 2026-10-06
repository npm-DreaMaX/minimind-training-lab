#!/usr/bin/env python3
"""Compare an opt-in BF16 projection recipe against the unchanged FP32 baseline.

Run only while holding the project's exclusive GPU slot. These are numerical
checks; a pass does not establish full-model speed, convergence or quality.
"""
import argparse,copy,fcntl,hashlib,importlib.util,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Keep prior precision evidence')
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); torch.set_num_threads(4); torch.manual_seed(20261007); torch.backends.cuda.matmul.allow_tf32=False
    path=ROOT/'models/02_hybrid_moe/src/model_hybrid.py'
    spec=importlib.util.spec_from_file_location('hybrid_precision_gpu',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ref=module.GatedDeltaNet(module.MiniMindConfig(hidden_size=768,require_fla=True),0).cuda()
    mixed=copy.deepcopy(ref); mixed.linear_precision='bfloat16'
    x=torch.randn(2,257,768,device='cuda'); probe=torch.randn_like(x); records=[]
    # Preset pilot gates, not claims that BF16 is mathematically identical.
    def compare(name,a,b,tolerance):
        af,bf=a.detach().float(),b.detach().float()
        error=float((af-bf).square().mean().sqrt()/af.square().mean().sqrt().clamp_min(1e-12))
        row=dict(test=name,relative_rms=error,max_absolute=float((af-bf).abs().max()),tolerance=tolerance,
                 passed=bool(torch.isfinite(bf).all()) and error<tolerance)
        records.append(row); print(json.dumps(row),flush=True)
    try:
        outputs=[]; inputs=[]
        for name,layer in [('fp32',ref),('bf16_projection',mixed)]:
            inp=x.detach().clone().requires_grad_(); start=time.time()
            y,_=layer(inp); (y*probe).mean().backward(); torch.cuda.synchronize()
            outputs.append(y.detach()); inputs.append(inp.grad.detach())
            records.append(dict(event='forward_backward',variant=name,seconds=time.time()-start,
                                parameters=sum(p.numel() for p in layer.parameters())))
        compare('output',outputs[0],outputs[1],.02); compare('input_gradient',inputs[0],inputs[1],.03)
        for (name,p),(_,q) in zip(ref.named_parameters(),mixed.named_parameters()): compare('gradient:'+name,p.grad,q.grad,.03)
        with torch.no_grad():
            sample=x[:1,:17]
            full,(parallel_conv,parallel_state)=mixed(sample,use_cache=True)
            conv=state=None; slices=[]
            for token in sample.split(1,dim=1):
                y,(conv,state)=mixed(token,conv_state=conv,recurrent_state=state,use_cache=True); slices.append(y)
            compare('mixed_cached_vs_parallel',full,torch.cat(slices,dim=1),.02)
            compare('mixed_final_state',parallel_state,state,.02)
            records.append(dict(event='cache_dtypes',convolution=str(conv.dtype),recurrent=str(state.dtype)))
            assert state.dtype==torch.float32,'State accumulation must remain FP32'
        assert all(r.get('passed',True) for r in records),'Preset numerical gate failed; inspect evidence before adopting the recipe'
    finally:
        (out/'results.json').write_text(json.dumps(dict(records=records,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='One full-width Gated DeltaNet layer, numerical pilot only',time=time.time()),indent=2)+'\n')

if __name__=='__main__': main()
