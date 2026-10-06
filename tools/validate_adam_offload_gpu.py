#!/usr/bin/env python3
"""CUDA AdamW vs identical CUDA AdamW with host moment storage; tiny contract."""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.optimizer_offload import AdamMomentOffload

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve prior offload results')
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); torch.set_num_threads(2); torch.manual_seed(43)
    plain=torch.nn.Sequential(torch.nn.Linear(32,48),torch.nn.SiLU(),torch.nn.Linear(48,16)).cuda()
    moved=copy.deepcopy(plain)
    opts=[torch.optim.AdamW(model.parameters(),lr=3e-4,foreach=False) for model in [plain,moved]]
    offload=AdamMomentOffload(opts[1]); x=torch.randn(3,2,32,device='cuda'); y=torch.randn(3,2,16,device='cuda'); checks=[]
    try:
        for step in range(6):
            for index,(model,opt) in enumerate(zip([plain,moved],opts)):
                opt.zero_grad(set_to_none=True)
                for inputs,target in zip(x,y):
                    with torch.autocast('cuda',dtype=torch.bfloat16): loss=(model(inputs)-target).square().mean()/3
                    loss.backward()
                if index: offload.restore()
                opt.step(); opt.zero_grad(set_to_none=True)
                if index: offload.evict()
            checks.append(dict(step=step+1,model_exact=equal(plain.state_dict(),moved.state_dict()),optimizer_exact=equal(opts[0].state_dict(),opts[1].state_dict())))
            if step==2:
                torch.save(dict(model=moved.state_dict(),optimizer=opts[1].state_dict()),out/'offload_resume.pt')
                del offload
                state=torch.load(out/'offload_resume.pt',map_location='cpu',weights_only=False,mmap=True)
                replacement=copy.deepcopy(moved); replacement.load_state_dict(state['model'])
                opts[1]=torch.optim.AdamW(replacement.parameters(),lr=3e-4,foreach=False); opts[1].load_state_dict(state['optimizer'])
                moved=replacement; del state
                offload=AdamMomentOffload(opts[1]); offload.evict()
        result=dict(checks=checks,passed=all(r['model_exact'] and r['optimizer_exact'] for r in checks),host_bytes=offload.host_bytes,
                    scope='Tiny CUDA BF16 forward, FP32 AdamW, three accumulation microbatches, pinned moment evict/restore and file resume; not full-model memory evidence',time=time.time())
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
        assert result['passed'],result
    except Exception as exc:
        import traceback
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),checks=checks,traceback=traceback.format_exc()),indent=2)+'\n'); raise

if __name__=='__main__': main()
