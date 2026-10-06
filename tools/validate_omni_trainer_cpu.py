#!/usr/bin/env python3
"""Actual-trainer resume contract: small CPU by default, full315M GPU opt-in."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--full-gpu',action='store_true',help='Full315M real-data GPU resume check, only in an exclusive GPU window')
    p.add_argument('--offload',action='store_true',help='Opt-in moment offload; full GPU validation only')
    args=p.parse_args()
    if args.offload and not args.full_gpu: raise ValueError('Moment offload executes optimizer math on CUDA')
    OUT=ROOT/args.out
    if OUT.exists(): raise RuntimeError('Keep previous validation evidence; choose a versioned path')
    OUT.mkdir(parents=True)
    base=dict(purpose='preflight',device='cpu',model=dict(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,
              num_talker_hidden_layers=2,use_moe=True,dropout=.1),modality='t2a',mode='all',
              corpus='artifacts/omni_validation/t2a_shards_v1',train_indices_override=[0,7,8,9,10,12,32,44],
              max_length=128,batch_size=1,accumulation_steps=2,epochs=2,max_steps=5,learning_rate=1e-4,warmup_steps=1,
              seed=20261006,cpu_threads=2,num_workers=2,deterministic=True,checkpointing=True,
              eval_interval=2,save_interval=2,validation_samples=3,log_interval=1)
    if args.full_gpu:
        base.update(device='cuda',model=dict(hidden_size=768,num_hidden_layers=8,use_moe=True,dropout=.1),
                    max_length=1536,allocator_gib=5.3,
                    init_weight='models/03_omni_moe/runs/transfer_preflight_early_ar/initialized_fp32.pth')
        if args.offload: base.update(optimizer_state_offload=True,memory_debug=True)
    configs={}
    for name in ['continuous','resumed']:
        config={**base,'run_dir':str((OUT/name).relative_to(ROOT))}
        path=OUT/f'{name}.json'; path.write_text(json.dumps(config,indent=2)+'\n'); configs[name]=path
    commands=[('continuous',[]),('resumed',['--stop-after-step','1']),('resumed',['--resume'])]
    for i,(name,flags) in enumerate(commands):
        cmd=[sys.executable,'-u','-B','-m','lab.omni_train','--config',str(configs[name]),*flags]
        print(json.dumps({'command':cmd,'time':time.time()}),flush=True)
        with (OUT/f'command_{i}.log').open('w') as log: subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    states=[torch.load(OUT/name/'checkpoints/latest_resume.pt',map_location='cpu',weights_only=False,mmap=True) for name in ['continuous','resumed']]
    checks={key:equal(states[0][key],states[1][key]) for key in ['model','optimizer','epoch','cursor','step','text_tokens','audio_tokens','best']}
    checks['torch_rng']=torch.equal(states[0]['rng']['torch'],states[1]['rng']['torch'])
    if args.full_gpu:
        checks['cuda_rng']=torch.equal(states[0]['rng']['cuda'],states[1]['rng']['cuda'])
        for name in ['continuous','resumed']:
            checks[name+'_full_parameter_count']=json.loads((OUT/name/'provenance.json').read_text())['parameters']==314887938
    scope=('Full315M GPU actual trainer, real T2A fixture, dropout .1, activation recomputation, two workers and boundary resume; not formal learned quality'
           if args.full_gpu else 'Small CPU actual trainer with real T2A data, dropout, activation recomputation, two workers and boundary resume; full315M GPU preflight still required')
    result={'scope':scope,
            'checks':checks,'passed':all(checks.values()),'time':time.time()}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    assert result['passed'],result

if __name__=='__main__': main()
