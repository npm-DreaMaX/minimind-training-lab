#!/usr/bin/env python3
"""Profile one real accumulation/update cycle of the complete configured AR model.

The live formal process must release the exclusive GPU slot first. This creates
a separate short-lived model; its updates are not part of the formal token budget.
"""
import argparse,fcntl,gc,hashlib,importlib.util,json,random,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from lab.data import TokenCorpus

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--weight',required=True); p.add_argument('--out',required=True)
    p.add_argument('--allocator-gib',type=float,default=5.3)
    a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve existing profiling evidence')
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); cfg=json.loads((ROOT/a.config).read_text())
    torch.set_num_threads(4); torch.manual_seed(20261007); torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.allocator_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model_path=ROOT/cfg['model_file']; spec=importlib.util.spec_from_file_location('profiled_model',model_path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model=module.MiniMindForCausalLM(module.MiniMindConfig(**cfg['model'])).cuda().train()
    weight_hash=hashlib.sha256()
    with (ROOT/a.weight).open('rb') as stream:
        while chunk:=stream.read(8*1024**2): weight_hash.update(chunk)
    state=torch.load(ROOT/a.weight,map_location='cpu',weights_only=True,mmap=True); model.load_state_dict(state,strict=True); del state; gc.collect()
    if cfg.get('checkpointing'):
        from lab.checkpointing import checkpoint_blocks
        checkpoint_blocks(model.model.layers)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],foreach=False)
    corpus=TokenCorpus(ROOT/cfg['corpus'],'train',cfg['seq_len'],cfg['stage'])
    n=cfg['batch_size']*cfg['accumulation_steps']; indices=random.Random(20261007).sample(range(len(corpus)),n)
    batches=[]
    for start in range(0,n,cfg['batch_size']):
        values=[corpus[i] for i in indices[start:start+cfg['batch_size']]]
        x=torch.stack([v[0] for v in values]).pin_memory(); y=torch.stack([v[1] for v in values]).pin_memory()
        count=int((y[:,1:]!=-100).sum()); batches.append((x,y,count))
    total=sum(v[2] for v in batches); times=[]
    def update():
        optimizer.zero_grad(set_to_none=True)
        for x,y,count in batches:
            with torch.profiler.record_function('data_to_cuda'):
                x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            with torch.profiler.record_function('microbatch_forward'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(x,labels=y); loss=result.loss*(count/total)+result.aux_loss/len(batches)
            with torch.profiler.record_function('microbatch_backward'): loss.backward()
            del result,loss,x,y
        with torch.profiler.record_function('clip_and_optimizer'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    try:
        for _ in range(2):
            start=time.perf_counter(); update(); times.append(time.perf_counter()-start)
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False,profile_memory=True,with_stack=False) as profile:
            update()
        profile.export_chrome_trace(str(out/'trace.json'))
        rows=[]
        for event in profile.key_averages():
            rows.append(dict(operator=event.key,calls=event.count,self_cpu_us=event.self_cpu_time_total,
                             self_device_us=event.self_device_time_total,total_device_us=event.device_time_total,
                             cpu_memory_bytes=event.cpu_memory_usage,device_memory_bytes=event.device_memory_usage))
        rows.sort(key=lambda r:r['self_device_us'],reverse=True)
        report=dict(scope='Separate full-model short profile on real samples; not a formal training continuation',config=cfg,
                    weight=a.weight,weight_sha256=weight_hash.hexdigest(),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),indices=indices,
                    valid_tokens_per_update=total,unprofiled_update_seconds=times,
                    second_unprofiled_valid_tokens_per_second=total/times[-1],
                    profile_peak_allocated=torch.cuda.max_memory_allocated(),operators=rows,
                    caveats='First update initializes optimizer states. Profiled timings include profiler overhead. GPU utilization is not a FLOP utilization estimate. Accumulation amortizes optimizer cost.',time=time.time())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'operators.txt').write_text(profile.key_averages().table(sort_by='self_device_time_total',row_limit=40)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['operators','indices','config']}),flush=True)
    except Exception as exc:
        import traceback
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n'); raise

if __name__=='__main__': main()
