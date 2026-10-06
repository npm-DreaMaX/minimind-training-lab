#!/usr/bin/env python3
"""Bounded T2A memory test of the complete official Omni architecture.

No encoder is called for a T2A record. This test cannot establish A2A/I2T
feasibility, learned quality, or multimodal completion. The allocator cap leaves
space for the observed Windows desktop instead of provoking OS-wide paging.
"""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sources/minimind-o')]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seq-len',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--steps',type=int,default=3)
    p.add_argument('--max-process-gib',type=float,default=5.3); p.add_argument('--out',required=True)
    p.add_argument('--checkpointing',action='store_true')
    a=p.parse_args(); dest=ROOT/a.out; dest.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.manual_seed(20261003)
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    record={'args':vars(a),'scope':'Synthetic T2A, full model, no input encoders used; explicit allocator cap',
            'status':'started','pid':os.getpid(),'torch':torch.__version__,'steps':[]}
    stream=(dest/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv','-l','1'],stdout=stream)
    try:
        from model.model_omni import MiniMindOmni,OmniConfig
        torch.cuda.init(); record['gpu']=torch.cuda.get_device_name(0)
        if a.max_process_gib:
            cap=int(a.max_process_gib*1024**3)
            torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
            record['allocator_cap_bytes']=cap
        model=MiniMindOmni(OmniConfig(hidden_size=768,num_hidden_layers=8,use_moe=True),
                           audio_encoder_path='/nonexistent/t2a_does_not_use_audio_encoder',
                           vision_model_path=None).cuda().train()
        if a.checkpointing:
            from lab.checkpointing import checkpoint_blocks
            checkpoint_blocks(list(model.thinker.layers)+list(model.talker.layers))
        record['parameters']=sum(p.numel() for p in model.parameters())
        record['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
        record['theoretical_fp32_parameter_gradient_adam_bytes']=record['parameters']*16
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
        b,t=a.batch_size,a.seq_len
        ids=torch.cat([torch.randint(0,2048,(b,8,t),device='cuda'),torch.randint(3,6400,(b,1,t),device='cuda')],dim=1)
        audio_labels=torch.randint(0,2048,(b,8,t),device='cuda'); audio_labels[:,:,-1]=2050
        text_labels=torch.randint(3,6400,(b,t),device='cuda')
        spk=torch.randn(b,192,device='cuda'); ids[:,:,0]=2051
        torch.cuda.reset_peak_memory_stats()
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); start=time.time()
            phases={'before_forward':memory()}
            record['active_step']=step+1; record['active_phases']=phases
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(ids,spk_emb=spk)
                text_loss=F.cross_entropy(result.logits.flatten(0,1),text_labels.flatten())
                audio_loss=0.
                for i,logits in enumerate(result.audio_logits):
                    labels=audio_labels[:,i].flatten()
                    audio_loss=audio_loss+(F.cross_entropy(logits.flatten(0,1),labels,reduction='none')*(1+(labels==2050)*9)).mean()/8
                loss=text_loss+audio_loss+result.aux_loss
            phases['after_forward']=memory(); loss.backward(); phases['after_backward']=memory()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize(); phases['after_optimizer']=memory()
            row=dict(step=step+1,loss=float(loss.detach()),text_ce=float(text_loss.detach()),audio_ce=float(audio_loss.detach()),
                     grad_norm=float(grad),seconds=time.time()-start,memory=phases)
            record['steps'].append(row); print(json.dumps(row),flush=True)
            del result,loss,text_loss,audio_loss
        record['status']='passed'
    except Exception as exc:
        record.update(status='failed',exception=repr(exc),traceback=traceback.format_exc(),memory=memory())
        raise
    finally:
        (dest/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        monitor.terminate(); monitor.wait(timeout=5); stream.close()

if __name__=='__main__': main()
