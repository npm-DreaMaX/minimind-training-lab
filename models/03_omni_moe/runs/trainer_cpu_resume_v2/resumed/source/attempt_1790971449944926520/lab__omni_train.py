"""Auditable Omni training: pinned architecture/dataset/loss, bounded storage.

This is our trainer. Augmentation assignment, warmup, held-out validation and
full-precision resume are explicit recipe/system choices, not the upstream CLI.
"""
import argparse,contextlib,fcntl,gc,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from transformers import AutoTokenizer
from lab.train import atomic_json,lr_at
from lab.data import EpochBatchSampler
from lab.omni_batch import official_modules,to_device
from lab.omni_data import ReproducibleOmniDataset,EpochTaggedSampler
from lab.omni_loss import omni_loss
from lab.checkpointing import checkpoint_blocks

ROOT=Path(__file__).resolve().parents[1]
STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)

def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
    if r['cuda'] is not None: torch.cuda.set_rng_state(r['cuda'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-step',type=int,default=0,help='Controlled boundary interruption for resume validation')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); run=ROOT/cfg['run_dir']
    device=cfg.get('device','cuda'); cuda=str(device).startswith('cuda'); preflight=cfg.get('purpose')=='preflight'
    if not cuda and not preflight: raise ValueError('CPU runs are validation only, not the formal Omni budget')
    if not preflight and (cfg.get('max_train_rows') or cfg.get('train_indices_override')):
        raise ValueError('Formal data may not be silently replaced by fixture indices')
    gpu_lock=None
    if cuda:
        gpu_lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.jsonl').exists() and not a.resume: raise RuntimeError('Existing run; preserve evidence or explicitly resume')
    if a.resume and json.loads((run/'config.json').read_text())!=cfg: raise RuntimeError('Resume config mismatch')
    for sub in ['source','evaluation','checkpoints']: (run/sub).mkdir(exist_ok=True)
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    torch.set_num_threads(cfg.get('cpu_threads',4)); random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    if cfg.get('deterministic',False):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; torch.use_deterministic_algorithms(True)
    if cuda and cfg.get('allocator_gib'):
        torch.cuda.set_per_process_memory_fraction(cfg['allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    module,collate=official_modules()
    modality=cfg['modality']; mode=cfg.get('mode','all')
    audio_path=str(ROOT/cfg['audio_encoder']) if modality=='a2a' else '/nonexistent/t2a_or_i2t_no_audio_encoder'
    vision_path=str(ROOT/cfg['vision_encoder']) if modality=='i2t' else None
    model=module.MiniMindOmni(module.OmniConfig(**cfg['model']),audio_encoder_path=audio_path,vision_model_path=vision_path)
    if modality=='a2a' and (model.audio_encoder is None or model.audio_processor is None): raise RuntimeError('A2A encoder was not actually loaded')
    if modality=='i2t' and (model.vision_encoder is None or model.vision_processor is None): raise RuntimeError('I2T encoder was not actually loaded')
    if cfg.get('init_weight') and not a.resume:
        state=torch.load(ROOT/cfg['init_weight'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True); del state
    if mode!='all':
        if mode not in ['audio_proj','vision_proj']: raise ValueError(mode)
        for param in model.parameters(): param.requires_grad_(False)
        for param in getattr(model,mode).parameters(): param.requires_grad_(True)
    model=model.to(device).train()
    for encoder in [model.audio_encoder,model.vision_encoder]:
        if encoder is not None: encoder.to(device).eval()
    if cfg.get('checkpointing',False): checkpoint_blocks(list(model.thinker.layers)+list(model.talker.layers))
    params=[p for p in model.parameters() if p.requires_grad]
    count=sum(p.numel() for p in model.parameters())
    if not preflight and count!=314887938: raise ValueError(f'Unexpected full Omni parameter count: {count}')
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg.get('betas',[.9,.999])),
                                weight_decay=cfg.get('weight_decay',.01),foreach=False)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model'); corpus=ROOT/cfg['corpus']
    corpus_meta=json.loads((corpus/'index.json').read_text()); audit=json.loads((corpus/'audit_v1/metadata.json').read_text())
    if not preflight and corpus_meta['scope']!='full verified corpus': raise ValueError('Formal stage requires a full verified corpus')
    rows=np.load(corpus/'audit_v1/train.npy',mmap_mode='r'); val_rows=np.load(corpus/'audit_v1/val.npy',mmap_mode='r')
    if cfg.get('train_indices_override'): rows=np.asarray(cfg['train_indices_override'],dtype=np.uint64)
    elif cfg.get('max_train_rows'): rows=rows[:cfg['max_train_rows']]
    common=dict(data_path=str(corpus),tokenizer=tokenizer,audio_processor=model.audio_processor,
                vision_processor=model.vision_processor,max_length=cfg['max_length'],image_token_len=model.config.image_token_len,seed=cfg['seed'])
    train=ReproducibleOmniDataset(**common,rows=rows,scheduled_sampling=cfg.get('scheduled_sampling',.05))
    # Stable held-out augmented inputs, with uncorrupted teacher-forced history.
    # Separate clean generation/task evaluation is also required after training.
    val=ReproducibleOmniDataset(**common,rows=val_rows,scheduled_sampling=0.)
    if not len(train) or not len(val): raise ValueError('Need train and held-out rows')
    epoch=cursor=step=text_tokens=audio_tokens=0; best=float('inf')
    total=cfg.get('max_steps') or math.ceil(len(train)/(cfg['batch_size']*cfg['accumulation_steps']))*cfg['epochs']
    tracked=['lab/omni_train.py','lab/omni_loss.py','lab/omni_data.py','lab/omni_table.py','lab/omni_batch.py','lab/checkpointing.py','lab/train.py',
             'lab/data.py','sources/minimind-o/model/model_omni.py','sources/minimind-o/model/model_minimind.py',
             'sources/minimind-o/dataset/omni_dataset.py','sources/minimind-o/trainer/train_sft_omni.py']
    fingerprints={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in tracked}
    latest=run/'checkpoints/latest_resume.pt'
    if a.resume:
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['config']!=cfg or state['fingerprints']!=fingerprints: raise RuntimeError('Resume source/config mismatch')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        epoch,cursor,step,text_tokens,audio_tokens,best=[state[k] for k in ['epoch','cursor','step','text_tokens','audio_tokens','best']]
        restore_rng(state['rng']); del state; gc.collect()
    atomic_json(run/'config.json',cfg)
    attempt=run/'source'/f'attempt_{time.time_ns()}'; attempt.mkdir()
    for path in tracked: shutil.copy2(ROOT/path,attempt/path.replace('/','__'))
    provenance=dict(pid=os.getpid(),parameters=count,trainable_parameters=sum(p.numel() for p in params),
                    fingerprints=fingerprints,torch=torch.__version__,device=device,train_rows=len(train),val_rows=len(val),
                    total_steps=total,loss_reduction='Official per-microbatch text CE plus mean of eight STOP-weighted audio CEs plus aux; accumulated microbatches equally weighted',
                    purpose=cfg.get('purpose','formal'),corpus_audit=audit,time=time.time())
    atomic_json(attempt/'provenance.json',provenance)
    if not (run/'provenance.json').exists(): atomic_json(run/'provenance.json',provenance)
    metrics=(run/'metrics.jsonl').open('a',buffering=1); start_time=time.time(); monitor=telemetry=None
    if cuda:
        torch.cuda.reset_peak_memory_stats(); telemetry=(run/'gpu.csv').open('a')
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv','-l','2'],stdout=telemetry,stderr=subprocess.STDOUT)
    def amp(): return torch.autocast('cuda',dtype=torch.bfloat16) if cuda else contextlib.nullcontext()
    def sync():
        if cuda: torch.cuda.synchronize()
    def forward(batch,diagnostics=False):
        ids,labels,audio,a_inputs,a_lens,pixels,spk=to_device(batch,device)
        with amp():
            out=model(ids,audio_inputs=a_inputs,audio_lens=a_lens,pixel_values=pixels,spk_emb=spk)
            result=omni_loss(out,labels,audio,diagnostics=diagnostics)
        return result
    def export(path):
        temp=Path(str(path)+'.tmp'); torch.save({k:v.detach().half().cpu() for k,v in model.state_dict().items()},temp); temp.replace(path)
    def save(kind):
        sync(); started=time.time(); temp=Path(str(latest)+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),config=cfg,fingerprints=fingerprints,
                        epoch=epoch,cursor=cursor,step=step,text_tokens=text_tokens,audio_tokens=audio_tokens,best=best),temp)
        temp.replace(latest)
        if kind in ['final','paused'] or step%cfg.get('milestone_interval',5000)==0: export(run/'checkpoints'/f'model_step_{step:07d}.pth')
        record=dict(event='checkpoint',kind=kind,step=step,epoch=epoch,cursor=cursor,bytes=latest.stat().st_size,seconds=time.time()-started,time=time.time())
        with (run/'checkpoint_progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    @torch.no_grad()
    def evaluate(final=False):
        state=capture_rng(); model.eval(); n=len(val) if final else min(len(val),cfg.get('validation_samples',128))
        selected=list(range(n)) if final else torch.randperm(len(val),generator=torch.Generator().manual_seed(cfg['seed']+1000003))[:n].tolist()
        loader=DataLoader(Subset(val,selected),batch_size=cfg['batch_size'],collate_fn=collate,num_workers=0,
                          generator=torch.Generator().manual_seed(cfg['seed']+2000003))
        text_sum=0.; text_n=0; audio_sum=np.zeros(8); audio_n=np.zeros(8,dtype=np.int64); stop_n=np.zeros(8,dtype=np.int64); stop_good=np.zeros(8,dtype=np.int64)
        for batch in loader:
            r=forward(batch,True)
            if not torch.isfinite(r['loss']): raise RuntimeError('Nonfinite Omni validation')
            tn=int(r['text_count']); an=r['audio_counts'].cpu().numpy()
            text_sum+=float(r['text_ce'])*tn; text_n+=tn
            audio_sum+=r['audio_codebook_ce'].cpu().numpy()*an; audio_n+=an
            stop_n+=r['stop_counts'].cpu().numpy(); stop_good+=r['stop_correct'].cpu().numpy()
        if not text_n and not audio_n.sum(): raise RuntimeError('No held-out supervised labels')
        text_ce=text_sum/max(1,text_n); audio_ce=audio_sum/np.maximum(1,audio_n)
        record=dict(step=step,final=final,validation_rows=n,text_ce=text_ce,audio_codebook_ce=audio_ce.tolist(),
                    audio_ce=float(audio_ce.mean()),selection_score=text_ce+float(audio_ce.mean()),text_labels=text_n,audio_labels=audio_n.tolist(),
                    stop_count=stop_n.tolist(),stop_correct=stop_good.tolist(),time=time.time(),
                    protocol='fixed per-record augmented inputs; scheduled_sampling=0; CE globally token-weighted within each head; no aux in selection score')
        atomic_json(run/'evaluation'/f'step_{step:07d}_{"full" if final else "subset"}.json',record)
        model.train(); restore_rng(state); metrics.write(json.dumps({'event':'evaluation',**record})+'\n'); return record
    def select(record):
        nonlocal best
        if record['selection_score']<best:
            best=record['selection_score']; export(run/'checkpoints/best_validation.pth'); atomic_json(run/'checkpoints/best_validation.json',record)
    try:
        if not a.resume: select(evaluate())
        while epoch<cfg['epochs'] and step<total and not STOP:
            sampler=EpochTaggedSampler(EpochBatchSampler(len(train),cfg['batch_size'],cfg['seed'],epoch,cursor),epoch)
            workers=cfg.get('num_workers',2)
            loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate,num_workers=workers,pin_memory=cuda,
                              generator=torch.Generator().manual_seed(cfg['seed']+epoch+2000003),
                              **({'prefetch_factor':2,'timeout':120} if workers else {}))
            it=iter(loader)
            while cursor<len(train) and step<total and not STOP:
                started=time.perf_counter(); batches=[]; consumed=0
                for _ in range(cfg['accumulation_steps']):
                    try: batch=next(it)
                    except StopIteration: break
                    consumed+=len(batch[0])
                    if (batch[1]!=-100).any() or (batch[2]!=-100).any(): batches.append(batch)
                if not consumed: break
                if not batches:
                    cursor+=consumed; metrics.write(json.dumps({'event':'zero_label_group','epoch':epoch,'cursor':cursor})+'\n'); continue
                lr=lr_at(step,total,cfg.get('warmup_steps',100),cfg['learning_rate'])
                for group in optimizer.param_groups: group['lr']=lr
                optimizer.zero_grad(set_to_none=True); values=np.zeros(4); nt=na=0; per_head=np.zeros(8)
                for batch in batches:
                    r=forward(batch)
                    if not torch.isfinite(r['loss']): raise RuntimeError('Nonfinite Omni training loss')
                    (r['loss']/len(batches)).backward()
                    values+=np.asarray([float(r[k].detach()) for k in ['loss','text_ce','audio_ce','aux_loss']])/len(batches)
                    per_head+=r['audio_codebook_ce'].detach().cpu().numpy()/len(batches)
                    nt+=int(r['text_count']); na+=int(r['audio_counts'].sum()); del r
                grad=float(torch.nn.utils.clip_grad_norm_(params,cfg.get('grad_clip',1.),error_if_nonfinite=True))
                optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()
                step+=1; cursor+=consumed; text_tokens+=nt; audio_tokens+=na; seconds=time.perf_counter()-started
                record=dict(event='train',step=step,epoch=epoch,cursor=cursor,loss=values[0],text_ce=values[1],audio_ce=values[2],aux_loss=values[3],
                            audio_codebook_ce=per_head.tolist(),grad_norm=grad,learning_rate=lr,text_labels=nt,audio_labels=na,
                            total_text_labels=text_tokens,total_audio_labels=audio_tokens,step_seconds=seconds,samples_per_second=consumed/seconds,
                            wall_seconds=time.time()-start_time,time=time.time())
                if cuda: record.update(cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),cuda_peak_allocated=torch.cuda.max_memory_allocated())
                metrics.write(json.dumps(record)+'\n'); atomic_json(run/'status.json',{'status':'training',**record})
                if step<=3 or step%cfg.get('log_interval',10)==0: print(json.dumps(record),flush=True)
                if step%cfg.get('eval_interval',500)==0: select(evaluate())
                if step%cfg.get('save_interval',500)==0 or step in cfg.get('early_checkpoint_steps',[10,100]): save('periodic')
                if a.stop_after_step and step>=a.stop_after_step: request_stop()
            if cursor>=len(train): epoch+=1; cursor=0
        if STOP:
            save('paused'); atomic_json(run/'status.json',dict(status='paused',step=step,text_tokens=text_tokens,audio_tokens=audio_tokens))
        else:
            select(evaluate()); final=evaluate(True); save('final')
            atomic_json(run/'status.json',dict(status='complete',step=step,text_tokens=text_tokens,audio_tokens=audio_tokens,validation=final,purpose=cfg.get('purpose','formal')))
    except Exception as exc:
        atomic_json(run/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),step=step,epoch=epoch,cursor=cursor,time=time.time()))
        atomic_json(run/'status.json',dict(status='failed',step=step,error=repr(exc))); raise
    finally:
        metrics.close()
        if monitor:
            monitor.terminate(); monitor.wait(timeout=5); telemetry.close()

if __name__=='__main__': main()
