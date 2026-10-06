"""Single-GPU, auditable AR training with token-weighted accumulation and exact state capture.

The model implementation is imported unchanged from the configured source file.
This is our training harness, not the upstream CLI. Resume occurs at optimizer
boundaries, with the shuffled epoch sample cursor and scheduler position saved.
"""
import argparse,contextlib,fcntl,gc,hashlib,importlib.util,json,math,os,random,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from transformers import AutoTokenizer
from lab.data import TokenCorpus,EpochBatchSampler

ROOT=Path(__file__).resolve().parents[1]
STOP=False
def stop_handler(*_):
    global STOP
    STOP=True

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)

def lr_at(step,total,warmup,peak):
    if step<warmup: return peak*(step+1)/max(1,warmup)
    return peak*(0.1+0.45*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup)))))

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())

def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch']); torch.cuda.set_rng_state(r['cuda'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    if cfg.get('triton_f32_default'):
        if cfg['triton_f32_default'] not in ['ieee','tf32','tf32x3']: raise ValueError('Unsupported Triton dot precision')
        # Independent of PyTorch matmul flags; set before importing FLA/JIT kernels.
        os.environ['TRITON_F32_DEFAULT']=cfg['triton_f32_default']
    run=ROOT/cfg['run_dir']; run.mkdir(parents=True,exist_ok=True)
    (ROOT/'runs').mkdir(exist_ok=True)
    gpu_lock=(ROOT/'runs/local_gpu.lock').open('a+')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock.seek(0); gpu_lock.truncate()
    gpu_lock.write(str(os.getpid())); gpu_lock.flush()
    if (run/'metrics.jsonl').exists() and not a.resume: raise RuntimeError('Existing run requires --resume; do not overwrite evidence')
    for sub in ['checkpoints','evaluation','source']: (run/sub).mkdir(exist_ok=True)
    if a.resume and (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=cfg:
        raise RuntimeError('Resume config mismatch; existing evidence was not overwritten')
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGINT,stop_handler)
    torch.set_num_threads(cfg.get('cpu_threads',4)); random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    # Opt-in only after independent output/gradient and full-model checks.
    # Default preserves all earlier recipes; FLA has a separate precision flag.
    torch.backends.cuda.matmul.allow_tf32=cfg.get('torch_allow_tf32',False)
    if cfg.get('deterministic',False):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; torch.use_deterministic_algorithms(True)
    if cfg.get('allocator_gib'):
        torch.cuda.set_per_process_memory_fraction(cfg['allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model_file=ROOT/cfg['model_file']
    source_sha=hashlib.sha256(model_file.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('training_model',model_file); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model_config=module.MiniMindConfig(**cfg['model'])
    model=module.MiniMindForCausalLM(model_config).cuda().train()
    if cfg.get('init_weight') and not a.resume:
        weights=torch.load(ROOT/cfg['init_weight'],map_location='cpu',weights_only=False,mmap=True)
        weights=weights.get('model',weights)
        incompatible=model.load_state_dict(weights,strict=cfg.get('strict_init',True))
        atomic_json(run/'initialization.json',{'missing':incompatible.missing_keys,'unexpected':incompatible.unexpected_keys})
        del weights; gc.collect()
    if cfg.get('train_only_linear',False):
        for name,param in model.named_parameters(): param.requires_grad_('linear_attn.' in name)
    if cfg.get('checkpointing',False):
        from lab.checkpointing import checkpoint_blocks
        checkpoint_blocks(model.model.layers)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg.get('betas',[0.9,0.999])),
                         weight_decay=cfg.get('weight_decay',0.01),foreach=False)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    train=TokenCorpus(ROOT/cfg['corpus'],'train',cfg['seq_len'],cfg['stage'])
    val=TokenCorpus(ROOT/cfg['corpus'],'val',cfg['seq_len'],cfg['stage'])
    if not len(train) or not len(val): raise RuntimeError('Need nonempty training and held-out splits')
    effective=cfg['batch_size']*cfg['accumulation_steps']
    total_steps=cfg.get('max_steps') or math.ceil(len(train)/effective)*cfg['epochs']
    warmup=cfg.get('warmup_steps',100)
    epoch=cursor=step=trained_tokens=0; best=float('inf'); history=[]
    latest=run/'checkpoints/latest_resume.pt'
    if a.resume:
        saved=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
        if saved['source_sha256']!=source_sha or saved['config']!=cfg: raise RuntimeError('Resume source/config mismatch')
        model.load_state_dict(saved['model'],strict=True); opt.load_state_dict(saved['optimizer'])
        epoch,cursor,step,trained_tokens,best=[saved[k] for k in ['epoch','cursor','step','trained_tokens','best_val']]
        restore_rng(saved['rng']); del saved; gc.collect()
    atomic_json(run/'config.json',cfg)
    attempt_dir=run/'source'/f'attempt_{int(time.time())}'
    attempt_dir.mkdir()
    shutil.copy2(model_file,attempt_dir/'model.py')
    shutil.copy2(Path(__file__),attempt_dir/'train.py')
    shutil.copy2(ROOT/'lab/data.py',attempt_dir/'data.py')
    if cfg.get('checkpointing',False): shutil.copy2(ROOT/'lab/checkpointing.py',attempt_dir/'checkpointing.py')
    provenance={'source_sha256':source_sha,'parameter_count':sum(p.numel() for p in model.parameters()),
                'trainable_parameters':sum(p.numel() for p in params),'torch':torch.__version__,'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(0),'train_rows':len(train),'validation_rows':len(val),
                'triton_f32_default':os.environ.get('TRITON_F32_DEFAULT','unset (backend default)'),
                'torch_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'filtered_zero_target_train_rows':train.original_rows-len(train),
                'total_steps':total_steps,'effective_batch_samples':effective,'started_unix':time.time(),'pid':os.getpid(),
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'loss_reduction':'CE weighted by valid labels over accumulation group; auxiliary loss averaged over nonempty microbatches'}
    atomic_json(attempt_dir/'provenance.json',provenance)
    if not (run/'provenance.json').exists(): atomic_json(run/'provenance.json',provenance)
    telemetry=(run/'gpu.csv').open('a')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit','--format=csv','-l','2'],stdout=telemetry,stderr=subprocess.STDOUT)
    metrics=(run/'metrics.jsonl').open('a',buffering=1)
    torch.cuda.reset_peak_memory_stats(); training_start=time.time(); last_save=step
    router_stats={}; router_mask=None; collect_router=False
    def router_hook(layer):
        def hook(_module,_inputs,output):
            if not collect_router or layer in router_stats: return
            with torch.no_grad():
                probs=output.detach().float().softmax(-1)
                if router_mask is not None: probs=probs[router_mask]
                if not len(probs): return
                counts=torch.bincount(probs.argmax(-1),minlength=probs.shape[-1])
                router_stats[layer]={'tokens_per_expert':counts.cpu().tolist(),
                                     'mean_probability_entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean()),
                                     'max_expert_share':float(counts.max()/counts.sum())}
        return hook
    for i,layer in enumerate(model.model.layers):
        if hasattr(layer.mlp,'gate'): layer.mlp.gate.register_forward_hook(router_hook(i))

    def export_weights(destination):
        temporary=destination.with_suffix('.tmp')
        torch.save({k:v.detach().half().cpu() for k,v in model.state_dict().items()},temporary)
        temporary.replace(destination)

    def update_best(record):
        nonlocal best
        if record['validation_ce']<best:
            best=record['validation_ce']
            export_weights(run/'checkpoints/best_validation.pth')
            atomic_json(run/'checkpoints/best_validation.json',record)

    @torch.no_grad()
    def evaluate(final=False):
        rng=capture_rng(); model.eval(); total_loss=0.; n=0
        count=len(val) if final else min(len(val),cfg.get('validation_samples',512))
        selection=(list(range(count)) if final else torch.randperm(len(val),generator=torch.Generator().manual_seed(cfg.get('validation_seed',cfg['seed']+1000003)))[:count].tolist())
        loader=DataLoader(Subset(val,selection),batch_size=cfg['batch_size'],num_workers=0)
        for x,y in loader:
            k=int((y[:,1:]!=-100).sum())
            if not k: continue
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(x.cuda(),labels=y.cuda())
            if not torch.isfinite(out.loss): raise RuntimeError('Nonfinite validation loss')
            total_loss+=out.loss.item()*k; n+=k
        if n==0: raise RuntimeError('No supervised validation tokens')
        examples=[]
        prompts=['人工智能的主要应用包括','中国的首都是','水的三种状态是'] if cfg['stage']=='pretrain' else ['你好，请介绍一下自己。','为什么天空是蓝色的？','计算 17 加 25。']
        for prompt in prompts:
            rendered=tokenizer.bos_token+prompt if cfg['stage']=='pretrain' else tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            ids=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.generate(ids,max_new_tokens=64,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=tokenizer.eos_token_id)
            generated=output[0,ids.shape[1]:]
            examples.append({'prompt':prompt,'rendered_input':rendered,'input_ids':ids[0].tolist(),
                             'continuation':tokenizer.decode(generated,skip_special_tokens=False),
                             'output_tokens':len(generated),'ended_with_eos':bool(len(generated) and int(generated[-1])==tokenizer.eos_token_id),
                             'truncated_by_budget':bool(len(generated)==64 and int(generated[-1])!=tokenizer.eos_token_id)})
        record={'step':step,'epoch':epoch,'trained_tokens':trained_tokens,'validation_ce':total_loss/n,'validation_tokens':n,'validation_rows':count,'final':final,'generation':examples,'time':time.time(),
                'validation_selection_sha256':hashlib.sha256(np.asarray(selection,dtype='<u8').tobytes()).hexdigest(),
                'validation_selection':'full split' if final else 'fixed seeded sample across the full held-out split',
                'generation_protocol':{'version':2,'pretrain_prefix':'explicit tokenizer BOS','sft_prefix':'official chat template',
                                       'dtype':'BF16 autocast','do_sample':False,'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.,'max_new_tokens':64}}
        atomic_json(run/'evaluation'/f'step_{step:07d}.json',record)
        model.train(); restore_rng(rng); return record

    def save(kind='periodic'):
        torch.cuda.synchronize(); start=time.time()
        payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':capture_rng(),'epoch':epoch,'cursor':cursor,
                 'step':step,'trained_tokens':trained_tokens,'best_val':best,'config':cfg,'source_sha256':source_sha}
        tmp=latest.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(latest); del payload
        if kind in ['final','paused'] or step%cfg.get('milestone_interval',5000)==0:
            dest=run/'checkpoints'/f'model_step_{step:07d}.pth'
            export_weights(dest)
        record={'event':'checkpoint','kind':kind,'step':step,'epoch':epoch,'cursor':cursor,'tokens':trained_tokens,'bytes':latest.stat().st_size,'seconds':time.time()-start,'time':time.time()}
        with (run/'checkpoint_progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if cfg.get('plot_on_save',False):
            with (run/'plot.log').open('a') as plot_log:
                plotted=subprocess.run([sys.executable,str(ROOT/'tools/plot_run.py'),str(run)],stdout=plot_log,stderr=subprocess.STDOUT)
            if plotted.returncode: print(json.dumps({'event':'plot_failed','returncode':plotted.returncode}),flush=True)

    try:
        initial=evaluate(); metrics.write(json.dumps({'event':'evaluation',**initial},ensure_ascii=False)+'\n'); update_best(initial)
        while epoch<cfg['epochs'] and step<total_steps and not STOP:
            sampler=EpochBatchSampler(len(train),cfg['batch_size'],cfg['seed'],epoch,cursor)
            loader=DataLoader(train,batch_sampler=sampler,num_workers=cfg.get('num_workers',2),pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg['seed']+epoch+2000003),
                              **({'prefetch_factor':2,'timeout':60} if cfg.get('num_workers',2) else {}))
            it=iter(loader)
            while cursor<len(train) and step<total_steps and not STOP:
                start=time.perf_counter(); batches=[]; consumed=0
                for _ in range(cfg['accumulation_steps']):
                    try: x,y=next(it)
                    except StopIteration: break
                    consumed+=len(x); n=int((y[:,1:]!=-100).sum())
                    if n: batches.append((x,y,n))
                if not consumed: break
                if not batches:
                    cursor+=consumed
                    metrics.write(json.dumps({'event':'zero_label_group','epoch':epoch,'cursor':cursor})+'\n'); continue
                total_valid=sum(b[2] for b in batches); nominal=sum(b[0].numel() for b in batches)
                nonpad_input=sum(int((b[0]!=tokenizer.pad_token_id).sum()) for b in batches)
                lr=lr_at(step,total_steps,warmup,cfg['learning_rate'])
                for g in opt.param_groups: g['lr']=lr
                opt.zero_grad(set_to_none=True); ce=aux=0.
                router_stats.clear(); collect_router=(step%cfg.get('router_log_interval',50)==0)
                for x,y,n in batches:
                    x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
                    router_mask=(x.reshape(-1)!=tokenizer.pad_token_id) if collect_router else None
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=model(x,labels=y)
                        loss=out.loss*(n/total_valid)+out.aux_loss/len(batches)
                    if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch={epoch} cursor={cursor}')
                    loss.backward(); ce+=out.loss.item()*n/total_valid; aux+=out.aux_loss.item()/len(batches)
                    del out,loss,x,y
                collect_router=False; router_mask=None
                grad=torch.nn.utils.clip_grad_norm_(params,cfg.get('grad_clip',1.0),error_if_nonfinite=True).item()
                opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
                step+=1; cursor+=consumed; trained_tokens+=total_valid
                seconds=time.perf_counter()-start
                record={'event':'train','step':step,'epoch':epoch,'cursor':cursor,'loss':ce+aux,'ce_loss':ce,'aux_loss':aux,
                        'grad_norm':grad,'learning_rate':lr,'valid_tokens':total_valid,'tokens_trained':trained_tokens,
                        'nominal_tokens':nominal,'nonpad_input_tokens':nonpad_input,
                        'input_padding_fraction':1-nonpad_input/nominal,'unsupervised_fraction':1-total_valid/nominal,'step_seconds':seconds,
                        'valid_tokens_per_second':total_valid/seconds,'nominal_tokens_per_second':nominal/seconds,
                        'cuda_allocated':torch.cuda.memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),
                        'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'wall_seconds':time.time()-training_start,'time':time.time()}
                if router_stats: record['router_first_microbatch_nonpadding']=dict(router_stats)
                metrics.write(json.dumps(record)+'\n'); atomic_json(run/'status.json',{'status':'training',**record})
                if step%cfg.get('log_interval',10)==0 or step<=3: print(json.dumps(record),flush=True)
                if step%cfg.get('eval_interval',500)==0:
                    evaluation=evaluate(); metrics.write(json.dumps({'event':'evaluation',**evaluation},ensure_ascii=False)+'\n')
                    update_best(evaluation); print(json.dumps({'event':'evaluation',**evaluation},ensure_ascii=False),flush=True)
                if step%cfg.get('save_interval',500)==0 or step in cfg.get('early_checkpoint_steps',[]): save(); last_save=step
            if cursor>=len(train): epoch+=1; cursor=0
        if STOP:
            save('paused'); atomic_json(run/'status.json',{'status':'paused','step':step,'trained_tokens':trained_tokens})
        else:
            # Compare checkpoints on the same fixed subset; full-split CE is a
            # separate final report and must not silently change model selection.
            selection=evaluate(); metrics.write(json.dumps({'event':'evaluation',**selection},ensure_ascii=False)+'\n'); update_best(selection)
            final=evaluate(final=True); metrics.write(json.dumps({'event':'evaluation',**final},ensure_ascii=False)+'\n')
            save('final'); atomic_json(run/'status.json',{'status':'complete','step':step,'trained_tokens':trained_tokens,'validation':final})
    except Exception as e:
        import traceback
        atomic_json(run/'failure.json',{'exception':repr(e),'traceback':traceback.format_exc(),'step':step,'epoch':epoch,'cursor':cursor,'last_saved_step':last_save,'time':time.time()})
        atomic_json(run/'status.json',{'status':'failed','step':step,'error':repr(e)})
        raise
    finally:
        metrics.close(); monitor.terminate()
        try: monitor.wait(timeout=3)
        except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
        telemetry.close()

if __name__=='__main__': main()
