#!/usr/bin/env python3
"""Bounded full-Omni tests of real audio/vision inputs on an idle remote GPU."""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from server_gpu_guard import wait_for_idle


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--out',required=True)
    p.add_argument('--only',help='Comma-separated pending checks; completed checks are not repeated');a=p.parse_args()
    if not str(ROOT).startswith('/new_data/REMOTE_USER/minimind/'):raise RuntimeError('Authorized remote workspace only')
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    shutil.copy2(ROOT/'tools/server_gpu_guard.py',out/'gpu_guard_source.py')
    manifest=json.loads((ROOT/'reports/server_omni_inputs_bundle_v1.json').read_text())
    for name,info in manifest['files'].items():
        path=ROOT/name
        assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
    (out/'verified_assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    common=dict(purpose='preflight',device='cuda',model=dict(hidden_size=768,num_hidden_layers=8,use_moe=True),
                epochs=2,learning_rate=1e-5,warmup_steps=1,seed=20261006,cpu_threads=4,num_workers=2,
                prefetch_factor=1,deterministic=True,checkpointing=True,allocator_gib=13.5,
                optimizer_state_offload=False,eval_interval=50,save_interval=50,validation_samples=8,
                log_interval=1,inspect_first_batches=2)
    tasks=[
        ('t2a_B16_recompute',dict(modality='t2a',mode='all',corpus='artifacts/omni_validation/t2a_shards_v1',max_length=1536,batch_size=16,accumulation_steps=1,max_steps=12)),
        ('a2a_all_long',dict(modality='a2a',mode='all',corpus='artifacts/omni_validation/server_inputs_v1/a2a',audio_encoder='data/frozen_models/SenseVoiceSmall',max_length=3072,batch_size=2,accumulation_steps=2,max_steps=5,train_indices_override=list(range(12))+[48,49,50,51])),
        ('a2a_audio_proj_long',dict(modality='a2a',mode='audio_proj',corpus='artifacts/omni_validation/server_inputs_v1/a2a',audio_encoder='data/frozen_models/SenseVoiceSmall',max_length=3072,batch_size=2,accumulation_steps=2,max_steps=3,epochs=3,train_indices_override=[48,49,50,51])),
        ('i2t_all',dict(modality='i2t',mode='all',corpus='artifacts/omni_validation/server_inputs_v1/i2t',vision_encoder='data/frozen_models/siglip2-base-p32-256-ve',max_length=768,batch_size=4,accumulation_steps=4,max_steps=3,image_placeholder_policy='user_only')),
        ('i2t_vision_proj',dict(modality='i2t',mode='vision_proj',corpus='artifacts/omni_validation/server_inputs_v1/i2t',vision_encoder='data/frozen_models/siglip2-base-p32-256-ve',max_length=768,batch_size=16,accumulation_steps=1,max_steps=3,image_placeholder_policy='user_only')),
    ]
    if a.only:
        names=a.only.split(',')
        if len(set(names))!=len(names) or set(names)-{name for name,_ in tasks}:raise ValueError('Unknown or duplicate check selection')
        tasks=[(name,cfg) for name,cfg in tasks if name in names]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'PYTHONDONTWRITEBYTECODE':'1'}
    report=dict(scope='Full random-initialized315M plus pinned real frozen encoders; small real-data fixtures and all official update modes; not formal learned quality',gpu=a.gpu,started=time.time(),checks=[])
    events=(out/'events.jsonl').open('x',buffering=1)
    try:
        for name,options in tasks:
            snapshot=wait_for_idle(a.gpu,lambda value:events.write(json.dumps(dict(event='gpu_idle_observation',check=name,**value))+'\n'))
            cfg={**common,**options,'run_dir':str((out/name).relative_to(ROOT))}
            cfg_path=out/f'{name}.json';cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
            command=[sys.executable,'-u','-B','-m','lab.omni_train','--config',str(cfg_path)]
            row=dict(name=name,command=command,gpu_snapshot=snapshot,started=time.time(),status='running')
            events.write(json.dumps(dict(event='check_start',**row))+'\n');print(json.dumps(row),flush=True)
            with (out/f'{name}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    row['timed_out']=True
            row.update(returncode=proc.returncode,finished=time.time(),status='passed' if proc.returncode==0 and not row.get('timed_out') else 'failed')
            run=out/name
            if (run/'provenance.json').exists():
                row['provenance']=json.loads((run/'provenance.json').read_text())
                assert row['provenance']['parameters']==314887938
            if (run/'metrics.jsonl').exists():
                metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
                row['projector_gradients']=[r for r in metrics if r.get('event')=='projector_gradient_after_clipping']
                row['input_contracts']=[r for r in metrics if r.get('event')=='batch_contract']
            if cfg['modality']!='t2a' and row['status']=='passed':
                key='audio_proj' if cfg['modality']=='a2a' else 'vision_proj'
                row['nonzero_projector_gradient']=any(r['gradient_norms'][key]>0 for r in row['projector_gradients'])
                if not row['nonzero_projector_gradient']:row['status']='failed_gradient_contract'
            report['checks'].append(row);events.write(json.dumps(dict(event='check_end',**row))+'\n')
            (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(name=name,status=row['status'],returncode=proc.returncode)),flush=True)
    except Exception as exc:
        import traceback
        report['controller_error']=dict(error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['finished']=time.time();report['passed']=len(report['checks'])==len(tasks) and all(r['status']=='passed' for r in report['checks'])
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');events.close()
    if not report['passed']:raise RuntimeError('Some remote paths failed; retained all evidence')


if __name__=='__main__':main()
