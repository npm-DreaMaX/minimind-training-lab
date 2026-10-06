#!/usr/bin/env python3
"""Run the authorized SFT after full pretraining and full SFT data are ready.

Failures stop this transition, rather than silently shrinking a model or claiming
success. The primary agent continues to monitor and diagnose retained evidence.
"""
import fcntl,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json
pre=ROOT/'models/01_moe/runs/pretrain_full_v1'
cfg_path=ROOT/'models/01_moe/configs/sft_full_v1.json'
data=ROOT/'data/processed/sft_t2t_full_v1'
print(json.dumps({'event':'successor_started','pid':os.getpid(),'time':time.time(),
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),flush=True)
print('Waiting for completed formal pretraining and verified full SFT cache',flush=True)
while True:
    status=read_live_json(pre/'status.json')
    if status.get('status')=='failed': raise RuntimeError('Pretraining failed; inspect its failure.json before a new stage')
    if status.get('status')=='complete' and (data/'metadata.json').exists(): break
    time.sleep(10)
meta=json.loads((data/'metadata.json').read_text())
assert meta['stage']=='sft' and meta['limit']==0 and meta['bad_rows']==0,meta
assert meta['train_rows']>100000 and meta['val_rows']>100,meta
assert math.isfinite(status['validation']['validation_ce']),status
audit=json.loads((ROOT/'reports/pretrain_data_audit.json').read_text())
expected=audit['recipes']['512']['estimated_supervised_train_tokens']
assert status['trained_tokens']>=expected*.999,'Not the planned full pretraining budget'
assert status['step']==audit['formal_optimizer_steps'],'Unexpected pretraining step budget'
cfg=json.loads(cfg_path.read_text())
coverage=meta['length_coverage'][str(cfg['seq_len'])]
assert coverage['zero_supervision_rows']/meta['rows']<.02,'Too many SFT examples lose all answer labels; review sequence length'
weight=ROOT/cfg['init_weight']; assert weight.exists(),weight
h=hashlib.sha256()
with weight.open('rb') as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
selection=json.loads((pre/'checkpoints/best_validation.json').read_text())
lineage={'pretrain_run':str(pre.relative_to(ROOT)),'selected_checkpoint_step':selection['step'],
         'selected_validation_ce':selection['validation_ce'],'init_weight_sha256':h.hexdigest(),
         'new_optimizer':True,'transfer_precision':'FP16 exported weights loaded into FP32 parameters; this is a new training stage, not exact resume',
         'sft_metadata':meta,'time':time.time()}
cfg['initialization_provenance']=lineage
cfg['recipe_status']='full data and completed pretrain budget checked; starting the authorized full SFT epoch'
cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
# The original PID becomes stale after resume. Wait for the actual shared GPU
# lock to be released after the final checkpoint/telemetry cleanup instead.
with (ROOT/'runs/local_gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    fcntl.flock(lock,fcntl.LOCK_UN)
command=[sys.executable,'-u','-B','-m','lab.train','--config',str(cfg_path.relative_to(ROOT))]
with (ROOT/'logs/phase_commands.jsonl').open('a') as f: f.write(json.dumps({'command':command,'time':time.time(),'cwd':str(ROOT)})+'\n')
print(json.dumps({'event':'starting_sft','command':command,'lineage':lineage},ensure_ascii=False),flush=True)
training_env=os.environ.copy()
if cfg.get('allocator_conf'): training_env['PYTORCH_ALLOC_CONF']=cfg['allocator_conf']
with (ROOT/'logs/sft_full_v1.log').open('x') as log:
    subprocess.run(command,cwd=ROOT,env=training_env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Formal SFT budget completed; evaluation and checkpoints retained in its run directory',flush=True)
