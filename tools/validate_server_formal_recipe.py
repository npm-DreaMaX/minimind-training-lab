#!/usr/bin/env python3
"""Actual accumulation8 memory check in the independent production runtime."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from server_gpu_guard import wait_for_idle
ROOT=Path(__file__).resolve().parents[1]


def main():
    assert sys.prefix=='/new_data/REMOTE_USER/minimind/envs/omni-formal'
    out=ROOT/'runs/formal_runtime_preflight_v1';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    cfg=dict(purpose='preflight',device='cuda',model=dict(hidden_size=768,num_hidden_layers=8,use_moe=True),
             run_dir=str((out/'t2a_B16_acc8').relative_to(ROOT)),modality='t2a',mode='all',
             corpus='artifacts/omni_validation/t2a_shards_v1',max_length=1536,batch_size=16,accumulation_steps=8,
             max_steps=3,epochs=1,learning_rate=5e-4,warmup_steps=100,seed=20261018,cpu_threads=4,
             num_workers=2,prefetch_factor=1,deterministic=True,checkpointing=True,allocator_gib=13.5,
             optimizer_state_offload=False,eval_interval=50,save_interval=50,validation_samples=8,
             log_interval=1,inspect_first_batches=2)
    path=out/'t2a_B16_acc8.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1'}
    wait_for_idle(2,lambda x:print(json.dumps(dict(event='idle_observation',**x)),flush=True))
    command=[sys.executable,'-u','-B','-m','lab.omni_train','--config',str(path)]
    (out/'command.json').write_text(json.dumps(command)+'\n')
    with (out/'t2a_B16_acc8.log').open('x') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
    # New independent Python and official torch wheels must also execute all
    # previously validated input paths; no inherited third-party site-packages.
    with (out/'input_modes.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-B','tools/validate_server_omni_inputs.py','--gpu','2',
                        '--out',str((out/'input_modes').relative_to(ROOT))],cwd=ROOT,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
    report=dict(passed=True,time=time.time(),prefix=sys.prefix,
                scope='Actual full315M T2A B16 accumulation8 plus all input/update modes; small fixtures only, not trained model quality')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
