#!/usr/bin/env python3
"""Deliver completed AR lineage and continuously mirror the authorized Omni branch.

Uses an existing SSH multiplex session, never a stored password. Network errors
remain visible and retry; remote training is not killed. No deletion propagation.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import check_completed,sha256
from lab.live_status import read_live_json

REMOTE='/new_data/REMOTE_USER/minimind/omni_formal_v1/project'
HOST='REMOTE_USER@SERVER_ADDRESS'
SSH=['ssh','-S','/tmp/minimind-server-control','-o','BatchMode=yes','-o','ConnectTimeout=10',
     '-o','UserKnownHostsFile=/tmp/minimind-server-known-hosts',HOST]
TRANSPORT='ssh -S /tmp/minimind-server-control -o BatchMode=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/tmp/minimind-server-known-hosts'
RSYNC=['rsync','-rt','--whole-file','--delay-updates','--partial-dir=.rsync-partial','--timeout=120',
       '--rsync-path=/new_data/REMOTE_USER/minimind/envs/transfer-tools/bin/rsync','-e',TRANSPORT]


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    out=ROOT/'runs/omni_delivery_bridge_v1';out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/f'source_{time.time_ns()}.py').write_bytes(Path(__file__).read_bytes())
    plan=json.loads((ROOT/'plans/formal_omni_server_v1.json').read_text())
    dependency=plan['stages'][0]['dependencies'][0]
    def event(name,**fields):
        row=dict(event=name,time=time.time(),pid=os.getpid(),**fields)
        with (out/'events.jsonl').open('a') as log:log.write(json.dumps(row,ensure_ascii=False)+'\n')
        atomic_json(out/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
    def execute(command):
        with (out/'transport.log').open('a') as log:
            log.write(json.dumps(dict(command=command,time=time.time()))+'\n');log.flush()
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def exists(path):
        result=subprocess.run(SSH+[f'test -e {REMOTE}/{path}'],capture_output=True)
        if result.returncode==255:raise ConnectionError('SSH multiplex authentication/network unavailable')
        if result.returncode not in [0,1]:raise RuntimeError(result.stderr.decode())
        return result.returncode==0
    def mirror_metadata():
        for folder in ['models/03_omni_moe/runs','runs/formal_omni_server_v1','runs/omni_formal_supervisor_v1']:
            if exists(folder):
                (ROOT/folder).mkdir(parents=True,exist_ok=True)
                execute(RSYNC+['--exclude=*.pt','--exclude=*.pth','--exclude=*.tmp','--exclude=*.lock',
                               f'{HOST}:{REMOTE}/{folder}/',str(ROOT/folder)+'/'])
    last_weights=last_full_resume=0.;mirrored_completed=set()
    uploaded=(out/'sft_uploaded.json').exists()
    event('started',uploaded_sft=uploaded)
    while True:
        try:
            # Also refreshes the authorized SSH master while waiting for days.
            execute(SSH+['true'])
            if not uploaded:
                status=read_live_json(ROOT/dependency['run_dir']/'status.json',missing_ok=True) or {}
                if status.get('status')=='failed':raise ValueError('Formal local SFT failed; do not transfer an early checkpoint')
                if status.get('status')=='complete':
                    lineage=check_completed(ROOT,dependency)
                    run=dependency['run_dir']
                    names=[run+'/'+x for x in ['config.json','provenance.json','checkpoints/best_validation.json',
                                              'checkpoints/best_validation.pth','checkpoints/latest_resume.pt']]
                    event('uploading_completed_sft',selected_sha256=lineage['selected_weight_sha256'])
                    execute(RSYNC+['--relative',*names,f'{HOST}:{REMOTE}/'])
                    result=subprocess.check_output(SSH+[f'sha256sum {REMOTE}/{run}/checkpoints/best_validation.pth'],text=True)
                    if result.split()[0]!=lineage['selected_weight_sha256']:raise ValueError('Remote selected SFT hash mismatch')
                    # Publish completion last, after the immutable selected weight,
                    # full recovery state and provenance have actually arrived.
                    execute(RSYNC+['--relative',run+'/status.json',f'{HOST}:{REMOTE}/'])
                    atomic_json(out/'sft_uploaded.json',lineage);uploaded=True
                    event('sft_uploaded_and_verified',selected_sha256=lineage['selected_weight_sha256'])
                else:
                    event('waiting_completed_sft',local_status=status.get('status','not_started'))
            mirror_metadata()
            controller=read_live_json(ROOT/'runs/formal_omni_server_v1/status.json',missing_ok=True) or {}
            complete=controller.get('event')=='pipeline_budget_complete'
            failed=controller.get('event')=='failed'
            completed_stages={p.name for p in (ROOT/plan['controller_dir']).glob('*_completed.json')}
            new_completed=bool(completed_stages-mirrored_completed)
            full_resume_due=time.time()-last_full_resume>=21600 or new_completed or complete or failed
            if (time.time()-last_weights>=3600 or new_completed or complete or failed) and exists('models/03_omni_moe/runs'):
                if shutil.disk_usage(ROOT).free<20*1024**3:
                    event('delivery_disk_headroom_low',note='Preserved local data; remote training not stopped; inspect storage allocation')
                else:
                    includes=['--include=*/','--include=*.pth']+(['--include=*.pt'] if full_resume_due else [])
                    execute(RSYNC+includes+['--exclude=*',
                                   f'{HOST}:{REMOTE}/models/03_omni_moe/runs/',str(ROOT/'models/03_omni_moe/runs')+'/'])
                    last_weights=time.time()
                    if full_resume_due:last_full_resume=last_weights
                    mirrored_completed=completed_stages
                    event('checkpoints_mirrored',full_resume_included=full_resume_due,
                          integrity='rsync verified transfer; selected weights checked against stage SHA below')
                    for stage in plan['stages']:
                        if stage['kind']!='train':continue
                        receipt=ROOT/plan['controller_dir']/f'{stage["id"]}_completed.json'
                        if not receipt.exists():continue
                        entry=json.loads(receipt.read_text());weight=ROOT/entry['run_dir']/'checkpoints/best_validation.pth'
                        if sha256(weight)!=entry['selected_weight_sha256']:raise ValueError(f'Delivered selected weight mismatch: {weight}')
                    if complete:
                        # Complete budgets/delivery still do not imply useful
                        # generation quality or the complete user learning goal.
                        event('all_omni_budgets_and_delivery_complete',quality_review_pending=True)
                        return
            if failed:
                event('remote_training_failed',remote_status=controller)
                return
        except (subprocess.CalledProcessError,ConnectionError) as exc:
            event('transport_retry_pending',error=repr(exc),note='Remote job left untouched; retry in60s; no credential stored')
        except Exception as exc:
            event('failed',error=repr(exc),traceback=traceback.format_exc())
            raise
        time.sleep(60)


if __name__=='__main__':main()
