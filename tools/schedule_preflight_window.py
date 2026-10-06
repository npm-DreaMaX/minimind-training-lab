#!/usr/bin/env python3
"""One-shot resource-check window after a formal optimizer-step boundary.

Waits without taking GPU memory; the existing controller owns save/check/resume.
This is a local project process, not an external scheduling service.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--after-step',required=True,type=int)
    p.add_argument('--checks',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    out=ROOT/(a.out+'_scheduler')
    if out.exists(): raise RuntimeError('Preserve existing scheduler evidence')
    out.mkdir(); (out/'source.py').write_bytes(Path(__file__).read_bytes())
    checks=ROOT/a.checks; digest=hashlib.sha256(checks.read_bytes()).hexdigest()
    (out/'request.json').write_text(json.dumps(dict(pid=os.getpid(),after_step=a.after_step,checks=a.checks,checks_sha256=digest,window=a.out),indent=2)+'\n')
    status_path=ROOT/'models/01_moe/runs/pretrain_full_v1/status.json'
    def record(event,**fields):
        value=dict(event=event,time=time.time(),**fields)
        (out/'status.json').write_text(json.dumps(value,indent=2)+'\n'); print(json.dumps(value),flush=True)
    record('waiting',after_step=a.after_step)
    try:
        while True:
            status=read_live_json(status_path)
            if status['status']=='failed': raise RuntimeError('Formal training failed; inspect before scheduling checks')
            if status['status']=='complete': raise RuntimeError('Formal run already completed; do not interrupt its next stage')
            if status['status']=='training' and status['step']>=a.after_step: break
            time.sleep(10)
        if hashlib.sha256(checks.read_bytes()).hexdigest()!=digest: raise RuntimeError('Queued check manifest changed; record a new scheduler')
        pid=int((ROOT/'runs/local_gpu.lock').read_text().strip())
        command=[sys.executable,'-u','-B','tools/run_preflight_window.py','--pid',str(pid),'--out',a.out,'--checks',a.checks]
        record('starting',pid=pid,observed_step=status['step'],command=command)
        with (ROOT/'logs'/Path(a.out).name).with_suffix('.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        record('controller_finished',window=a.out,note='Check individual event results and resumed process; controller exit alone is not a model completion signal')
    except Exception as exc:
        record('failed',error=repr(exc),traceback=traceback.format_exc()); raise

if __name__=='__main__': main()
