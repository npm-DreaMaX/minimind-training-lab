#!/usr/bin/env python3
"""Record CPU generation probes after each completed formal language stage."""
import argparse,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/completed_language_evaluation_v1');a=p.parse_args();out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve existing observer evidence')
    out.mkdir(); (out/'source.py').write_bytes(Path(__file__).read_bytes())
    jobs=[dict(run=f'models/01_moe/runs/{stage}_full_v1',config=f'models/01_moe/configs/{stage}_full_v1.json',stage=stage) for stage in ['pretrain','sft']]
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    def record(event,**fields):
        row=dict(event=event,time=time.time(),pid=os.getpid(),**fields)
        with (out/'events.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        for job in jobs:
            run=ROOT/job['run']; record('waiting',run=job['run']);seen=False
            while True:
                status=read_live_json(run/'status.json',missing_ok=not seen) or {}
                seen=seen or bool(status)
                if status.get('status')=='failed': raise RuntimeError(f'Formal stage failed: {job["run"]}')
                if status.get('status')=='complete': break
                time.sleep(20)
            if not math.isfinite(status['validation']['validation_ce']): raise RuntimeError('Invalid completed validation')
            dest=run/'evaluation/final_selected_cpu_probes_v1'
            command=[sys.executable,'-u','-B','tools/evaluate_text_probes.py','--config',job['config'],'--stage',job['stage'],
                     '--weights',str((run/'checkpoints/best_validation.pth').relative_to(ROOT)),
                     '--out',str(dest.relative_to(ROOT)),'--device','cpu']
            # Keep the primary suite protocol identical to the official reference.
            record('evaluation_start',command=command,completed_status=status,
                   selection=json.loads((run/'checkpoints/best_validation.json').read_text()))
            with (ROOT/'logs'/f'final_{job["stage"]}_cpu_probes_v1.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((dest/'report.json').read_text())
            if not report['complete']: raise RuntimeError('Incomplete generation suite')
            record('evaluation_written',report=str((dest/'report.json').relative_to(ROOT)),note='Generated outputs require analysis; no automatic quality-success claim')
    except Exception as exc:
        record('failed',error=repr(exc),traceback=traceback.format_exc()); raise

if __name__=='__main__': main()
