#!/usr/bin/env python3
"""Wait for verified deployment/runtime, then queue the authorized Omni branch."""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from bridge_omni_server import SSH,RSYNC,REMOTE,HOST,atomic_json
READY='/new_data/REMOTE_USER/minimind/omni_readiness_v1/project/runs/formal_runtime_preflight_v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/omni_formal_arming_v1')
    parser.add_argument('--deployment-run',default='runs/omni_formal_deployment_v2')
    args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(name,**fields):
        row=dict(event=name,time=time.time(),**fields)
        with (out/'events.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    try:
        event('waiting_verified_deployment_and_runtime')
        while True:
            path=ROOT/args.deployment_run/'result.json'
            if path.exists():
                if not json.loads(path.read_text())['passed']:raise RuntimeError('Deployment failed')
                break
            # A failed rsync/verification is preserved, never treated as ready.
            state_path=ROOT/args.deployment_run/'status.json'
            if state_path.exists() and json.loads(state_path.read_text()).get('status')=='failed':
                raise RuntimeError('Deployment process failed; inspect retained log')
            time.sleep(30)
        while True:
            result=subprocess.run(SSH+[f'test -f {READY}/result.json && cat {READY}/result.json'],capture_output=True,text=True)
            if result.returncode==0:
                if not json.loads(result.stdout)['passed']:raise RuntimeError('Runtime preflight failed')
                break
            failure=subprocess.run(SSH+['tail -20 /new_data/REMOTE_USER/minimind/logs/formal_runtime_preflight_v1.log'],capture_output=True,text=True)
            if 'Traceback (most recent call last)' in failure.stdout:raise RuntimeError('Runtime preflight failed; inspect remote raw log')
            time.sleep(30)
        local=ROOT/'models/03_omni_moe/runs/server_formal_runtime_v1';local.mkdir(parents=True,exist_ok=True)
        subprocess.run(RSYNC+['--exclude=*.pt','--exclude=*.pth',f'{HOST}:{READY}/',str(local)+'/'],check=True)
        command=f'''set -eu
mkdir -p {REMOTE}/runs/formal_runtime_preflight_v1
/new_data/REMOTE_USER/minimind/envs/transfer-tools/bin/rsync -rt --exclude='*.pt' --exclude='*.pth' {READY}/ {REMOTE}/runs/formal_runtime_preflight_v1/
source /new_data/REMOTE_USER/minimind/env/activate_server_omni_formal.sh
cd {REMOTE}
python - <<'PY'
import json,subprocess,sys,time
from pathlib import Path
Path('logs').mkdir(exist_ok=True)
with open('logs/omni_formal_supervisor_v1.log','x') as stream:
    proc=subprocess.Popen([sys.executable,'-u','-B','tools/start_remote_omni.py'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,time=time.time())))
PY
'''
        (out/'remote_launch.sh').write_text(command)
        result=subprocess.run(SSH+['bash','-s'],input=command,text=True,capture_output=True,check=True)
        event('remote_supervisor_launched',launch=result.stdout.strip(),note='Formal Omni waits for completed local SFT; no early weight is uploaded')
        # Confirm that the child survived launch and reached a real wait state.
        for _ in range(12):
            result=subprocess.run(SSH+[f'cat {REMOTE}/runs/formal_omni_server_v1/status.json'],capture_output=True,text=True)
            if result.returncode==0:
                status=json.loads(result.stdout)
                if status['event']=='failed':raise RuntimeError(f'Remote controller failed: {status}')
                event('queue_alive',remote_status=status)
                return
            time.sleep(5)
        raise RuntimeError('Remote supervisor launch did not produce controller status')
    except Exception as exc:
        event('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
