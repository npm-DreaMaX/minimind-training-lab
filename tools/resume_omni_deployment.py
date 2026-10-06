#!/usr/bin/env python3
"""Resume the same byte manifest without a timeout-prone full checksum prescan."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import sha256
from bridge_omni_server import SSH,RSYNC,REMOTE,HOST,atomic_json


def main():
    out=ROOT/'runs/omni_formal_deployment_v3';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    manifest=ROOT/'reports/omni_formal_deployment_v2/manifest.json'
    transport=[value if value!='--timeout=120' else '--timeout=900' for value in RSYNC]
    command=transport+['--copy-links','--compress','--compress-choice=zstd','--compress-level=3',
                       '--info=progress2,stats2','--from0','--files-from='+str(manifest.parent/'transfer_files.nul'),
                       './',f'{HOST}:{REMOTE}/']
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    atomic_json(out/'status.json',dict(status='transferring',time=time.time(),note='Same expected hashes; transport quick-check plus mandatory post-transfer full SHA256'))
    try:
        for attempt in range(1,4):
            with (out/f'rsync_attempt_{attempt}.log').open('x') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:break
            atomic_json(out/'status.json',dict(status='transport_retry',attempt=attempt,returncode=result.returncode,time=time.time()))
            if result.returncode not in [10,12,23,30,35,255] or attempt==3:raise RuntimeError(f'Transport failed: {result.returncode}')
            time.sleep(10)
        atomic_json(out/'status.json',dict(status='verifying_all_hashes',time=time.time()))
        with (out/'remote_verification.log').open('x') as log:
            subprocess.run(SSH+[f'cd {REMOTE} && /new_data/REMOTE_USER/minimind/envs/omni-formal/bin/python -u -B tools/verify_omni_deployment.py --manifest reports/omni_formal_deployment_v2/manifest.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=dict(passed=True,time=time.time(),manifest_sha256=sha256(manifest))
        atomic_json(out/'result.json',result);atomic_json(out/'status.json',dict(status='complete',**result))
    except Exception as exc:
        atomic_json(out/'status.json',dict(status='failed',error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
