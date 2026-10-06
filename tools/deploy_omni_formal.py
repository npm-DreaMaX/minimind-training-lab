#!/usr/bin/env python3
"""Stream verified project data to the authorized server; no credential storage."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/new_data/REMOTE_USER/minimind/omni_formal_v1/project'
SSH=['ssh','-S','/tmp/minimind-server-control','-o','BatchMode=yes','-o','ConnectTimeout=10',
     '-o','UserKnownHostsFile=/tmp/minimind-server-known-hosts','REMOTE_USER@SERVER_ADDRESS']


def main():
    out=ROOT/'runs/omni_formal_deployment_v1';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    manifest=ROOT/'reports/omni_formal_deployment_v1/manifest.json'
    while not (manifest.parent/'transfer_files.nul').exists():time.sleep(5)
    setup=f'''set -eu
test ! -e {REMOTE}
mkdir -p {REMOTE}/data/frozen_models {REMOTE}/runs
ln -s /new_data/REMOTE_USER/minimind/omni_readiness_v1/project/data/frozen_models/SenseVoiceSmall {REMOTE}/data/frozen_models/SenseVoiceSmall
ln -s /new_data/REMOTE_USER/minimind/omni_readiness_v1/project/data/frozen_models/siglip2-base-p32-256-ve {REMOTE}/data/frozen_models/siglip2-base-p32-256-ve
'''
    (out/'setup.sh').write_text(setup)
    subprocess.run(SSH+['bash','-s'],input=setup,text=True,check=True)
    commands=[['tar','--dereference','--null','-T',str(manifest.parent/'transfer_files.nul'),'-cf','-'],
              SSH+[f'tar -xf - -C {REMOTE}']]
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    started=time.time()
    sender=subprocess.Popen(commands[0],cwd=ROOT,stdout=subprocess.PIPE)
    receiver=subprocess.Popen(commands[1],stdin=sender.stdout)
    sender.stdout.close()
    receive_code=receiver.wait();send_code=sender.wait()
    result=dict(send_returncode=send_code,receive_returncode=receive_code,started=started,finished=time.time())
    (out/'transfer_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if receive_code or send_code:raise RuntimeError('Incomplete deployment retained; do not train')
    with (out/'remote_verification.log').open('x') as log:
        subprocess.run(SSH+[f'cd {REMOTE} && /new_data/REMOTE_USER/minimind/envs/omni-formal/bin/python -u -B tools/verify_omni_deployment.py'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result.update(passed=True,finished=time.time())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
