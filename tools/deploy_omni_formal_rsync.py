#!/usr/bin/env python3
"""Resume the retained slow tar deployment with compressed, verified rsync."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import sha256
from bridge_omni_server import REMOTE,HOST,SSH,RSYNC


def main():
    out=ROOT/'runs/omni_formal_deployment_v2';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    old=json.loads((ROOT/'reports/omni_formal_deployment_v1/manifest.json').read_text())
    files=old['files']
    # Data expected hashes stay exactly the same as the already completed local
    # inventory. Verify those hashes remotely; do not assume unchanged mtime is proof.
    updates=['plans/formal_omni_server_v1.json','tools/verify_omni_deployment.py','tools/start_remote_omni.py',
             'sources/minimind-o/trainer/trainer_utils.py','sources/minimind-o/trainer/train.sh']
    for name in updates:files[name]=dict(bytes=(ROOT/name).stat().st_size,sha256=sha256(ROOT/name),shared_remote_asset=False)
    directory=ROOT/'reports/omni_formal_deployment_v2';directory.mkdir(exist_ok=False)
    manifest=dict(files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files.values()),time=time.time(),
                  prior_manifest_sha256=sha256(ROOT/'reports/omni_formal_deployment_v1/manifest.json'),
                  change='Resumable compressed transfer; include transitively imported trainer_utils; no dataset/model/recipe change')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    names=sorted(name for name,info in files.items() if not info.get('shared_remote_asset'))+['reports/omni_formal_deployment_v2/manifest.json']
    file_list=directory/'transfer_files.nul';file_list.write_bytes(b''.join(name.encode()+b'\0' for name in names))
    command=RSYNC+['--copy-links','--checksum','--compress','--compress-choice=zstd','--compress-level=3',
                   '--from0','--files-from='+str(file_list),'./',f'{HOST}:{REMOTE}/']
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    started=time.time()
    with (out/'rsync.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'remote_verification.log').open('x') as log:
        subprocess.run(SSH+[f'cd {REMOTE} && /new_data/REMOTE_USER/minimind/envs/omni-formal/bin/python -u -B tools/verify_omni_deployment.py --manifest reports/omni_formal_deployment_v2/manifest.json'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(passed=True,started=started,finished=time.time(),manifest_sha256=sha256(directory/'manifest.json'))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
