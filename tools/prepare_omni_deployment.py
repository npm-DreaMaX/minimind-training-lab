#!/usr/bin/env python3
"""Inventory full immutable data and executable inputs for one remote deployment.

No large local tar copy: the NUL file list can be streamed with tar to the server.
Frozen encoders already on the server are shared via owned, verified symlinks.
"""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import sha256


def main():
    out=ROOT/'reports/omni_formal_deployment_v1'
    out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'plans/formal_omni_server_v1.json').read_text())
    names=set(plan['fingerprints'])
    for folder in ['lab','evaluation','sources/minimind-o/model','sources/minimind-o/dataset','sources/minimind-o/trainer','upstream/model']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts:
                if path.suffix not in ['.pyc','.pt','.pth','.safetensors']:
                    names.add(str(path.relative_to(ROOT)))
    for modality in ['t2a','a2a','i2t']:
        folder=ROOT/f'data/processed/sft_{modality}_shards_v1'
        names.update(str(path.relative_to(ROOT)) for path in folder.rglob('*') if path.is_file() and path.suffix!='.tmp')
    for folder in ['artifacts/omni_validation/t2a_shards_v1','artifacts/omni_validation/server_inputs_v1']:
        names.update(str(path.relative_to(ROOT)) for path in (ROOT/folder).rglob('*') if path.is_file())
    shared=set()
    frozen=json.loads((ROOT/'data/manifests/omni_frozen_models.json').read_text())
    for model in ['SenseVoiceSmall','siglip2-base-p32-256-ve','mimi']:
        for filename in frozen[model]['files']:
            if filename=='.gitattributes':continue
            name=f'data/frozen_models/{model}/{filename}'
            names.add(name)
            if model!='mimi':shared.add(name)
    for path in ['plans/formal_omni_server_v1.json','data/manifests/omni_frozen_models.json',
                 'reports/server_omni_inputs_bundle_v1.json','tools/validate_server_omni_inputs.py',
                 'tools/validate_server_formal_recipe.py','tools/verify_omni_deployment.py']:
        names.add(path)
    manifest={}
    for i,name in enumerate(sorted(names)):
        path=ROOT/name
        manifest[name]=dict(bytes=path.stat().st_size,sha256=sha256(path),shared_remote_asset=name in shared)
        if i%100==0:print(json.dumps(dict(files_hashed=i+1,total=len(names),time=time.time())),flush=True)
    report=dict(files=manifest,bytes=sum(x['bytes'] for x in manifest.values()),file_count=len(manifest),time=time.time(),
                data='Full immutable indexed corpora, original row identity retained; fixtures only for preflight')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    transfer=sorted(names-shared)+[str((out/'manifest.json').relative_to(ROOT))]
    (out/'transfer_files.nul').write_bytes(b''.join(x.encode()+b'\0' for x in transfer))
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
