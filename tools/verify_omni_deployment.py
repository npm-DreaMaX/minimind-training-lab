#!/usr/bin/env python3
"""Verify every deployed byte before formal remote data use."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import sha256,assert_pinned


def main():
    if not str(ROOT).startswith('/new_data/REMOTE_USER/minimind/'):
        raise RuntimeError('Expected authorized remote deployment')
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='reports/omni_formal_deployment_v1/manifest.json')
    args=parser.parse_args();path=ROOT/args.manifest
    manifest=json.loads(path.read_text())
    for i,(name,info) in enumerate(manifest['files'].items()):
        source=ROOT/name
        if source.stat().st_size!=info['bytes'] or sha256(source)!=info['sha256']:
            raise RuntimeError(f'Deployment byte mismatch: {name}')
        if i%100==0:print(json.dumps(dict(verified=i+1,total=len(manifest['files']))),flush=True)
    plan=json.loads((ROOT/'plans/formal_omni_server_v1.json').read_text())
    assert_pinned(ROOT,plan['fingerprints'])
    from lab.omni_batch import official_modules
    module,collate=official_modules()
    assert callable(collate) and hasattr(module,'MiniMindOmni')
    result=dict(passed=True,manifest_sha256=sha256(path),files=manifest['file_count'],bytes=manifest['bytes'],time=time.time())
    (path.parent/'remote_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
