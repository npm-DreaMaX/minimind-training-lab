#!/usr/bin/env python3
"""Download pinned official files; retain partials and verify LFS SHA256."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('files',nargs='+')
a=p.parse_args()
m=json.loads((ROOT/'data/manifests/minimind_full.json').read_text())
dest=ROOT/'data/raw/minimind'
dest.mkdir(parents=True,exist_ok=True)
for name in a.files:
    expected=m['files'][name]
    target=dest/name
    if not target.exists():
        part=target.with_suffix(target.suffix+'.part')
        url=f"https://huggingface.co/datasets/{m['repo']}/resolve/{m['revision']}/{name}?download=true"
        print(json.dumps({'event':'download_start','file':name,'time':time.time(),'url':url}),flush=True)
        cmd=['curl','--location','--fail','--silent','--show-error','--retry','8','--retry-all-errors',
             '--retry-delay','3','--connect-timeout','30','--continue-at','-','--output',str(part),url]
        subprocess.run(cmd,check=True)
        if part.stat().st_size!=expected['bytes']: raise RuntimeError(f'Size mismatch: {name}')
        target=part
    h=hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    if target.stat().st_size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
        raise RuntimeError(f'Integrity mismatch: {name}')
    if target.name.endswith('.part'): target.replace(dest/name)
    (dest/(name+'.verified.json')).write_text(json.dumps({'name':name,'sha256':h.hexdigest(),'bytes':expected['bytes'],
                                                        'revision':m['revision'],'verified_unix':time.time()},indent=2)+'\n')
    print(json.dumps({'event':'verified','file':name,'time':time.time()}),flush=True)
