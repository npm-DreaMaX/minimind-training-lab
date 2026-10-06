#!/usr/bin/env python3
"""Resume an interrupted prefix using bounded parallel HTTP ranges; SHA256 gate."""
import argparse,concurrent.futures,hashlib,json,os,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('file'); p.add_argument('--workers',type=int,default=4)
p.add_argument('--manifest',default='data/manifests/minimind_full.json')
p.add_argument('--out-dir',default='data/raw/minimind')
p.add_argument('--repo-type',choices=['dataset','model'],default='dataset')
p.add_argument('--chunk-mib',type=int,default=64)
a=p.parse_args(); m=json.loads((ROOT/a.manifest).read_text()); info=m['files'][a.file]
base=ROOT/a.out_dir; base.mkdir(parents=True,exist_ok=True)
prefix=base/(a.file+'.part'); final=base/a.file
parts=base/(a.file+'.ranges'); parts.mkdir(exist_ok=True)
planfile=parts/'plan.json'
if final.exists(): raise SystemExit('Final file already exists; use the verifier')
if planfile.exists(): plan=json.loads(planfile.read_text())
else:
    n=prefix.stat().st_size if prefix.exists() else 0
    if a.chunk_mib<=0: raise ValueError('chunk-mib must be positive')
    chunk=a.chunk_mib*1024*1024
    plan={'prefix_bytes':n,'ranges':[[start,min(start+chunk,info['bytes'])-1] for start in range(n,info['bytes'],chunk)]}
    planfile.write_text(json.dumps(plan,indent=2))
repo_prefix='datasets/' if a.repo_type=='dataset' else ''
url=f"https://huggingface.co/{repo_prefix}{m['repo']}/resolve/{m['revision']}/{a.file}?download=true"
started=time.time()

def fetch(bounds):
    start,end=bounds; dest=parts/f'{start}_{end}.bin'; need=end-start+1
    if dest.exists() and dest.stat().st_size==need: return need
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url+f'&segment={start}',headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=60) as r:
                expected=f'bytes {start}-{end}/{info["bytes"]}'
                if r.status!=206 or r.headers.get('Content-Range')!=expected:
                    raise RuntimeError(f'Range response mismatch: {r.status}, {r.headers.get("Content-Range")}')
                tmp=dest.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    while True:
                        b=r.read(1024*1024)
                        if not b: break
                        f.write(b)
            if tmp.stat().st_size!=need: raise RuntimeError(f'Incomplete range: received {tmp.stat().st_size}, expected {need}')
            tmp.replace(dest); return need
        except Exception as e:
            last=e; print(json.dumps({'event':'range_retry','start':start,'attempt':attempt,'error':repr(e)}),flush=True); time.sleep(min(20,2+attempt*2))
    raise RuntimeError(f'Failed range {bounds}: {last}')

completed=plan['prefix_bytes']
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for n in pool.map(fetch,plan['ranges']):
        completed+=n
        progress={'file':a.file,'completed_bytes':completed,'total_bytes':info['bytes'],'elapsed_s':time.time()-started,'time':time.time()}
        (parts/'progress.json').write_text(json.dumps(progress,indent=2))
        print(json.dumps(progress),flush=True)
assembled=base/(a.file+'.assembled')
h=hashlib.sha256()
with assembled.open('wb') as out:
    paths=([prefix] if plan['prefix_bytes'] else [])+[parts/f'{start}_{end}.bin' for start,end in plan['ranges']]
    for path in paths:
        with path.open('rb') as f:
            while True:
                b=f.read(8*1024*1024)
                if not b: break
                h.update(b); out.write(b)
if assembled.stat().st_size!=info['bytes'] or h.hexdigest()!=info['sha256']:
    raise RuntimeError('Assembled dataset SHA256/size mismatch; all evidence retained')
assembled.replace(final)
(base/(a.file+'.verified.json')).write_text(json.dumps({'name':a.file,'sha256':h.hexdigest(),'bytes':info['bytes'],'revision':m['revision'],'verified_unix':time.time()},indent=2)+'\n')
# Verified segments and interrupted prefix are redundant download cache, not experiments.
if prefix.exists(): prefix.unlink()
for start,end in plan['ranges']: (parts/f'{start}_{end}.bin').unlink()
print(json.dumps({'event':'verified','file':a.file,'seconds':time.time()-started}),flush=True)
