#!/usr/bin/env python3
"""Fetch only the pinned frozen encoders/codecs, never a pretrained MiniMind LLM."""
import hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data/manifests/omni_frozen_models.json').read_text())
for name,model in manifest.items():
    out=ROOT/'data/frozen_models'/name
    out.mkdir(parents=True,exist_ok=True)
    pinned=ROOT/'data/manifests/frozen'/f'{name}.json'; pinned.parent.mkdir(exist_ok=True)
    pinned.write_text(json.dumps({'repo':model['repo'],'revision':model['revision'],'files':model['files']},indent=2)+'\n')
    checked={}
    for filename,info in model['files'].items():
        path=out/filename
        if not path.exists() or path.stat().st_size!=info['bytes']:
            if info['bytes']>8*1024*1024:
                if path.exists(): raise RuntimeError(f'Unexpected incomplete final file: {path}')
                subprocess.run([sys.executable,str(ROOT/'tools/download_ranges.py'),filename,'--manifest',str(pinned),
                                '--out-dir',str(out),'--repo-type','model','--workers','2','--chunk-mib','8'],check=True)
            else:
                url=f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{filename}"
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(url,timeout=60) as response: data=response.read(info['bytes']+1)
                        if len(data)!=info['bytes']: raise IOError(f'Incomplete small file: {filename}')
                        temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_bytes(data); temporary.replace(path)
                        break
                    except Exception:
                        if attempt==4: raise
                        time.sleep(2+attempt)
        assert path.stat().st_size==info['bytes'],filename
        sha=hashlib.sha256(); git_blob=hashlib.sha1(f'blob {info["bytes"]}\0'.encode())
        with path.open('rb') as f:
            while data:=f.read(4*1024*1024): sha.update(data); git_blob.update(data)
        if info['sha256']: assert sha.hexdigest()==info['sha256'],filename
        elif info['git_blob_id']: assert git_blob.hexdigest()==info['git_blob_id'],filename
        checked[filename]={'bytes':info['bytes'],'sha256':sha.hexdigest()}
    (out/'verified.json').write_text(json.dumps({'repo':model['repo'],'revision':model['revision'],'files':checked,'time':time.time()},indent=2)+'\n')
    print(json.dumps({'event':'frozen_model_verified','name':name,'revision':model['revision']}),flush=True)
