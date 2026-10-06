#!/usr/bin/env python3
"""Bounded scan of encoded image assets, headers and train/val sharing."""
import argparse,hashlib,io,json,resource,time
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',default='data/processed/sft_i2t_shards_v1'); p.add_argument('--report',required=True)
    args=p.parse_args(); corpus=ROOT/args.corpus; index=json.loads((corpus/'index.json').read_text())
    out=corpus/'image_audit_v1'
    if out.exists(): raise RuntimeError('Preserve prior audit')
    out.mkdir(); start=time.perf_counter(); n=index['rows']; cursor=0
    split=np.load(corpus/'audit_v1/content_hash64.npy',mmap_mode='r')
    hashes=np.lib.format.open_memmap(out/'image_sha256.npy',mode='w+',dtype='u1',shape=(n,32)); hashes[:]=0
    sizes=np.lib.format.open_memmap(out/'width_height.npy',mode='w+',dtype='<u4',shape=(n,2)); sizes[:]=0
    assets={}; train=set(); val=set(); counts=Counter(); formats=Counter(); errors=[]; val_rows=[]
    for shard in index['shards']:
        with pa.memory_map(str(corpus/shard['file']),'r') as stream:
            batch=ipc.open_file(stream).get_batch(0); images=batch.column(batch.schema.get_field_index('image_bytes'))
            for i in range(batch.num_rows):
                row=cursor+i; encoded=images[i].as_py()
                if not encoded: counts['empty_image_rows']+=1; continue
                if not isinstance(encoded,bytes): raise TypeError(f'Unexpected image payload at row {row}: {type(encoded)}')
                digest=hashlib.sha256(encoded).digest(); hashes[row]=np.frombuffer(digest,dtype='u1')
                counts['image_rows']+=1; counts['encoded_image_bytes_with_repetition']+=len(encoded)
                if digest not in assets:
                    try:
                        with Image.open(io.BytesIO(encoded)) as image: info=(image.width,image.height,image.format)
                    except Exception as exc:
                        info=(0,0,'unreadable'); errors.append(dict(row=row,sha256=digest.hex(),error=repr(exc)))
                    assets[digest]=info; counts['unique_encoded_image_bytes']+=len(encoded)
                w,h,fmt=assets[digest]; sizes[row]=[w,h]; formats[fmt]+=1
                if split[row]%1000<1: val.add(digest); val_rows.append(row)
                else: train.add(digest)
        cursor+=batch.num_rows
        if cursor%100000<batch.num_rows: print(json.dumps(dict(rows=cursor,unique=len(assets),seconds=time.perf_counter()-start)),flush=True)
    assert cursor==n; hashes.flush(); sizes.flush(); shared=train&val
    unseen=[row for row in val_rows if hashes[row].tobytes() not in train]
    np.save(out/'validation_unseen_asset_rows.npy',np.asarray(unseen,dtype='<u8'))
    report=dict(corpus=args.corpus,rows=n,counts=dict(counts),formats=dict(formats),unique_assets=len(assets),
                train_assets=len(train),validation_assets=len(val),shared_assets=len(shared),validation_image_rows=len(val_rows),
                validation_unseen_asset_rows=len(unseen),header_errors=errors,
                observed_width_height_min=sizes[sizes[:,0]>0].min(0).tolist() if assets else None,
                observed_width_height_max=sizes.max(0).tolist() if assets else None,
                seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Exact encoded image SHA256 and headers, not perceptual deduplication or full pixel decode; preserves original conversation split',
                index_sha256=hashlib.sha256((corpus/'index.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifacts=str(out.relative_to(ROOT)))
    target=ROOT/args.report; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
