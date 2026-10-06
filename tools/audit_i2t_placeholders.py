#!/usr/bin/env python3
"""Audit image control strings by conversation role before modifying any data."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',default='data/processed/sft_i2t_shards_v1'); args=p.parse_args()
    root=ROOT/args.corpus; index=json.loads((root/'index.json').read_text()); out=root/'placeholder_audit_v1'
    if out.exists(): raise RuntimeError('Preserve previous role audit')
    out.mkdir(); n=index['rows']; arrays={role:np.lib.format.open_memmap(out/f'{role}_image_count.npy',mode='w+',dtype='<u4',shape=(n,)) for role in ['user','assistant']}
    sizes=np.load(root/'image_audit_v1/width_height.npy',mmap_mode='r'); counts=Counter(); examples=[]; cursor=0; start=time.perf_counter()
    for shard in index['shards']:
        with pa.memory_map(str(root/shard['file']),'r') as stream:
            batch=ipc.open_file(stream).get_batch(0); convs=batch.column(batch.schema.get_field_index('conversations')).to_pylist()
            for i,encoded in enumerate(convs):
                row=cursor+i; conv=json.loads(encoded); per=Counter()
                for turn in conv:
                    content=turn['content']; role=turn['role']; per[role]+=content.count('<image>')
                    counts['rows_with_raw_special_image_token']+=int('<|image_pad|>' in content)
                for role,arr in arrays.items(): arr[row]=per[role]
                counts['text_only_rows']+=int(per['user']==0); counts['visual_rows']+=int(per['user']>0)
                counts['assistant_placeholder_rows']+=int(per['assistant']>0); counts['multiple_user_placeholders_rows']+=int(per['user']>1)
                counts['system_placeholder_rows']+=int(per['system']>0)
                if per['assistant'] and len(examples)<12: examples.append(dict(row=row,conversation=conv))
        cursor+=len(convs)
        if cursor%500000<len(convs): print(json.dumps(dict(rows=cursor,seconds=time.perf_counter()-start)),flush=True)
    assert cursor==n
    for arr in arrays.values(): arr.flush()
    visual=arrays['user']>0; tiny=(sizes[:,0]<=16)&(sizes[:,1]<=16)
    counts['text_only_rows_with_tiny_image']=int((~visual&tiny).sum()); counts['visual_rows_with_tiny_image']=int((visual&tiny).sum())
    report=dict(rows=n,counts=dict(counts),examples=examples,seconds=time.perf_counter()-start,
                scope='Full raw conversations, not randomly truncated turns; no data changed; <image> counts separated by role',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifacts=str(out.relative_to(ROOT)))
    (ROOT/'reports/omni_i2t_placeholder_audit_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}),flush=True)

if __name__=='__main__': main()
