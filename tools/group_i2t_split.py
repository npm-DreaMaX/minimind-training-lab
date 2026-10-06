#!/usr/bin/env python3
"""Versioned split: connect shared visual assets and identical conversations.

Text-only replay rows do not share a visual input merely because they carry
the same unused thumbnail. Original conversation-hash split is preserved.
"""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    root=ROOT/'data/processed/sft_i2t_shards_v1'; out=root/'split_image_v2'
    if out.exists(): raise RuntimeError('Preserve previous grouping')
    original=json.loads((root/'audit_v1/metadata.json').read_text()); n=original['rows']; start=time.perf_counter()
    conversation=np.load(root/'audit_v1/content_hash64.npy',mmap_mode='r')
    image=np.load(root/'image_audit_v1/image_sha256.npy',mmap_mode='r').view('V32').reshape(-1)
    visual=np.load(root/'placeholder_audit_v1/user_image_count.npy',mmap_mode='r')>0
    parent=np.arange(n,dtype=np.int32)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=int(parent[x])
        return x
    def union(a,b):
        a,b=find(int(a)),find(int(b))
        if a!=b: parent[max(a,b)]=min(a,b)
    def link_same(keys,rows):
        order=np.argsort(keys,kind='stable'); ordered=keys[order]; rows=rows[order]
        duplicates=np.flatnonzero(ordered[1:]==ordered[:-1])+1
        for i in duplicates: union(rows[i-1],rows[i])
        return len(duplicates)
    visual_rows=np.flatnonzero(visual)
    image_edges=link_same(image[visual_rows],visual_rows)
    conversation_edges=link_same(conversation,np.arange(n))
    while not np.array_equal(parent,parent[parent]): parent=parent[parent]
    val=(conversation[parent]%1000)<1
    # Audit both invariants independently of the union loop.
    train_images=np.unique(image[visual&~val]); val_images=np.unique(image[visual&val])
    shared_images=np.intersect1d(train_images,val_images)
    shared_conversations=np.intersect1d(conversation[~val],conversation[val])
    assert len(shared_images)==0 and len(shared_conversations)==0
    out.mkdir()
    for name,mask in [('train',~val),('val',val),('visual_train',visual&~val),('visual_val',visual&val)]:
        np.save(out/f'{name}.npy',np.flatnonzero(mask).astype('<u8'))
    np.save(out/'component_root.npy',parent)
    components,multiplicity=np.unique(parent,return_counts=True)
    report={**original,'train_rows':int((~val).sum()),'val_rows':int(val.sum()),
            'visual_train_rows':int((visual&~val).sum()),'visual_val_rows':int((visual&val).sum()),
            'train_visual_assets':len(train_images),'val_visual_assets':len(val_images),
            'shared_visual_assets':0,'shared_conversation_hashes':0,'components':len(components),'largest_component':int(multiplicity.max()),
            'image_union_edges':image_edges,'conversation_union_edges':conversation_edges,
            'changed_from_original_val_membership':int((val!=(conversation%1000<1)).sum()),
            'split':'Connected components of exact encoded image SHA256 for user-image rows plus all exact conversation hash64 groups; canonical hash of minimum raw-row representative modulo1000 <1 is held out',
            'limitations':'Exact assets only, no perceptual/semantic deduplication; text-only rows have no visual marker and their unused thumbnail is not an input asset; language pretraining/SFT may contain related knowledge',
            'original_split':'audit_v1 unchanged; old preflights and original held-out scores remain tied to that split',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/omni_i2t_grouped_split_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
