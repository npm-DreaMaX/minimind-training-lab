#!/usr/bin/env python3
"""Audit exact conversation overlap across planned stages and unseen A2A assets."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='reports/omni_multistage_overlap_v1');args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes());started=time.perf_counter()
    data=ROOT/'data/processed';sets={};summaries={};fingerprints={}
    specs=[('language_sft','sft_t2t_full_v1','','',''),
           ('t2a','sft_t2a_shards_v1','audit_v1','audit_v1',''),
           ('a2a','sft_a2a_shards_v1','audit_v1','audit_v1',''),
           ('i2t_visual_v2','sft_i2t_shards_v1','audit_v1','split_image_v2','visual_'),
           ('i2t_mixed_v2','sft_i2t_shards_v1','audit_v1','split_image_v2','')]
    for name,folder,hash_subdir,split_subdir,prefix in specs:
        root=data/folder;hp=root/hash_subdir/'content_hash64.npy';hashes=np.load(hp,mmap_mode='r')
        fingerprints[str(hp.relative_to(ROOT))]=sha(hp);summaries[name]={}
        for split in ['train','val']:
            ip=root/split_subdir/f'{prefix}{split}.npy';rows=np.load(ip,mmap_mode='r')
            fingerprints[str(ip.relative_to(ROOT))]=sha(ip)
            if name=='language_sft':
                first=np.load(root/'first_targets.npy',mmap_mode='r');rows=rows[first[rows]<1536]
            selected=np.asarray(hashes[rows]);sets[name,split]=selected
            summaries[name][split]=dict(rows=len(selected),unique_conversations=len(np.unique(selected)))
    comparisons=[]
    for j,(later,*_) in enumerate(specs):
        val=sets[later,'val']
        for earlier,*_ in specs[:j]:
            overlapping=np.isin(val,np.unique(sets[earlier,'train']))
            comparisons.append(dict(validation_stage=later,prior_training_stage=earlier,
                overlap_validation_rows=int(overlapping.sum()),overlap_unique_hashes=len(np.unique(val[overlapping])),validation_rows=len(val)))
    # Forgetting evaluations revisit earlier validation sets after later stages.
    # Their contamination must therefore also be checked in the reverse order.
    matrix=[]
    for training,*_ in specs:
        unique_train=np.unique(sets[training,'train'])
        for validation,*_ in specs:
            val=sets[validation,'val'];overlapping=np.isin(val,unique_train)
            matrix.append(dict(validation_stage=validation,training_stage=training,
                overlap_validation_rows=int(overlapping.sum()),overlap_unique_hashes=len(np.unique(val[overlapping])),validation_rows=len(val)))
    # This stricter evaluation subset preserves original training indices and
    # conversation-hash holdout, unlike repartitioning connected components.
    root=data/'sft_a2a_shards_v1';assets_path=ROOT/'reports/omni_a2a_audio_audit_v1/assets.npy'
    audit=json.loads((assets_path.parent/'report.json').read_text());assert audit['header_errors']==0
    assets=np.load(assets_path,mmap_mode='r');old_val=np.load(root/'audit_v1/val.npy')
    n=json.loads((root/'index.json').read_text())['rows'];is_val=np.zeros(n,dtype=bool);is_val[old_val]=True
    asset_val=is_val[assets['row']];train_assets=np.unique(assets['hash64'][~asset_val])
    shared=np.intersect1d(train_assets,np.unique(assets['hash64'][asset_val]))
    affected_rows=np.unique(assets['row'][asset_val & np.isin(assets['hash64'],shared)])
    rows_with_audio=np.unique(assets['row']);has_audio=np.isin(old_val,rows_with_audio)
    unseen=old_val[has_audio & ~np.isin(old_val,affected_rows)]
    missing_audio=old_val[~has_audio]
    subset=root/'evaluation_audio_unseen_v2'
    if subset.exists():
        for name,value in [('val',unseen),('excluded_shared_audio',affected_rows),('excluded_no_audio',missing_audio)]:
            assert np.array_equal(np.load(subset/f'{name}.npy'),value),'Existing audio cohort differs; preserve it and investigate'
    else:
        subset.mkdir()
        np.save(subset/'val.npy',unseen);np.save(subset/'excluded_shared_audio.npy',affected_rows)
        np.save(subset/'excluded_no_audio.npy',missing_audio)
    hashes=np.load(root/'audit_v1/content_hash64.npy',mmap_mode='r')
    unseen_assets=np.unique(assets['hash64'][np.isin(assets['row'],unseen)])
    assert not len(np.intersect1d(unseen_assets,train_assets))
    for earlier in ['language_sft','t2a']:
        assert not np.isin(hashes[unseen],sets[earlier,'train']).any()
    metadata=dict(original_validation_rows=len(old_val),unseen_audio_validation_rows=len(unseen),
        excluded_shared_audio_rows=len(affected_rows),excluded_no_audio_rows=len(missing_audio),
        unseen_input_assets=len(unseen_assets),shared_input_assets=len(shared),training_indices='audit_v1/train.npy unchanged',
        selection='Original conversation-hash held-out rows with at least one valid input audio and no input audio hash64 occurring in the unchanged A2A training split',
        asset_identity=audit['asset_identity'],index_sha256=sha(subset/'val.npy'),asset_manifest_sha256=sha(assets_path),
        limitations='Exact encoded asset hash64 and exact full-conversation hash64 only; no speaker, waveform-perceptual, question-only, answer-only, semantic or pretraining-text decontamination; excludes text-only replay from this audio-specific subset; old full validation remains available')
    if (subset/'metadata.json').exists():assert json.loads((subset/'metadata.json').read_text())==metadata
    else:(subset/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report=dict(stage_sizes=summaries,cross_stage_exact_conversation_overlap=comparisons,
        all_stage_train_validation_overlap_matrix=matrix,
        a2a_unseen_audio=metadata,input_sha256=fingerprints,source_sha256=sha(Path(__file__)),
        seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
