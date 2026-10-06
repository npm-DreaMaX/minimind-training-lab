#!/usr/bin/env python3
"""Audit full shards and assign conversation-hash splits shared with text SFT."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); a=p.parse_args()
    root=Path(a.corpus); source=json.loads((root/'index.json').read_text()); out=root/'audit_v1'
    if out.exists(): raise RuntimeError('Preserve existing audit; do not overwrite')
    out.mkdir(); rows=source['rows']; start=time.time(); cursor=0
    hashes=np.lib.format.open_memmap(out/'content_hash64.npy',mode='w+',dtype='<u8',shape=(rows,))
    frames=np.lib.format.open_memmap(out/'answer_frames.npy',mode='w+',dtype='<u4',shape=(rows,))
    last_frames=np.lib.format.open_memmap(out/'last_answer_frames.npy',mode='w+',dtype='<u4',shape=(rows,))
    counts=dict(missing_user=0,missing_assistant=0,answer_turn_count_mismatch=0,rows_with_answer_audio=0,
                rows_with_question_audio=0,rows_with_image=0,rows_with_speaker=0)
    total_codes=0
    for shard in source['shards']:
        with pa.memory_map(str(root/shard['file']),'r') as mapping:
            reader=ipc.open_file(mapping); batch=reader.get_batch(0); n=batch.num_rows
            convs=batch.column(batch.schema.get_field_index('conversations')).to_pylist()
            pos=batch.schema.get_field_index('answer_audios')
            answer_counts=np.zeros(n,dtype=np.int64); lengths=np.zeros(n,dtype=np.int64); last=np.zeros(n,dtype=np.int64)
            if pos>=0:
                answers=batch.column(pos); offsets=answers.offsets.to_numpy()
                code_lengths=pc.fill_null(pc.list_value_length(answers.values),0).to_numpy()
                if np.any(code_lengths%8): raise ValueError(f'Incomplete 8-codebook frame: {shard["file"]}')
                sums=np.concatenate(([0],np.cumsum(code_lengths,dtype=np.int64)))
                lengths=sums[offsets[1:]]-sums[offsets[:-1]]; answer_counts=np.diff(offsets)
                present=answer_counts>0; last[present]=code_lengths[offsets[1:][present]-1]
                total_codes+=int(lengths.sum()); counts['rows_with_answer_audio']+=int((lengths>0).sum())
            frames[cursor:cursor+n]=lengths//8; last_frames[cursor:cursor+n]=last//8
            for i,raw in enumerate(convs):
                turns=json.loads(raw); roles=[turn.get('role') for turn in turns]
                counts['missing_user']+=int('user' not in roles); counts['missing_assistant']+=int('assistant' not in roles)
                if lengths[i]: counts['answer_turn_count_mismatch']+=int(roles.count('assistant')!=answer_counts[i])
                canonical=json.dumps(turns,ensure_ascii=False,sort_keys=True)
                hashes[cursor+i]=int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8],'little')
            for column,key in [('question_audios','rows_with_question_audio'),('image_bytes','rows_with_image'),('spk_emb','rows_with_speaker')]:
                pos=batch.schema.get_field_index(column)
                if pos<0: continue
                arr=batch.column(pos)
                if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
                    # Binary values themselves are not decoded/materialized here.
                    if column=='question_audios':
                        child=arr.values
                        sizes=pc.fill_null(pc.binary_length(child),0).to_numpy()
                        sums=np.concatenate(([0],np.cumsum(sizes,dtype=np.int64))); offsets=arr.offsets.to_numpy()
                        counts[key]+=int(((sums[offsets[1:]]-sums[offsets[:-1]])>0).sum())
                    else: counts[key]+=int(pc.sum(pc.greater(pc.fill_null(pc.list_value_length(arr),0),0)).as_py() or 0)
                elif pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
                    counts[key]+=int(pc.sum(pc.greater(pc.fill_null(pc.binary_length(arr),0),0)).as_py() or 0)
            cursor+=n
        if cursor%100000<n: print(json.dumps({'rows':cursor,'elapsed_s':time.time()-start}),flush=True)
    assert cursor==rows,(cursor,rows)
    for array in [hashes,frames,last_frames]: array.flush()
    val=hashes%1000<1
    np.save(out/'train.npy',np.flatnonzero(~val).astype('<u8')); np.save(out/'val.npy',np.flatnonzero(val).astype('<u8'))
    unique,frequency=np.unique(hashes,return_counts=True)
    meta=dict(rows=rows,train_rows=int((~val).sum()),val_rows=int(val.sum()),counts=counts,
              full_answer_codes=total_codes,full_answer_hours=total_codes/8/12.5/3600,
              answer_frames_percentiles={str(q):float(np.percentile(frames,q)) for q in [50,90,95,99,100]},
              last_answer_frames_percentiles={str(q):float(np.percentile(last_frames,q)) for q in [50,90,95,99,100]},
              unique_conversation_hashes=len(unique),extra_exact_conversation_duplicates=int(rows-len(unique)),max_multiplicity=int(frequency.max()),
              split='Same canonical JSON and SHA256 first uint64 LE modulo1000 <1 as text SFT; shared conversations are held out together',
              limitations='Exact conversation grouping only; no claim of semantic decontamination or unseen answer knowledge',
              corpus_index_sha256=hashlib.sha256((root/'index.json').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.time()-start)
    (out/'metadata.json.tmp').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out/'metadata.json.tmp').replace(out/'metadata.json')
    print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
