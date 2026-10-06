#!/usr/bin/env python3
"""Inspect all encoded input-audio headers and exact asset overlap by split.

No waveform augmentation or encoder forward is run. Container durations are
pre-augmentation; official speed perturbation can lengthen them by up to 1/0.7.
"""
import argparse,collections,hashlib,io,json,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import soundfile as sf

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    root=Path(a.corpus); out=Path(a.out)
    if out.exists(): raise RuntimeError('Preserve earlier audio audit')
    out.mkdir(parents=True); index=json.loads((root/'index.json').read_text()); start=time.time()
    conversations=np.load(root/'audit_v1/content_hash64.npy',mmap_mode='r'); rows=index['rows']
    maxima=np.zeros(rows,dtype=np.float32); last=np.zeros(rows,dtype=np.float32)
    asset_flags={}; durations={}; records=[]; errors=[]; sample_rates=collections.Counter(); formats=collections.Counter(); speaker_lengths=collections.Counter()
    cursor=0; audio_count=0; nonfinite_speaker=0; incomplete_ref=0
    for shard in index['shards']:
        with pa.memory_map(str(root/shard['file']),'r') as mapping:
            batch=ipc.open_file(mapping).get_batch(0)
            if 'spk_emb' in batch.schema.names:
                emb=batch.column(batch.schema.get_field_index('spk_emb'))
                speaker_lengths.update(pc.fill_null(pc.list_value_length(emb),0).to_pylist())
                nonfinite_speaker+=int(pc.sum(pc.invert(pc.fill_null(pc.is_finite(emb.values),False))).as_py() or 0)
            if 'ref_audios' in batch.schema.names:
                lengths=pc.fill_null(pc.list_value_length(batch.column(batch.schema.get_field_index('ref_audios'))),0).to_numpy()
                incomplete_ref+=int((lengths%8!=0).sum())
            audio=batch.column(batch.schema.get_field_index('question_audios'))
            for i in range(batch.num_rows):
                row=cursor+i; flag=2 if int(conversations[row])%1000<1 else 1
                clips=audio[i].as_py() or []; found=[]
                for turn,blob in enumerate(clips):
                    if not blob:
                        found.append(0.); continue
                    code=int.from_bytes(hashlib.sha256(blob).digest()[:8],'little'); asset_flags[code]=asset_flags.get(code,0)|flag
                    try:
                        if code not in durations:
                            info=sf.info(io.BytesIO(blob)); duration=info.frames/info.samplerate
                            if duration<=0 or not np.isfinite(duration): raise ValueError('Invalid duration')
                            durations[code]=(duration,info.samplerate,info.format,info.channels)
                        duration,rate,fmt,channels=durations[code]
                        sample_rates[rate]+=1; formats[fmt]+=1; audio_count+=1; found.append(duration)
                        records.append((row,turn,code,duration,flag))
                    except Exception as exc:
                        errors.append(dict(row=row,turn=turn,bytes=len(blob),error=repr(exc))); found.append(0.)
                maxima[row]=max(found,default=0.); last[row]=found[-1] if found else 0.
            cursor+=batch.num_rows
        if cursor%50000<batch.num_rows: print(json.dumps(dict(rows=cursor,unique_audio_assets=len(durations),elapsed_s=time.time()-start)),flush=True)
    assert cursor==rows
    array=np.array(records,dtype=[('row','<u8'),('turn','<u4'),('hash64','<u8'),('seconds','<f4'),('split_flag','u1')])
    np.save(out/'assets.npy',array); np.save(out/'max_question_seconds.npy',maxima); np.save(out/'last_question_seconds.npy',last)
    (out/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    valid=maxima[maxima>0]; overlap=[key for key,flag in asset_flags.items() if flag==3]
    val_assets={int(r['hash64']) for r in array if r['split_flag']==2}
    longest=np.argsort(maxima)[-20:][::-1]
    report=dict(corpus=str(root),rows=rows,rows_with_nonempty_input_audio=int((maxima>0).sum()),decoded_headers=audio_count,
                unique_exact_audio_assets=len(durations),asset_identity='SHA256 first uint64 LE of exact encoded bytes; no perceptual/audio-semantic matching',
                exact_assets_present_in_both_splits=len(overlap),unique_validation_assets=len(val_assets),
                validation_assets_also_in_train=len(val_assets.intersection(overlap)),
                input_seconds_percentiles={str(q):float(np.percentile(valid,q)) for q in [50,90,95,99,99.9,100]},
                rows_exceeding_seconds={str(s):int((maxima>s).sum()) for s in [10,30,60,90,120]},
                longest_rows=[dict(row=int(row),max_input_seconds=float(maxima[row]),split='val' if int(conversations[row])%1000<1 else 'train') for row in longest],
                sample_rates=dict(sample_rates),formats=dict(formats),speaker_length_counts=dict(speaker_lengths),
                nonfinite_speaker_values=nonfinite_speaker,incomplete_reference_frames=incomplete_ref,header_errors=len(errors),
                scope='Full stored input-audio headers, not post-augmentation lengths or full waveform-decoding validation; shared input assets may occur in different conversations',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.time()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
