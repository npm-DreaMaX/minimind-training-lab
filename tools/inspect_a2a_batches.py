#!/usr/bin/env python3
"""CPU inspection of real A2A augmentation, input frames, markers and labels."""
import argparse,gc,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.omni_batch import official_modules
from lab.omni_data import ReproducibleOmniDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--max-length',type=int,default=1536); p.add_argument('--samples',type=int,default=48)
    a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve previous A2A inspection')
    out.mkdir(parents=True); shutil.copy2(__file__,out/'source_inspection.py')
    torch.set_num_threads(2); start=time.time(); module,_=official_modules()
    encoder,processor=module.MiniMindOmni.load_sensevoice(str(ROOT/'data/frozen_models/SenseVoiceSmall'))
    assert encoder is not None and processor is not None
    del encoder; gc.collect()
    corpus=ROOT/'data/processed/sft_a2a_shards_v1'; train=np.load(corpus/'audit_v1/train.npy',mmap_mode='r')
    durations=np.load(ROOT/'reports/omni_a2a_audio_audit_v1/max_question_seconds.npy',mmap_mode='r')
    random_rows=np.random.default_rng(20261008).choice(train,size=a.samples,replace=False)
    tail_rows=train[np.argsort(durations[train])[-16:]]; rows=np.unique(np.concatenate([random_rows,tail_rows]))
    dataset=ReproducibleOmniDataset(str(corpus),AutoTokenizer.from_pretrained(ROOT/'upstream/model'),audio_processor=processor,
                                    max_length=a.max_length,rows=rows,seed=20261006,scheduled_sampling=.05)
    records=[]
    with (out/'batches.jsonl').open('w') as stream:
        for epoch in [0,1]:
            for i,row in enumerate(rows):
                begin=time.time(); ids,text,audio,fbank,length,pixels,speaker=dataset[(epoch,i)]
                record=dict(row=int(row),epoch=epoch,uniform_sample=bool(row in random_rows),stored_max_input_seconds=float(durations[row]),
                      ids_shape=list(ids.shape),fbank_shape=list(fbank.shape) if fbank is not None else None,fbank_length=length,
                      audio_markers=int((ids[8]==16).sum()),text_labels=int((text!=-100).sum()),audio_labels_per_head=(audio!=-100).sum(1).tolist(),
                      stops_per_head=(audio==2050).sum(1).tolist(),speaker_norm=float(speaker.norm()),data_seconds=time.time()-begin)
                records.append(record); stream.write(json.dumps(record)+'\n'); stream.flush()
    usable=[r for r in records if r['epoch']==0 and r['audio_markers']>0 and r['text_labels'] and sum(r['audio_labels_per_head'])]
    random_usable=[r['row'] for r in usable if r['uniform_sample']][:16]
    longest=sorted(usable,key=lambda r:r['fbank_length'],reverse=True)[:4]
    report=dict(max_length=a.max_length,records=len(records),uniform_row_count=len(random_rows),tail_row_count=len(tail_rows),
                zero_supervision_draws=sum(r['text_labels']==0 and sum(r['audio_labels_per_head'])==0 for r in records),
                no_audio_marker_draws=sum(r['audio_markers']==0 for r in records),
                random_gpu_preflight_rows=random_usable,long_gpu_preflight_rows=[r['row'] for r in longest],
                maximum_observed_fbank_length=max(r['fbank_length'] for r in records),
                scope='Full-source uniform sample plus deliberately selected long tail, two epochs of actual official augmentation; only CPU frontend, no encoder or LLM forward; tail mixture is not a population estimate',elapsed_s=time.time()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
