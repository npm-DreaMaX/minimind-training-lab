#!/usr/bin/env python3
"""Measure actual pinned T2A Dataset labels across a full-corpus random sample.

Audio frames in storage do not equal supervised frames after turn sampling,
text prefixes, codebook delay and truncation. This audit uses the real Dataset.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.omni_data import ReproducibleOmniDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); p.add_argument('--out',required=True)
    p.add_argument('--samples',type=int,default=2048); p.add_argument('--lengths',type=int,nargs='+',default=[768,1536])
    a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve previous label audit')
    out.mkdir(parents=True); torch.set_num_threads(2); start=time.time()
    corpus=ROOT/a.corpus; meta=json.loads((corpus/'audit_v1/metadata.json').read_text())
    assert meta['counts']['rows_with_answer_audio']==meta['rows'] and meta['counts']['rows_with_question_audio']==0
    candidates=np.load(corpus/'audit_v1/train.npy',mmap_mode='r')
    generator=np.random.default_rng(20261007)
    rows=np.sort(generator.choice(candidates,size=min(a.samples,len(candidates)),replace=False))
    np.save(out/'rows.npy',rows)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model'); summaries={}
    for length in a.lengths:
        dataset=ReproducibleOmniDataset(str(corpus),tokenizer,max_length=length,rows=rows,seed=20261006,scheduled_sampling=.05)
        for epoch in [0,1]:
            records=[]
            with (out/f'labels_L{length}_epoch{epoch}.jsonl').open('w') as log:
                for i,row in enumerate(rows):
                    ids,text,audio,*_=dataset[(epoch,i)]
                    record=dict(row=int(row),epoch=epoch,text_labels=int((text!=-100).sum()),
                                audio_labels_per_head=(audio!=-100).sum(dim=1).tolist(),
                                stop_labels_per_head=(audio==2050).sum(dim=1).tolist(),
                                nonpad_text_inputs=int((ids[8]!=tokenizer.pad_token_id).sum()))
                    records.append(record); log.write(json.dumps(record)+'\n')
            text=np.asarray([r['text_labels'] for r in records]); audio=np.asarray([r['audio_labels_per_head'] for r in records]); stops=np.asarray([r['stop_labels_per_head'] for r in records])
            summaries[f'{length}/epoch{epoch}']=dict(rows=len(records),zero_text_rows=int((text==0).sum()),
                zero_audio_rows=int((audio.sum(1)==0).sum()),zero_both_rows=int(((text==0)&(audio.sum(1)==0)).sum()),
                rows_with_all_eight_stops=int((stops==1).all(1).sum()),rows_with_any_missing_stop=int((stops==0).any(1).sum()),
                supervised_text_tokens=int(text.sum()),supervised_audio_codes=int(audio.sum()),
                audio_labels_per_head=audio.sum(0).tolist(),
                head0_target_percentiles={str(q):float(np.percentile(audio[:,0],q)) for q in [0,50,90,99,100]},
                text_input_padding_fraction=1-sum(r['nonpad_text_inputs'] for r in records)/(len(records)*(length-1)))
            print(json.dumps({'length':length,'epoch':epoch,**summaries[f'{length}/epoch{epoch}']},ensure_ascii=False),flush=True)
        del dataset
    result=dict(corpus=a.corpus,rows_sha256=hashlib.sha256((out/'rows.npy').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summaries=summaries,
                scope='Uniform full-training-split sample; exact official per-epoch turn selection and label construction via reproducible wrapper; not full-corpus exact totals',
                elapsed_s=time.time()-start)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
