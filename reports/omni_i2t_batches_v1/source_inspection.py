#!/usr/bin/env python3
"""Real image/text Dataset inspection across sequence lengths, CPU only."""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,SiglipImageProcessor
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.omni_data import ReproducibleOmniDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--samples',type=int,default=256)
    args=p.parse_args(); out=ROOT/args.out
    if out.exists(): raise RuntimeError('Preserve previous image inspection')
    out.mkdir(parents=True); shutil.copy2(__file__,out/'source_inspection.py'); start=time.perf_counter(); torch.set_num_threads(2)
    corpus=ROOT/'data/processed/sft_i2t_shards_v1'; train=np.load(corpus/'audit_v1/train.npy',mmap_mode='r')
    selected=np.sort(np.random.default_rng(20261011).choice(train,size=args.samples,replace=False))
    processor=SiglipImageProcessor.from_pretrained(ROOT/'data/frozen_models/siglip2-base-p32-256-ve')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model'); records=[]; summaries={}
    for length in [768,1536,3072]:
        dataset=ReproducibleOmniDataset(str(corpus),tokenizer,vision_processor=processor,max_length=length,
                                        rows=selected,seed=20261006,scheduled_sampling=.05)
        for epoch in [0,1]:
            current=[]
            with (out/f'length_{length}_epoch_{epoch}.jsonl').open('w') as stream:
                for i,row in enumerate(selected):
                    ids,text,audio,fbank,flen,pixels,spk=dataset[(epoch,i)]
                    if isinstance(pixels,dict): pixel_tensor=pixels['pixel_values']
                    else: pixel_tensor=pixels
                    r=dict(row=int(row),epoch=epoch,max_length=length,ids_shape=list(ids.shape),
                           image_markers=int((ids[8]==12).sum()),text_labels=int((text!=-100).sum()),audio_labels=int((audio!=-100).sum()),
                           pixel_shape=list(pixel_tensor.shape) if pixel_tensor is not None else None,
                           nonzero_pixels=bool(pixel_tensor is not None and pixel_tensor.any()),input_nonpad=int((ids[8]!=0).sum()))
                    current.append(r); stream.write(json.dumps(r)+'\n')
            summaries[f'{length}/epoch{epoch}']=dict(rows=len(current),zero_supervision=sum(r['text_labels']==0 for r in current),
                missing_image_markers=sum(r['image_markers']!=64 for r in current),empty_pixels=sum(not r['nonzero_pixels'] for r in current),
                text_labels=sum(r['text_labels'] for r in current),rows_filling_context=sum(r['input_nonpad']==length-1 for r in current),
                label_percentiles={str(q):float(np.percentile([r['text_labels'] for r in current],q)) for q in [50,90,99,100]})
            if length==1536 and epoch==0: records=current
    chosen=[r['row'] for r in records if r['text_labels'] and r['image_markers']==64 and r['nonzero_pixels']][:32]
    report=dict(summaries=summaries,random_gpu_preflight_rows=chosen,seconds=time.perf_counter()-start,
                scope='Uniform train rows, two deterministic augmentations, actual image decoding/processor/labels; no encoder or LLM forward; sampled coverage only')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
