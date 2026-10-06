#!/usr/bin/env python3
"""One streaming pass: pinned raw JSONL -> untruncated tokens + hash-held-out split.

No full corpus in RAM. Labels use the official SFT span parser. Cache augmentation
is sampled once per record; this deliberate recipe difference is in metadata.
"""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream'))
from dataset.lm_dataset import SFTDataset,pre_processing_chat,post_processing_chat


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True); p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=['pretrain','sft'],required=True)
    p.add_argument('--limit',type=int,default=0,help='Only for pipeline tests; 0 means full corpus')
    args=p.parse_args()
    raw=Path(args.raw)
    if not raw.is_file(): raise FileNotFoundError(f'Raw input is not ready: {raw}')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():
        print('Completed cache already exists:',out,flush=True); return
    if (out/'tokens.bin').exists():
        raise RuntimeError('Incomplete cache exists. Preserve it as a failed attempt before rerun.')
    if not args.limit and not Path(str(raw)+'.verified.json').exists():
        raise RuntimeError('Full corpus must pass the download SHA256 gate before tokenization')
    os.environ['TOKENIZERS_PARALLELISM']='true'
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    helper=object.__new__(SFTDataset)
    helper.tokenizer=tokenizer; helper.max_length=2**31
    helper.bos_id=tokenizer(f'{tokenizer.bos_token}assistant\n',add_special_tokens=False).input_ids
    helper.eos_id=tokenizer(f'{tokenizer.eos_token}\n',add_special_tokens=False).input_ids
    random.seed(20261003)
    offsets=[0]; train=[]; val=[]; lengths=[]; first_targets=[]; content_hashes=[]; empty=0; bad=0; rows=0
    full_tokens=0; supervised=0; started=time.time(); last_report=started
    histogram={str(n):{'cut_rows':0,'tokens_retained':0,'supervised_retained':0,'zero_supervision_rows':0} for n in [384,512,768,1024,1536,2048]}
    f_ids=(out/'tokens.bin').open('wb')
    f_labels=(out/'labels.bin').open('wb') if args.stage=='sft' else None
    f_errors=(out/'rejected.jsonl').open('w')

    def process(batch):
        nonlocal rows,empty,bad,full_tokens,supervised,last_report
        texts=[]; hashes=[]
        for line_no,item in batch:
            try:
                if args.stage=='pretrain':
                    text=str(item['text'])
                    if not text.strip(): empty+=1; continue
                    canonical=text
                else:
                    canonical=json.dumps(item['conversations'],ensure_ascii=False,sort_keys=True)
                    text=post_processing_chat(helper.create_chat_prompt(pre_processing_chat(item['conversations'])))
                texts.append(text)
                hashes.append(int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8],'little'))
            except Exception as e:
                bad+=1; f_errors.write(json.dumps({'line':line_no,'error':repr(e)})+'\n')
        enc=tokenizer(texts,add_special_tokens=False,return_attention_mask=False)['input_ids'] if texts else []
        id_chunks=[]; label_chunks=[]
        for ids,h in zip(enc,hashes):
            if not ids: empty+=1; continue
            labels=helper.generate_labels(ids) if args.stage=='sft' else None
            n=len(ids)
            if min(ids)<0 or max(ids)>=32768: raise ValueError('Token IDs do not fit the audited cache dtypes')
            if labels is not None and not any(v!=-100 for v in labels[1:]):
                empty+=1; continue
            first_target=next((i for i,v in enumerate(labels) if i>0 and v!=-100),2**31-1) if labels is not None else 1
            id_chunks.append(np.asarray(ids,dtype='<u2'))
            if labels is not None: label_chunks.append(np.asarray(labels,dtype='<i2'))
            offsets.append(offsets[-1]+n); lengths.append(n); first_targets.append(first_target); content_hashes.append(h)
            (val if h%1000<1 else train).append(rows)
            rows+=1; full_tokens+=n+(2 if args.stage=='pretrain' else 0)
            supervised+=n+1 if labels is None else sum(v!=-100 for v in labels[1:])
            valid_prefix=np.cumsum(np.asarray(labels[1:])!=-100) if labels is not None else None
            for s,hist in histogram.items():
                s=int(s); orig=n+2 if labels is None else n
                hist['cut_rows']+=int(orig>s); hist['tokens_retained']+=min(orig,s)
                hist['supervised_retained']+=min(n,s-2)+1 if labels is None else int(valid_prefix[min(n-1,s-1)-1])
                if labels is not None and first_target>=s: hist['zero_supervision_rows']+=1
        if id_chunks: f_ids.write(np.concatenate(id_chunks).tobytes())
        if label_chunks: f_labels.write(np.concatenate(label_chunks).tobytes())
        if time.time()-last_report>20:
            progress={'rows':rows,'full_tokens':full_tokens,'elapsed_s':time.time()-started,'raw':str(raw),'time':time.time()}
            (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress),flush=True); last_report=time.time()

    batch=[]
    with raw.open() as f:
        for line_no,line in enumerate(f,1):
            if args.limit and line_no>args.limit: break
            try: batch.append((line_no,json.loads(line)))
            except Exception as e:
                bad+=1; f_errors.write(json.dumps({'line':line_no,'error':repr(e)})+'\n')
            if len(batch)>=512: process(batch); batch=[]
        if batch: process(batch)
    f_ids.close(); f_errors.close()
    if f_labels: f_labels.close()
    np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.uint64))
    np.save(out/'train.npy',np.asarray(train,dtype=np.uint64))
    np.save(out/'val.npy',np.asarray(val,dtype=np.uint64))
    np.save(out/'lengths.npy',np.asarray(lengths,dtype=np.uint32))
    np.save(out/'first_targets.npy',np.asarray(first_targets,dtype=np.uint32))
    np.save(out/'content_hash64.npy',np.asarray(content_hashes,dtype=np.uint64))
    meta={'stage':args.stage,'raw':str(raw),'rows':rows,'train_rows':len(train),'val_rows':len(val),
          'full_input_tokens':full_tokens,'full_supervised_tokens':supervised,'bad_rows':bad,'empty_rows':empty,
          'length_percentiles':{str(x):float(np.percentile(lengths,x)) for x in [50,90,95,99,100]},
          'length_coverage':histogram,'elapsed_s':time.time()-started,'limit':args.limit,
          'split':'SHA256(content) first uint64 little endian modulo 1000 < 1; identical content stays in same split',
          'augmentation':'official SFT augmentation sampled once offline with Python seed 20261003; no fresh augmentation per epoch',
          'tokenizer_sha256':hashlib.sha256((ROOT/'upstream/model/tokenizer.json').read_bytes()).hexdigest()}
    meta['preprocessor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['cache_note']='supervised_retained is before split; pretrain count assumes no internal literal PAD token'
    (out/'metadata.json.tmp').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out/'metadata.json.tmp').replace(out/'metadata.json')
    print(json.dumps(meta,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
