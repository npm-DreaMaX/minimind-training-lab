#!/usr/bin/env python3
"""Inspect actual rendered examples from the highest-frequency content groups."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    root=ROOT/a.corpus; out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve previous inspection')
    hashes=np.load(root/'content_hash64.npy',mmap_mode='r')
    unique,first,counts=np.unique(hashes,return_index=True,return_counts=True)
    offsets=np.load(root/'offsets.npy',mmap_mode='r'); tokens=np.memmap(root/'tokens.bin',dtype='<u2',mode='r')
    labels=np.memmap(root/'labels.bin',dtype='<i2',mode='r'); tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    examples=[]
    for i in np.argsort(counts)[-20:][::-1]:
        row=int(first[i]); start,end=map(int,offsets[row:row+2]); selected=labels[start:min(end,start+1536)]
        examples.append(dict(hash64=int(unique[i]),multiplicity=int(counts[i]),first_row=row,full_length=end-start,
                        supervised_labels_at_1536=int((selected[1:]!=-100).sum()),
                        rendered_prefix=tokenizer.decode(tokens[start:min(end,start+1200)].tolist(),skip_special_tokens=False)))
    report=dict(corpus=a.corpus,rows=len(hashes),unique_conversations=len(unique),extra_repeated_records=len(hashes)-len(unique),
                highest_frequency_groups=examples,scope='Content hashes group canonical raw conversations before augmentation; shown text is the first cached rendered example, not a reconstruction of every raw record',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time())
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),extra_repeated_records=report['extra_repeated_records'],max_multiplicity=int(counts.max()))),flush=True)

if __name__=='__main__': main()
