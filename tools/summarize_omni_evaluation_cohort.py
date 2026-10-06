#!/usr/bin/env python3
"""Recompute a token-weighted cohort metric from retained per-record losses."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True)
    p.add_argument('--indices',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-partial',action='store_true',help='Explicitly permit a periodic validation subset; record missing rows')
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise RuntimeError('Preserve previous cohort report')
    evaluation=Path(a.evaluation);index=Path(a.indices)
    expected=np.load(index);wanted=set(map(int,expected));seen=set();records=[]
    assert len(wanted)==len(expected),'Duplicate requested rows'
    with evaluation.open() as stream:
        for line in stream:
            row=json.loads(line);raw=row['raw_row']
            if raw in seen:raise ValueError('Duplicate evaluation row')
            seen.add(raw)
            if raw in wanted:records.append(row)
    missing=wanted-seen
    if missing and not a.allow_partial:raise ValueError(f'{len(missing)} requested rows absent; this is not a complete cohort evaluation')
    if not records:raise ValueError('No evaluated cohort rows')
    tn=sum(x['text_count'] for x in records);ts=sum(x['text_nll_sum'] for x in records)
    an=np.asarray([x['audio_count'] for x in records],dtype=np.int64).sum(0)
    asum=np.asarray([x['audio_weighted_nll_sum'] for x in records],dtype=np.float64).sum(0)
    audio=asum/np.maximum(1,an);text=ts/max(1,tn)
    report=dict(evaluated_rows=len(records),requested_rows=len(wanted),missing_rows=len(missing),
        complete_cohort=not missing,text_ce=text,audio_codebook_ce=audio.tolist(),audio_ce=float(audio.mean()),
        selection_score=text+float(audio.mean()),text_labels=tn,audio_labels=an.tolist(),
        stop_count=np.asarray([x['stop_count'] for x in records]).sum(0).tolist(),
        stop_correct=np.asarray([x['stop_correct'] for x in records]).sum(0).tolist(),
        evaluation=str(evaluation),evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        indices=str(index),indices_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        protocol='Sum per-record NLL numerator and valid label counts, then divide within each head; audio STOP weighted 10x; no aux; same forward pass as original evaluation',
        limitation='Teacher-forced augmented inputs, not clean generation or task accuracy')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
