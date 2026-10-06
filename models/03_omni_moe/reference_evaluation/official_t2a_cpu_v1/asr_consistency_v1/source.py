#!/usr/bin/env python3
"""ASR transcription and character error versus generated Thinker text.

This measures an ASR proxy for speech/text consistency, not answer correctness,
voice naturalness, or a replacement for listening. Truncated audio is flagged.
"""
import argparse,hashlib,json,re,sys,time,unicodedata
from pathlib import Path
import torch
from funasr import AutoModel
ROOT=Path(__file__).resolve().parents[1]

def normalize(text):
    text=re.sub(r'<\|[^>]+\|>','',text)
    return ''.join(c for c in unicodedata.normalize('NFKC',text).lower() if c.isalnum())

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1): row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024**2): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation');a=p.parse_args(); run=ROOT/a.evaluation; out=run/'asr_consistency_v1'
    if out.exists(): raise RuntimeError('Preserve prior ASR evaluation')
    out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    rows=[json.loads(l) for l in (run/'outputs.jsonl').read_text().splitlines()]
    torch.set_num_threads(2)
    model=AutoModel(model=str(ROOT/'data/frozen_models/SenseVoiceSmall'),trust_remote_code=True,disable_update=True,
                    device='cpu',ncpu=2,disable_pbar=True,disable_log=True)
    records=[]
    for row in rows:
        base=dict(id=row['id'],thinker_text=row['text'],generation_reached_budget=row['reached_step_budget'])
        if row['audio']['empty_audio']: records.append(dict(**base,skipped='No emitted audio'));continue
        wav=ROOT/row['audio']['wav']; started=time.perf_counter()
        result=model.generate(input=str(wav),language='auto',use_itn=True,batch_size=1)
        transcript=' '.join(x.get('text','') for x in result); reference=normalize(row['text']);hypothesis=normalize(transcript)
        edits=distance(reference,hypothesis)
        record=dict(**base,asr_result=result,transcript=transcript,reference_normalized=reference,asr_normalized=hypothesis,
                    edit_distance=edits,reference_characters=len(reference),character_error_rate=edits/len(reference) if reference else None,
                    wav_sha256=sha(wav),seconds=time.perf_counter()-started)
        records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    report=dict(records=records,normalization='Strip SenseVoice tags, Unicode NFKC, lowercase, keep only letters/numbers; English is character error, not word error',
                scope='ASR proxy for generated speech versus generated text; not task correctness or naturalness; inspect budget truncation separately',
                asr_verified_manifest=json.loads((ROOT/'data/frozen_models/SenseVoiceSmall/model.pt.verified.json').read_text()),
                source_evaluation_sha256=sha(run/'outputs.jsonl'),script_sha256=sha(Path(__file__)),time=time.time())
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
