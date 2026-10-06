#!/usr/bin/env python3
"""Inspect one remote T2A row group to validate code ranges before any GPU loss."""
import json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from inspect_omni_parquet import RemoteFooter,manifest
name='sft_t2a.parquet'; remote=RemoteFooter(name,manifest['files'][name]['bytes'])
parquet=pq.ParquetFile(remote)
table=parquet.read_row_group(0,columns=['conversations','answer_audios'],use_threads=False)
out=ROOT/'artifacts/omni_validation'; out.mkdir(parents=True,exist_ok=True)
pq.write_table(table,out/'t2a_first_row_group.parquet')
total=0; minimum=10**9; maximum=-1; rows=[]; turns=[]
for index in range(len(table)):
    answers=table['answer_audios'][index].as_py(); turns.append(len(answers))
    for audio in answers:
        if not audio: continue
        codes=np.asarray(audio,dtype=np.int64); total+=len(codes)
        minimum=min(minimum,int(codes.min())); maximum=max(maximum,int(codes.max()))
    if index<8:
        conv=json.loads(table['conversations'][index].as_py())
        rows.append({'index':index,'conversation_turns':len(conv),'answer_audio_turns':len(answers),
                     'audio_code_lengths':[len(a) for a in answers],
                     'last_assistant_text':next((x['content'][:160] for x in reversed(conv) if x['role']=='assistant'),'')})
record={'file':name,'revision':manifest['revision'],'scope':'first row group, not a representative whole-file sample',
        'rows':len(table),'network_bytes':remote.bytes_read,'min_code':minimum,'max_code':maximum,
        'total_audio_codes':total,'answer_turns_percentiles':{str(p):float(np.percentile(turns,p)) for p in [50,90,100]},
        'audio_hours_if_8_codebooks_12_5_hz':total/8/12.5/3600,
        'audio_hours_if_8_codebooks_75_hz':total/8/75/3600,'examples':rows}
(ROOT/'reports/omni_t2a_sample_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
assert 0<=minimum and maximum<2048,'Dataset code range incompatible with the configured Mimi vocabulary'
