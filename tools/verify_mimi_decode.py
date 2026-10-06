#!/usr/bin/env python3
"""Decode a real dataset code sequence to check rate and checkpoint compatibility."""
import json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import MimiModel
ROOT=Path(__file__).resolve().parents[1]; torch.set_num_threads(4)
path=ROOT/'data/frozen_models/mimi'; assert (path/'verified.json').exists()
model,info=MimiModel.from_pretrained(path,torch_dtype=torch.float32,output_loading_info=True)
assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'],info
model.eval()
table=pq.read_table(ROOT/'artifacts/omni_validation/t2a_first_row_group.parquet')
row=7; codes=table['answer_audios'][row].as_py()[-1]
assert len(codes)%8==0
ids=torch.tensor(codes,dtype=torch.long).reshape(-1,8).T.unsqueeze(0).contiguous()
start=time.time()
with torch.inference_mode(): wave=model.decode(ids).audio_values.squeeze().numpy()
assert np.isfinite(wave).all()
out=ROOT/'artifacts/omni_validation/codec_decode_sample.wav'; sf.write(out,wave,model.config.sampling_rate,subtype='FLOAT')
conversations=json.loads(table['conversations'][row].as_py())
record={'scope':'Frozen Mimi decoding of existing dataset codes, not output of our trained Omni model',
        'dataset_row_in_fixture':row,'text':conversations[-1]['content'],'code_shape':list(ids.shape),
        'sampling_rate':model.config.sampling_rate,'configured_frame_rate':model.config.frame_rate,
        'output_samples':len(wave),'output_seconds':len(wave)/model.config.sampling_rate,
        'expected_seconds':ids.shape[-1]/model.config.frame_rate,'peak_amplitude':float(np.abs(wave).max()),
        'seconds_on_cpu':time.time()-start,'loading_info':info,'wav':str(out.relative_to(ROOT))}
assert abs(record['output_seconds']-record['expected_seconds'])<1/model.config.frame_rate,record
(ROOT/'reports/mimi_codec_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
