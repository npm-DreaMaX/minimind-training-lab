#!/usr/bin/env python3
"""Export a token-by-token data-flow trace, without running a model or optimizer."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from lab.data import TokenCorpus

p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); p.add_argument('--stage',choices=['pretrain','sft'],required=True)
p.add_argument('--row',type=int,default=0); p.add_argument('--length',type=int,default=512); p.add_argument('--out',required=True)
a=p.parse_args(); data=TokenCorpus(ROOT/a.corpus,'train',a.length,a.stage)
tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model'); x,y=data[a.row]
records=[]
for t in range(len(x)-1):
    label=int(y[t+1]); token=int(x[t])
    records.append({'logit_position_t':t,'input_id_at_t':token,'raw_token_at_t':tok.convert_ids_to_tokens(token),
                    'label_at_t_plus_1':label,'contributes_to_ce':label!=-100,
                    'target_raw_token':tok.convert_ids_to_tokens(label) if label!=-100 else '',
                    'input_is_padding':token==tok.pad_token_id})
out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
with (out/'token_trace.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
meta={'stage':a.stage,'corpus':a.corpus,'training_index':a.row,'original_cache_row':int(data.rows[a.row]),
      'input_shape':[1,a.length],'labels_shape':[1,a.length],'logits_shape_if_model_runs':[1,a.length,6400],
      'input_dtype':str(x.dtype),'input_nonpadding':int((x!=tok.pad_token_id).sum()),
      'supervised_next_tokens':int((y[1:]!=-100).sum()),
      'decoded_nonpad_input':tok.decode(x[x!=tok.pad_token_id],skip_special_tokens=False),
      'note':'CSV position t predicts labels[t+1]; -100 masks direct CE, not the entire computation graph. Raw BPE tokens are not necessarily whole Chinese characters.'}
(out/'sample.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k!='decoded_nonpad_input'},ensure_ascii=False))
