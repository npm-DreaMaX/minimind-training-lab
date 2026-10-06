#!/usr/bin/env python3
"""Compare BOS vs bare prefixes on CPU, leaving the formal GPU job running."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--weight',required=True); p.add_argument('--out',required=True); a=p.parse_args()
torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('official_mainline',ROOT/'upstream/model/model_minimind.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
weight=ROOT/a.weight; before=weight.stat()
state=torch.load(weight,map_location='cpu',weights_only=True)
model=module.MiniMindForCausalLM(module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True)).eval()
model.load_state_dict(state,strict=True); del state
assert before.st_ino==weight.stat().st_ino and before.st_mtime_ns==weight.stat().st_mtime_ns,'Checkpoint changed during load'
sha=hashlib.sha256()
with weight.open('rb') as f:
    while block:=f.read(8*1024*1024): sha.update(block)
tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
record={'weight':a.weight,'sha256':sha.hexdigest(),'source':str(weight.with_suffix('.json')),
        'device':'CPU','parameter_dtype':'FP32 loaded from FP16 export',
        'purpose':'Within-CPU controlled input-prefix comparison; not a bitwise comparison against BF16 GPU decoding',
        'results':[]}
for prompt in ['人工智能的主要应用包括','中国的首都是','水的三种状态是']:
    plain=tok(prompt,add_special_tokens=False).input_ids
    for bos in [False,True]:
        ids=torch.tensor([[tok.bos_token_id]+plain if bos else plain],dtype=torch.long)
        start=time.time()
        with torch.inference_mode(): result=model.generate(ids,max_new_tokens=64,do_sample=False,eos_token_id=tok.eos_token_id)
        row={'prompt':prompt,'prepend_bos':bos,'input_ids':ids[0].tolist(),
             'continuation':tok.decode(result[0,len(ids[0]):],skip_special_tokens=False),'seconds':time.time()-start}
        record['results'].append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
