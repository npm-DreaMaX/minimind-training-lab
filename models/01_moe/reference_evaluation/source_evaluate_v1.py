#!/usr/bin/env python3
"""Pinned prospective probes; save every output and avoid claiming benchmark quality."""
import argparse,contextlib,fcntl,gc,hashlib,importlib.util,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]

def score(case,text):
    clean=text.strip()
    if case['kind']=='exact': return clean==case['expected']
    if case['kind']=='json':
        try: parsed=json.loads(clean)
        except json.JSONDecodeError: return False
        return parsed==case['expected']
    return None

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024**2): digest.update(chunk)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--weights',required=True)
    p.add_argument('--stage',choices=['pretrain','sft'],required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--suite',default='evaluation/text_probes_v1.json'); a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve prior evaluation; use a versioned destination')
    if a.device=='cuda':
        lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); cfg=json.loads((ROOT/a.config).read_text()); suite=json.loads((ROOT/a.suite).read_text())
    cases=suite[a.stage][:a.limit] if a.limit else suite[a.stage]
    torch.set_num_threads(2 if a.device=='cpu' else 4); torch.manual_seed(20261008); torch.backends.cuda.matmul.allow_tf32=False
    model_path=ROOT/cfg['model_file']; spec=importlib.util.spec_from_file_location('evaluated_model',model_path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model=module.MiniMindForCausalLM(module.MiniMindConfig(**cfg['model']))
    weight=ROOT/a.weights; weight_sha=sha(weight)
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True); model.load_state_dict(state,strict=True); del state; gc.collect()
    if weight_sha!=sha(weight): raise RuntimeError('Weights changed during load')
    model=model.to(a.device).eval(); tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    protocol=dict(stage=a.stage,device=a.device,parameter_dtype='float32',autocast='bfloat16' if a.device=='cuda' else 'disabled',
                  pretrain_prefix='explicit tokenizer BOS',sft_prefix='official chat template with generation prompt',
                  do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,use_cache=True)
    records=[]
    try:
        for case in cases:
            if a.stage=='pretrain': rendered=tokenizer.bos_token+case['prompt']
            else: rendered=tokenizer.apply_chat_template(case.get('messages',[{'role':'user','content':case.get('prompt','')}]),tokenize=False,add_generation_prompt=True)
            ids=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to(a.device)
            started=time.perf_counter()
            with torch.inference_mode(),(torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext()):
                output=model.generate(ids,max_new_tokens=case['max_new_tokens'],do_sample=False,temperature=1.,top_p=1.,top_k=0,
                                      repetition_penalty=1.,use_cache=True,eos_token_id=tokenizer.eos_token_id)
            generated=output[0,ids.shape[1]:].tolist(); plain=tokenizer.decode(generated,skip_special_tokens=True)
            pairs=list(zip(generated,generated[1:])); row=dict(id=case['id'],case=case,rendered_input=rendered,input_tokens=ids.numel(),
                  output_ids=generated,text=plain,text_with_special_tokens=tokenizer.decode(generated,skip_special_tokens=False),
                  passed=score(case,plain),seconds=time.perf_counter()-started,
                  ended_with_eos=bool(generated and generated[-1]==tokenizer.eos_token_id),
                  distinct_token_bigrams=len(set(pairs))/len(pairs) if pairs else None)
            records.append(row)
            with (out/'outputs.jsonl').open('a') as stream: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(dict(id=case['id'],passed=row['passed'],seconds=row['seconds'],output_tokens=len(generated))),flush=True)
    finally:
        scored=[row for row in records if row['passed'] is not None]
        report=dict(scope=suite['scope'],protocol=protocol,config_path=a.config,config=cfg,weights=a.weights,weights_sha256=weight_sha,
                    model_sha256=sha(model_path),suite_sha256=sha(ROOT/a.suite),script_sha256=sha(Path(__file__)),
                    tokenizer_sha256=sha(ROOT/'upstream/model/tokenizer.json'),
                    complete=len(records)==len(cases),partial_suite=len(cases)<len(suite[a.stage]),cases=len(records),scored_cases=len(scored),
                    exact_or_json_passes=sum(row['passed'] for row in scored),parameters=sum(p.numel() for p in model.parameters()),time=time.time())
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
