#!/usr/bin/env python3
"""Check trained AR cache math independently of greedy decoding quality, CPU only."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024**2): h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--weight',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    weight=ROOT/a.weight; out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve previous cache evidence')
    torch.set_num_threads(4); source_sha=sha(weight)
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    if sha(weight)!=source_sha: raise RuntimeError('Source changed during load')
    spec=importlib.util.spec_from_file_location('ar_cache_reference',ROOT/'upstream/model/model_minimind.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model=module.MiniMindForCausalLM(module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True)).eval()
    model.load_state_dict(state,strict=True); del state
    tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model'); results=[]
    with torch.inference_mode():
        for text in ['中国的首都是北京。','人工智能的主要应用包括自然语言处理、计算机视觉。']:
            for bos in [False,True]:
                ids=tok(text,return_tensors='pt').input_ids
                if bos: ids=torch.cat([torch.tensor([[tok.bos_token_id]]),ids],dim=1)
                full=model(ids).logits; past=None; slices=[]
                for token in ids.split(1,dim=1):
                    result=model(token,past_key_values=past,use_cache=True); past=result.past_key_values; slices.append(result.logits)
                cached=torch.cat(slices,dim=1); error=(full-cached).abs()
                record=dict(text=text,bos=bos,shape=list(full.shape),max_absolute_error=float(error.max()),
                            relative_rms=float((full-cached).square().mean().sqrt()/full.square().mean().sqrt()),
                            same_argmax=bool(torch.equal(full.argmax(-1),cached.argmax(-1))),
                            passed=bool(torch.allclose(full,cached,atol=3e-5,rtol=3e-5)))
                results.append(record); print(json.dumps(record,ensure_ascii=False),flush=True)
    report=dict(scope='Full 198M trained checkpoint on CPU, full-sequence vs token-by-token cached logits; not a quality score',
                weight=str(weight.relative_to(ROOT)),weight_sha256=source_sha,tests=results,passed=all(r['passed'] for r in results),time=time.time())
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert report['passed'],report
if __name__=='__main__': main()
