#!/usr/bin/env python3
"""Prepare a complete target checkpoint plus auditable key-level provenance on CPU."""
import argparse,hashlib,importlib,importlib.util,json,sys,time,types,warnings
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.transfer import transfer_ar_weights

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024**2): h.update(chunk)
    return h.hexdigest()

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def main():
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['hybrid','omni'],required=True)
    p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261005); a=p.parse_args()
    source=Path(a.source); out=Path(a.out)
    if out.exists(): raise RuntimeError('Preserve existing transfer artifacts; choose a new directory')
    if not source.exists(): raise FileNotFoundError(source)
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    source_sha=sha(source)
    state=torch.load(source,map_location='cpu',weights_only=True)
    if sha(source)!=source_sha: raise RuntimeError('Source file changed while loading; use a stable checkpoint')
    if not all(isinstance(x,torch.Tensor) for x in state.values()): raise ValueError('Expected a plain AR model export, not an optimizer/resume payload')
    # Validate every source key and shape against the full pinned 198M AR model.
    ar=load_file('transfer_source_ar',ROOT/'upstream/model/model_minimind.py')
    with torch.device('meta'):
        reference=ar.MiniMindForCausalLM(ar.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True))
    reference.load_state_dict(state,strict=True,assign=True); del reference
    if a.kind=='hybrid':
        model_file=ROOT/'models/02_hybrid_moe/src/model_hybrid.py'
        module=load_file('transfer_target_hybrid',model_file)
        model=module.MiniMindForCausalLM(module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,require_fla=True))
        expected=205623072
    else:
        package=types.ModuleType('transfer_official_omni'); package.__path__=[str(ROOT/'sources/minimind-o/model')]
        sys.modules[package.__name__]=package; module=importlib.import_module(package.__name__+'.model_omni')
        model_file=Path(module.__file__)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=module.MiniMindOmni(module.OmniConfig(hidden_size=768,num_hidden_layers=8,use_moe=True),
                                     audio_encoder_path='/nonexistent/transfer_does_not_need_input_encoders',vision_model_path=None)
        expected=314887938
    report=transfer_ar_weights(model,state,a.kind)
    assert report['total_unique_parameters']==expected,report['total_unique_parameters']
    out.mkdir(parents=True)
    weight=out/'initialized_fp32.pth'
    torch.save(model.state_dict(),weight)
    report.update(source=str(source),source_sha256=source_sha,target_file=str(model_file.relative_to(ROOT)),
                  ar_source_sha256=sha(ROOT/'upstream/model/model_minimind.py'),
                  tokenizer_sha256=sha(ROOT/'upstream/model/tokenizer.json'),
                  target_source_sha256=sha(model_file),output=str(weight),output_sha256=sha(weight),seed=a.seed,
                  created=time.time(),precision='source export values loaded into FP32 target; new modules initialized in FP32; new stage requires new optimizer')
    (out/'transfer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
