#!/usr/bin/env python3
"""CPU gradient/optimizer mechanism demonstration; never a formal model run."""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lab.omni_batch import official_modules
from lab.omni_data import ReproducibleOmniDataset
from lab.omni_loss import omni_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    torch.set_num_threads(2);torch.manual_seed(20261013)
    module,collate=official_modules()
    model=module.MiniMindOmni(module.OmniConfig(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,num_talker_hidden_layers=2,use_moe=True),audio_encoder_path='/nonexistent/no_input_audio',vision_model_path=None).train()
    for param in model.parameters():param.requires_grad_(False)
    for param in model.audio_proj.parameters():param.requires_grad_(True)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    data=ReproducibleOmniDataset(data_path=str(ROOT/'artifacts/omni_validation/t2a_shards_v1'),tokenizer=tokenizer,max_length=512,rows=[7],seed=20261013,scheduled_sampling=0.)
    batch=collate([data[0]])
    ids,text,audio,a_inputs,a_lens,pixels,spk=batch
    assert a_inputs is None and (text!=-100).any() and (audio!=-100).any()
    optimizer=torch.optim.AdamW(model.audio_proj.parameters(),lr=1e-4,weight_decay=.01,foreach=False)
    parameter=model.audio_proj.mlp[0].weight
    def inactive_backward():
        result=omni_loss(model(ids,spk_emb=spk),text,audio)
        assert torch.isfinite(result['loss']) and result['loss'].requires_grad
        result['loss'].backward()
        return dict(loss=float(result['loss'].detach()),grad_max=max(float(p.grad.abs().max()) for p in model.audio_proj.parameters() if p.grad is not None),all_have_grad=all(p.grad is not None for p in model.audio_proj.parameters()))
    before=parameter.detach().clone();first=inactive_backward();optimizer.step()
    first['parameter_max_change']=float((parameter-before).abs().max())
    assert first['grad_max']==0 and first['parameter_max_change']>0
    optimizer.zero_grad(set_to_none=True);before=parameter.detach().clone();optimizer.step()
    none=dict(all_grads_none=all(p.grad is None for p in model.audio_proj.parameters()),parameter_max_change=float((parameter-before).abs().max()))
    assert none['all_grads_none'] and none['parameter_max_change']==0
    # Explicit synthetic prior objective establishes nonzero Adam moments.
    optimizer.zero_grad(set_to_none=True);((parameter-.5).square().mean()).backward();optimizer.step()
    optimizer.zero_grad(set_to_none=True);before=parameter.detach().clone();later=inactive_backward();optimizer.step()
    later['parameter_max_change']=float((parameter-before).abs().max())
    assert later['grad_max']==0 and later['parameter_max_change']>first['parameter_max_change']
    report=dict(scope='Small CPU instance of official Omni solely to isolate dummy gradients and AdamW; actual T2A record has text/audio-output supervision but no input audio; no formal model updated',
                model_parameters=sum(p.numel() for p in model.parameters()),input_shape=list(ids.shape),text_labels=int((text!=-100).sum()),audio_labels=int((audio!=-100).sum()),
                first_zero_gradient=first,none_gradient_control=none,zero_gradient_after_synthetic_prior=later,
                tracked_parameter='audio_proj.mlp.0.weight',optimizer=dict(lr=1e-4,weight_decay=.01),
                model_source_sha256=hashlib.sha256((ROOT/'sources/minimind-o/model/model_omni.py').read_bytes()).hexdigest(),time=time.time())
    assert not torch.cuda.is_initialized()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
