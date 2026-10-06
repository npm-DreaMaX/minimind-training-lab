#!/usr/bin/env python3
"""CPU-only inspection of the archived dLM paired with the pinned Dense AR."""
import argparse,hashlib,importlib.util,json,resource,sys,time,types
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'upstream'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from lab.data import TokenCorpus

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=ROOT/'sources/experimental/dlm/model_dlm_author.py'
    (out/'source_inspection.py').write_bytes(Path(__file__).read_bytes())
    spec=importlib.util.spec_from_file_location('archived_dlm',source);dlm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dlm)
    torch.set_num_threads(2);torch.manual_seed(20261011)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    config=dict(hidden_size=768,num_hidden_layers=8,use_moe=False)
    ar=MiniMindForCausalLM(MiniMindConfig(**config)).eval()
    model=dlm.MiniMindForMaskedDiffusion(dlm.MiniMindDLLMConfig(**config)).eval()
    incompatible=model.load_state_dict(ar.state_dict(),strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    count=sum(x.numel() for x in model.parameters())
    # Changing a future token must influence earlier positions only for the
    # bidirectional model. Avoid padding to exercise the SDPA path too.
    ids=torch.tensor([[1,71,83,101,55,81,2]])
    changed=ids.clone();changed[0,5]=95
    with torch.no_grad():
        ar_delta=float((ar(ids).logits[:,0]-ar(changed).logits[:,0]).abs().max())
        dlm_delta=float((model(ids).logits[:,0]-model(changed).logits[:,0]).abs().max())
        masked=torch.ones_like(ids);masked[:,5]=0
        excluded_delta=float((model(ids,attention_mask=masked).logits[:,0]-model(changed,attention_mask=masked).logits[:,0]).abs().max())
    assert ar_delta==0 and dlm_delta>1e-5 and excluded_delta==0,(ar_delta,dlm_delta,excluded_delta)
    corpus=TokenCorpus(ROOT/'data/processed/sft_t2t_full_v1','val',128,'sft')
    chosen=[]
    for index in range(len(corpus)):
        x,y=corpus[index]
        if int((y!=-100).sum())>=12:
            chosen.append((index,x,y))
            if len(chosen)==2:break
    assert len(chosen)==2
    x=torch.stack([r[1] for r in chosen]);y=torch.stack([r[2] for r in chosen]);eligible=y!=-100
    noisy,corruption,prob=model.add_noise_to_tokens(x,torch.tensor([.25,.75]),pad_token_id=tokenizer.pad_token_id)
    corruption &= eligible;noisy=torch.where(eligible,noisy,x)
    assert torch.equal(noisy[~eligible],x[~eligible]);assert not corruption[x==0].any()
    result=model(noisy,attention_mask=(x!=0).long(),labels=x,corruption_mask=corruption,p_mask=prob,n_valid=eligible.sum())
    raw=F.cross_entropy(result.logits.flatten(0,1),x.flatten(),reduction='none').reshape_as(x)
    expected=(raw[corruption]/prob[corruption]).sum()/eligible.sum()
    torch.testing.assert_close(result.loss,expected,atol=0,rtol=0)
    result.logits.retain_grad();result.loss.backward()
    nonmasked_grad=float(result.logits.grad[~corruption].abs().max())
    gate_grad=float(model.model.layers[0].self_attn.q_proj.weight.grad.norm())
    assert nonmasked_grad==0 and gate_grad>0
    shape=list(result.logits.shape);loss=float(result.loss.detach())
    del result,raw,expected;model.zero_grad(set_to_none=True)
    with torch.no_grad():
        empty=model(ids,labels=ids,corruption_mask=torch.zeros_like(ids,dtype=torch.bool),p_mask=torch.ones_like(ids,dtype=torch.float32),n_valid=torch.tensor(0)).loss
    # Controlled generation counterexample: if the model proposes the mask
    # token itself, selecting it as 'unmasked' makes no actual progress.
    original_forward=model.forward
    def mask_only(self,input_ids,**kwargs):
        logits=torch.zeros(*input_ids.shape,self.config.vocab_size);logits[...,self.config.mask_token_id]=10
        return types.SimpleNamespace(logits=logits)
    model.forward=types.MethodType(mask_only,model)
    generated=model.generate(ids[:,:2],max_new_tokens=4,steps=4,temperature=0,top_k=0)
    model.forward=original_forward
    remaining=int((generated[:,2:]==model.config.mask_token_id).sum())
    report=dict(scope='Full official-size Dense AR/dLM CPU forward/backward and controlled author-code failure cases; random identical weights; no optimizer update or formal dLM training',
        parameters=count,strict_ar_weight_compatibility=True,model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ar_source_sha256=hashlib.sha256((ROOT/'upstream/model/model_minimind.py').read_bytes()).hexdigest(),
        mask_token=dict(id=model.config.mask_token_id,token=tokenizer.convert_ids_to_tokens(model.config.mask_token_id)),
        attention=dict(causal_future_edit_max_difference=ar_delta,bidirectional_future_edit_max_difference=dlm_delta,excluded_future_edit_max_difference=excluded_delta),
        real_sft=dict(raw_rows=[int(corpus.rows[r[0]]) for r in chosen],input_shape=list(x.shape),logits_shape=shape,
            eligible_targets=int(eligible.sum()),masked_targets=int(corruption.sum()),p_mask_per_sample=prob[:,0].tolist(),
            weighted_same_position_loss=loss,unmasked_logits_gradient_max=nonmasked_grad,q_proj_gradient_norm=gate_grad,
            mask_positions=corruption.nonzero().tolist()),
        author_limitations=dict(zero_valid_targets_loss=str(float(empty)),mask_token_generation_counterexample_remaining_masks=remaining,
            moe_aux='Author forward computes backbone aux but returns only MaskedLMOutput loss without aux; Dense unaffected; do not silently treat optional MoE as the AR CE+aux recipe',
            cache='Bidirectional denoising changes positions together; ordinary AR KV-cache reuse across denoising steps is not justified'),
        inspection_checks_passed=True,elapsed_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
