#!/usr/bin/env python3
"""One real full198M CPU example: tokens, CE/aux gradients and isolated Adam math.

No update to the loaded model or formal training checkpoint. The Adam example
updates a copy of one router matrix, not a replacement training algorithm.
"""
import argparse,csv,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lab.data import TokenCorpus

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while b:=stream.read(8*1024**2):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=ROOT/a.out
    if out.exists():raise RuntimeError('Preserve prior trace')
    out.mkdir(parents=True);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);torch.manual_seed(20261011);start=time.perf_counter()
    path=ROOT/'upstream/model/model_minimind.py';spec=importlib.util.spec_from_file_location('learning_model',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.MiniMindForCausalLM(module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True)).train()
    weight=ROOT/a.weights;weight_sha=sha(weight);state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state;gc.collect()
    corpus=TokenCorpus(ROOT/'data/processed/pretrain_t2t_full_v1','val',512,'pretrain')
    position=next(i for i in range(100) if 34<=int((corpus[i][0]!=0).sum())<=130)
    ids,labels=corpus[position];ids,labels=ids.unsqueeze(0),labels.unsqueeze(0)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model');captured=[]
    handle=model.model.embed_tokens.register_forward_hook(lambda _m,_i,value:captured.append(value))
    result=model(ids,labels=labels);handle.remove();embedding=captured[0];gate=model.model.layers[0].mlp.gate.weight
    ce_emb,ce_gate=torch.autograd.grad(result.loss,(embedding,gate),retain_graph=True)
    aux_emb,aux_gate=torch.autograd.grad(result.aux_loss,(embedding,gate),retain_graph=True)
    total_emb,total_gate=torch.autograd.grad(result.loss+result.aux_loss,(embedding,gate))
    torch.testing.assert_close(total_gate,ce_gate+aux_gate,atol=1e-7,rtol=1e-5)
    valid=labels[:,1:]!=-100; logits=result.logits.detach()[:,:-1]
    nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction='none').reshape_as(valid)
    torch.testing.assert_close(nll[valid].mean(),result.loss.detach(),atol=1e-6,rtol=1e-6)
    probability=logits.softmax(-1);predicted=logits.argmax(-1)
    ce_norm=ce_emb[0].norm(dim=-1).detach();aux_norm=aux_emb[0].norm(dim=-1).detach()
    pad=ids[0]==0; nonpad=int((~pad).sum())
    assert float(ce_norm[pad].max())==0.,'Causal CE must not depend on future padding tokens'
    assert float(ce_gate.norm())>0.,'Top1 router task gradient unexpectedly absent'
    with (out/'positions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['position','input_id','input_text','target_id','target_text','ce_supervised','nll','target_probability','top1_id','top1_text','embedding_ce_grad_norm','embedding_aux_grad_norm']);writer.writeheader()
        for t in range(ids.shape[1]):
            target=int(labels[0,t+1]) if t<ids.shape[1]-1 else -100;active=target!=-100
            pred=int(predicted[0,t]) if t<ids.shape[1]-1 else None
            writer.writerow(dict(position=t,input_id=int(ids[0,t]),input_text=repr(tokenizer.decode([int(ids[0,t])])),target_id=target,
                target_text=repr(tokenizer.decode([target])) if active else '',ce_supervised=active,nll=float(nll[0,t]) if active else '',
                target_probability=float(probability[0,t,target]) if active else '',top1_id=pred,top1_text=repr(tokenizer.decode([pred])) if pred is not None else '',
                embedding_ce_grad_norm=float(ce_norm[t]),embedding_aux_grad_norm=float(aux_norm[t])))
    # First AdamW step on an isolated copy makes the moment equations observable.
    old=gate.detach().clone();probe=torch.nn.Parameter(old.clone());g=total_gate.detach().clone();probe.grad=g.clone()
    lr,wd,eps=3e-4,.01,1e-8;opt=torch.optim.AdamW([probe],lr=lr,betas=(.9,.999),eps=eps,weight_decay=wd,foreach=False)
    opt.step();expected=old*(1-lr*wd)-lr*g/(g.abs()+eps)
    torch.testing.assert_close(probe.detach(),expected,atol=1e-8,rtol=1e-5)
    torch.testing.assert_close(gate.detach(),old,atol=0,rtol=0)
    report=dict(scope='One short held-out record selected to illustrate padding; full198M CPU FP32 forward and selected gradients. Not representative quality and not a formal optimizer update.',
                weights=a.weights,weights_sha256=weight_sha,source_sha256=sha(path),script_sha256=sha(Path(__file__)),
                corpus_position=position,raw_row=int(corpus.rows[position]),selection='First of first100 held-out records with34..130 non-PAD input tokens',
                parameters=sum(p.numel() for p in model.parameters()),ids_shape=list(ids.shape),logits_shape=list(result.logits.shape),embedding_shape=list(embedding.shape),router_shape=list(gate.shape),
                nonpad_tokens=nonpad,valid_next_token_labels=int(valid.sum()),ce=float(result.loss.detach()),aux=float(result.aux_loss.detach()),
                valid_position_top1_accuracy=float((predicted[valid]==labels[:,1:][valid]).float().mean()),
                router_ce_gradient_norm=float(ce_gate.norm()),router_aux_gradient_norm=float(aux_gate.norm()),router_total_gradient_norm=float(total_gate.norm()),
                router_gradient_addition_max_error=float((total_gate-ce_gate-aux_gate).abs().max()),
                pad_embedding_ce_max_norm=float(ce_norm[pad].max()),pad_embedding_aux_max_norm=float(aux_norm[pad].max()),
                isolated_adam=dict(lr=lr,weight_decay=wd,eps=eps,step=float(opt.state[probe]['step']),formula_max_error=float((probe.detach()-expected).abs().max()),
                                   moment1_max_error=float((opt.state[probe]['exp_avg']-.1*g).abs().max()),moment2_max_error=float((opt.state[probe]['exp_avg_sq']-.001*g.square()).abs().max()),
                                   actual_parameter_update_max=float((probe.detach()-old).abs().max()),loaded_model_unchanged=True,
                                   limitation='Router copy only, new zero moments, no full-model clipping or accumulation; not equivalent to a formal resumed AdamW step'),
                seconds=time.perf_counter()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    active_nll=nll[0].numpy().copy();active_nll[~valid[0].numpy()]=np.nan
    axes[0].plot(active_nll,label='Supervised next-token NLL');axes[0].legend();axes[0].set_ylabel('-log p(target)')
    axes[1].plot(ce_norm.numpy(),label='CE gradient to input embedding');axes[1].plot(aux_norm.numpy(),label='Aux gradient to input embedding')
    axes[1].set_yscale('symlog',linthresh=1e-9);axes[1].legend();axes[1].set_ylabel('Gradient vector norm')
    for ax in axes:ax.axvspan(nonpad-.5,511,color='grey',alpha=.15,label='PAD inputs');ax.set_xlabel('Input token position');ax.grid(alpha=.2)
    fig.suptitle('Real full198M checkpoint, one CPU example: ignored CE labels do not mask MoE auxiliary routing')
    fig.savefig(out/'loss_and_embedding_gradients.png',dpi=150);plt.close(fig)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
