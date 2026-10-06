#!/usr/bin/env python3
"""Small, inspectable course calculations; standard library only, no GPU/model load.

These arithmetic examples explain mechanisms, not model quality or kernel
correctness. `records` reads existing formal logs. Nothing resumes training.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1'


def softmax(values):
    exps = [math.exp(v - max(values)) for v in values]
    return [v / sum(exps) for v in exps]


def attention():
    q = [1.0, 0.0]
    keys = [[1.0, 0.0], [0.0, 1.0], [100.0, 0.0]]
    values = [[2.0, 0.0], [0.0, 4.0], [999.0, 999.0]]

    def attend(ks, vs, visible):
        scores = [sum(a*b for a, b in zip(q, k))/math.sqrt(2) for k in ks[:visible]]
        weights = softmax(scores)
        output = [sum(p*v[d] for p, v in zip(weights, vs[:visible])) for d in range(2)]
        return scores, weights, output

    scores, weights, output = attend(keys, values, 2)
    mutated = attend(keys[:2]+[[-999.0, 999.0]], values[:2]+[[-777.0, -777.0]], 2)[2]
    assert output == mutated
    return {'scope':'二维教学例，未运行正式 Attention 类', 'query':q,
            'keys':keys,'values':values,'query_position':1,'visible_scores':scores,
            'causal_weights':weights+[0.0],'output':output,
            'future_changed_output':mutated,'without_causal_mask':attend(keys,values,3)[2]}


def delta():
    # A single scalar key/value component isolates decay and error correction.
    old, alpha, beta, key, value = 2.0, 0.5, 0.5, 1.0, 6.0
    decayed = alpha*old
    prediction = decayed*key
    innovation = beta*(value-prediction)
    state = decayed + key*innovation
    repeated = state + (value-state)  # alpha=beta=key=1 on next observation
    assert state == 3.5 and repeated == 6.0
    return {'scope':'一维教学特例；正式每头状态为96×96，另有投影/卷积/归一化/输出门',
            'old_state':old,'retention_alpha':alpha,'beta':beta,'key':key,'value':value,
            'decayed_state':decayed,'old_prediction':prediction,'correction':innovation,
            'new_state':state,'next_same_value_with_alpha_beta_one':repeated,
            'naive_addition_on_next_observation':state+value}


def moe():
    p = [0.1,0.6,0.2,0.1]
    chosen = p.index(max(p))
    balanced = 5e-4*4*sum(0.25*0.25 for _ in range(4))
    collapsed = 5e-4*4*0.97
    return {'scope':'路由与梯度的算术演示，不是正式模型的路由测量',
            'probabilities':p,'selected_expert_zero_based':chosen,
            'top1_normalized_forward_weight':1.0,'p_divided_by_itself_derivative':0.0,
            'straight_through_weight':'p - stop_gradient(p) + 1',
            'straight_through_derivative_wrt_selected_score':p[chosen]*(1-p[chosen]),
            'balanced_single_layer_aux':balanced,'collapsed_example_single_layer_aux':collapsed,
            'eight_balanced_layers_aux':8*balanced}


def ce_and_gradient(w,x=2.0):
    # Two candidate logits: [0, w*x], target is candidate 1.
    p = 1/(1+math.exp(-w*x))
    return math.log1p(math.exp(-w*x)), (p-1)*x, p


def adam_update(state, gradient, lr, decay=0.01):
    state['adam_step'] += 1
    t=state['adam_step']
    state['m']=0.9*state['m']+0.1*gradient
    state['v']=0.999*state['v']+0.001*gradient**2
    mh=state['m']/(1-0.9**t)
    vh=state['v']/(1-0.999**t)
    before=state['w']
    state['w']=before*(1-lr*decay)-lr*mh/(math.sqrt(vh)+1e-8)
    return {'weight_before':before,'gradient':gradient,'m':state['m'],'v':state['v'],
            'm_hat':mh,'v_hat':vh,'weight_after':state['w']}


def optimizer():
    state={'w':0.0,'m':0.0,'v':0.0,'adam_step':0}
    loss, grad, prob=ce_and_gradient(state['w'])
    eps=1e-6
    numerical=(ce_and_gradient(eps)[0]-ce_and_gradient(-eps)[0])/(2*eps)
    assert abs(grad-numerical)<1e-8
    before=state['w']
    updated=adam_update(state,grad,0.1)
    norm=5.0
    clipped=[v/norm for v in [3.0,4.0]]
    weighted=(100*2.0+900*1.0)/1000
    assert ce_and_gradient(state['w'])[0]<loss
    return {'scope':'仅更新临时的一个标量；不是MiniMind训练，学习率0.1也不是正式recipe',
            'logits_before':[0.0,0.0],'target':1,'correct_probability':prob,
            'ce_before':loss,'analytic_gradient':grad,'finite_difference_gradient':numerical,
            'weight_after_gradient_calculation_before_update':before,
            'adamw_update':updated,'ce_after':ce_and_gradient(state['w'])[0],
            'clip_example':{'before':[3.0,4.0],'norm':norm,'threshold':1.0,'after':clipped},
            'accumulation':{'target_counts':[100,900],'batch_mean_ce':[2.0,1.0],
                            'correct_token_mean':weighted,'wrong_equal_batch_mean':1.5}}


def resume():
    data=[1.0,2.0,0.5,1.5,3.0,0.8]
    def initial():
        return {'w':0.0,'m':0.0,'v':0.0,'adam_step':0,'step':0,'cursor':0},random.Random(19)
    def advance(state,rng,n):
        consumed=[]
        for _ in range(n):
            index=state['cursor']
            x=data[index]*(0.8+0.4*rng.random())
            lr=0.1*(0.1+0.45*(1+math.cos(math.pi*state['step']/4)))
            _,g,_=ce_and_gradient(state['w'],x)
            adam_update(state,g,lr)
            state['step']+=1; state['cursor']+=1
            consumed.append(index)
        return consumed
    baseline,rg=initial(); advance(baseline,rg,4)
    mid,rg=initial(); advance(mid,rg,2)
    saved=json.loads(json.dumps({'state':mid,'rng':rg.getstate()}))
    def as_tuple(obj):
        return tuple(as_tuple(v) for v in obj) if isinstance(obj,list) else obj
    outcomes={}
    for omission in ['none','optimizer','cursor','rng']:
        state=copy.deepcopy(saved['state']); generator=random.Random(19)
        generator.setstate(as_tuple(saved['rng']))
        if omission=='optimizer': state.update(m=0.0,v=0.0,adam_step=0)
        if omission=='cursor': state['cursor']=0
        if omission=='rng': generator=random.Random(999)
        indices=advance(state,generator,2)
        outcomes[omission]={'final_weight':state['w'],'next_indices':indices,
                            'same_full_state_as_uninterrupted':state==baseline}
    assert outcomes['none']['same_full_state_as_uninterrupted']
    assert all(not outcomes[k]['same_full_state_as_uninterrupted'] for k in ['optimizer','cursor','rng'])
    return {'scope':'单标量AdamW、带随机扰动的合成输入；JSON序列化后恢复，不加载正式checkpoint',
            'uninterrupted_final_state':baseline,'saved_step':2,'saved_payload':saved,
            'continuation_cases':outcomes,
            'limitation':'此演示解释遗漏状态的后果，不能替代正式GPU恢复一致性验证。'}


def records():
    selected={1,12000,56541,56542,113082}
    rows=[]; total=0; seconds=0.0; count=0; first=None; last=None; nonfinite=0
    with (RUN/'metrics.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('event')!='train': continue
            count+=1
            assert r['step']==count
            total+=r['valid_tokens']; seconds+=r['step_seconds']
            first=r['time'] if first is None else first; last=r['time']
            nonfinite+=int(not all(math.isfinite(r[k]) for k in ['ce_loss','aux_loss','grad_norm']))
            if r['step'] in selected:
                row={k:r[k] for k in ['step','epoch','cursor','ce_loss','aux_loss','loss','grad_norm',
                     'learning_rate','valid_tokens','tokens_trained','nominal_tokens',
                     'nonpad_input_tokens','step_seconds','cuda_peak_allocated']}
                row['valid_tokens_per_second_recalculated']=r['valid_tokens']/r['step_seconds']
                rows.append(row)
    assert count==113082 and total==735833934
    return {'scope':'真实正式训练日志，只读重算；不是本次新训练',
            'path':str((RUN/'metrics.jsonl').relative_to(ROOT)),
            'updates':count,'supervised_tokens':total,'update_hours':seconds/3600,
            'first_last_wall_hours':(last-first)/3600,'nonfinite_rows':nonfinite,
            'selected_updates':rows,'parameter_state_lower_bound_gib':205623072*16/1024**3}


LESSONS={'attention':attention,'delta':delta,'moe':moe,'optimizer':optimizer,'resume':resume,'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('lesson',choices=[*LESSONS,'all'])
    p.add_argument('--out',help='Optional NEW directory under 学习手册/实操输出; refuses overwrite')
    args=p.parse_args()
    out=None
    if args.out:
        out=Path(args.out).resolve()
        allowed=(ROOT/'学习手册/实操输出').resolve()
        if out==allowed or allowed not in out.parents:
            p.error('--out must be a new child of 学习手册/实操输出')
        if out.exists(): p.error('--out already exists; choose a new directory')
    chosen=LESSONS if args.lesson=='all' else {args.lesson:LESSONS[args.lesson]}
    report={name:fn() for name,fn in chosen.items()}
    if out:
        out.mkdir(parents=True,exist_ok=False)
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'output':str((out/'results.json').relative_to(ROOT)),
                          'lessons':list(report),'formal_training_started':False},ensure_ascii=False))
    else:
        display=copy.deepcopy(report)
        if 'resume' in display: display['resume'].pop('saved_payload')
        print(json.dumps(display,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
