#!/usr/bin/env python3
"""Generate static learning plots and searchable tables from real, retained logs."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1'
BOOK=ROOT/'学习手册'
FIG=BOOK/'图表'
TAB=BOOK/'数据表'
FONT=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})


def table(name, fields, rows):
    with (TAB/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def save(fig,name):
    fig.savefig(FIG/name,dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in (RUN/'metrics.jsonl').read_text().splitlines() if s.strip()]
    train=[r for r in records if r.get('event')=='train']
    assert [r['step'] for r in train]==list(range(1,113083))
    assert sum(r['valid_tokens'] for r in train)==735833934
    # The final full-validation JSON replaced the same step's fixed-subset file;
    # the append-only event log retains both protocols and is authoritative here.
    evaluations=[r for r in records if r.get('event')=='evaluation']
    fixed=[r for r in records if r.get('event')=='evaluation' and r.get('validation_rows')==512 and not r.get('final')]
    fixed=list({r['step']:r for r in fixed}.values())
    fields=['step','epoch','cursor','ce_loss','aux_loss','loss','learning_rate','grad_norm','valid_tokens','nominal_tokens','nonpad_input_tokens','unsupervised_fraction',
            'tokens_trained','step_seconds','valid_tokens_per_second','nominal_tokens_per_second',
            'input_padding_fraction','cuda_allocated','cuda_reserved','cuda_peak_allocated','wall_seconds','time']
    table('全部训练更新.csv',fields,train)
    table('验证记录.csv',['step','epoch','trained_tokens','validation_ce','validation_rows','validation_tokens','final','validation_selection_sha256'],evaluations)
    ck=[json.loads(s) for s in (RUN/'checkpoint_progress.jsonl').read_text().splitlines()]
    table('Checkpoint记录.csv',['kind','step','epoch','cursor','tokens','bytes','seconds','time'],ck)
    mapping=[]
    for file in ['lab/train.py','lab/data.py','lab/checkpointing.py','lab/transfer.py',
                 'models/02_hybrid_moe/src/model_hybrid.py','tools/prepare_tokens.py','tools/evaluate_text_probes.py',
                 'tools/study_minimind.py','tools/build_learning_delivery.py']:
        p=ROOT/file
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                mapping.append({'path':file,'symbol':node.name,'line':node.lineno,'kind':type(node).__name__})
    table('源码地图.csv',['path','symbol','line','kind'],mapping)
    # A single figure describes the real forward graph and update order.
    fig,ax=plt.subplots(figsize=(13,9));ax.set(xlim=(0,13),ylim=(0,9));ax.axis('off')
    def box(x,y,w,h,label,color='#e8f0f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc=color,ec='#50677a',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    def arrow(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#405565','lw':1.5})
    ax.text(.2,8.55,'这次205M Hybrid：数据、结构与一次更新',fontsize=21,weight='bold')
    box(.3,6.7,3.1,1.1,'原始对话 → 模板 → tokenizer\ninput_ids / labels [16,768]')
    box(4.3,6.7,3.5,1.1,'Embedding [6400,768]\nhidden [16,768,768]')
    box(8.7,6.7,3.8,1.1,'8个block：L L L F L L L F\n每层都含4专家Top-1 FFN')
    arrow(3.55,7.25,4.1,7.25);arrow(7.95,7.25,8.5,7.25)
    box(.3,4.1,5.5,1.7,'L = Gated DeltaNet × 6\nq/k/v [B,T,8,96]；g/beta [B,T,8]\n每层递推状态 [B,8,96,96]\n压缩历史；遗忘＋delta纠错写入','#e6f3ee')
    box(6.7,4.1,5.8,1.7,'F = Full Attention × 2\nQ [B,T,8,96]；K/V [B,T,4,96]\nQ/K norm＋RoPE＋因果softmax\n直接检索历史；KV cache随长度增长','#fff1dc')
    box(.3,2.2,3.1,1.0,'每层：Norm → Attention＋残差\nNorm → MoE＋残差')
    box(4.3,2.2,3.5,1.0,'末端Norm → 共享词表投影\nlogits [16,768,6400]')
    box(8.7,2.2,3.8,1.0,'logits[:, :-1] 对 labels[:, 1:]\n有效assistant CE＋MoE aux')
    arrow(3.55,2.7,4.1,2.7);arrow(7.95,2.7,8.5,2.7)
    ax.text(.35,1.25,'LR → zero_grad → forward → backward → clip(norm≤1) → AdamW.step → 计数/日志 → 评估/保存',fontsize=12)
    ax.text(.35,.55,'训练来源：官方198M SFT → 六层随机注意力替换 → 两轮全参数SFT。没有Hybrid预训练。\n同shape只保证接口兼容，不保证保留旧能力；两轮预算完成，评估结果与生成样例已记录。',fontsize=11,color='#9a3d34')
    save(fig,'01_模型和训练数据流.png')
    x=np.array([r['step'] for r in train]);ce=np.array([r['ce_loss'] for r in train])
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    axs[0,0].plot(x[::25],ce[::25],alpha=.17,color='#3577a0',label='训练CE：每25步展示一个原始点')
    smooth=np.convolve(ce,np.ones(500)/500,mode='valid')
    axs[0,0].plot(x[499:],smooth,lw=1,label='训练CE：过去500步均值')
    axs[0,0].plot([r['step'] for r in fixed],[r['validation_ce'] for r in fixed],color='#d0613c',label='固定512条验证CE')
    axs[0,0].set(title='预测损失下降 ≠ 自由生成正确',ylabel='CE',yscale='log');axs[0,0].legend(fontsize=8)
    axs[0,1].plot(x,[r['learning_rate'] for r in train],color='#586fa5');axs[0,1].set(title='学习率：无warmup，cosine降至约1e-6',ylabel='LR')
    axs[1,0].plot(x[::20],[r['grad_norm'] for r in train[::20]],lw=.7);axs[1,0].axhline(1,color='#c65d49',ls='--',label='裁剪阈值1')
    axs[1,0].set(title='记录的是裁剪前梯度范数',ylabel='global grad norm',yscale='log');axs[1,0].legend()
    rates=np.array([r['valid_tokens_per_second'] for r in train]);axs[1,1].plot(x[::25],rates[::25],alpha=.25,lw=.6)
    axs[1,1].plot(x[499:],np.convolve(rates,np.ones(500)/500,'valid'),color='#227860',lw=1)
    axs[1,1].set(title='有效监督token吞吐：受样本长度与padding影响',ylabel='有效labels / 秒')
    for a in axs.flat:a.set_xlabel('optimizer step');a.grid(alpha=.2)
    fig.suptitle('完整113,082步真实记录；逐步原值全部保存在CSV/JSONL',fontsize=16)
    save(fig,'02_完整训练曲线.png')
    gpu=[]
    with (RUN/'gpu.csv').open() as f:
        for row in csv.reader(f):
            try:gpu.append([float(row[i].strip().split()[0]) for i in (2,3,4,5)])
            except (ValueError,IndexError):continue
    g=np.array(gpu);fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for i,(title,unit) in enumerate([('整张GPU显存（含桌面/其他进程）','MiB'),('GPU利用率','%'),('GPU温度','℃'),('GPU功耗','W')]):
        a=axs.flat[i];a.plot(np.arange(len(g))[::20],g[::20,i],lw=.7);a.set(title=title,ylabel=unit,xlabel='采样序号（图中每20条显示一点）');a.grid(alpha=.2)
    fig.suptitle('原始nvidia-smi约每2秒采样；存在休眠/时钟异常，横轴不当作连续墙钟时间',fontsize=14)
    save(fig,'03_GPU资源曲线.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    values=[1.3074592289684233,1.6645607152115836]
    bars=axs[0].bar(['官方198M SFT','本次205M Hybrid'],values,color=['#39788e','#c06c53'])
    axs[0].bar_label(bars,fmt='%.4f');axs[0].set(ylabel='同512条验证CE，越低越好',ylim=(0,2),title='同协议、同FP16导出后加载')
    axs[0].text(.02,.02,'官方基座可能见过留出数据\n此图不是干净泛化benchmark或预算匹配消融',transform=axs[0].transAxes,fontsize=9)
    memory=np.array([4,4,8])*205623072/1024**3
    bars=axs[1].bar(['FP32参数','FP32梯度','AdamW m+v'],memory,color=['#386e98','#4d987b','#bc8c3d']);axs[1].bar_label(bars,fmt='%.3f')
    axs[1].set(ylabel='GiB（理论账本）',ylim=(0,2),title=f'训练状态约{memory.sum():.3f} GiB，不含激活/临时量/context')
    save(fig,'04_验证对照与显存账本.png')
    manifest={'training_rows':len(train),'tokens':sum(r['valid_tokens'] for r in train),
              'fixed_validation_points':len(fixed),'evaluation_records':len(evaluations),
              'saved_evaluation_files':len(list((RUN/'evaluation').glob('step_*.json'))),
              'gpu_samples':len(gpu),'checkpoint_records':len(ck),
              'metrics_sha256':hashlib.sha256((RUN/'metrics.jsonl').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note':'Plots are explanatory exports; original logs and all numerical rows retained.'}
    (ROOT/'reports/learning_delivery_v1/generated.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
