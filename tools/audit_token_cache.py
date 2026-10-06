#!/usr/bin/env python3
"""Exact supervision budget after prefix cropping, plus duplicate/length audit."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
SCRIPT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True); p.add_argument('--sequence',type=int,required=True)
    p.add_argument('--batch',type=int,required=True); p.add_argument('--accumulation',type=int,required=True)
    p.add_argument('--out',required=True); p.add_argument('--wait',action='store_true'); a=p.parse_args()
    root=Path(a.corpus); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    while a.wait and not (root/'metadata.json').exists(): time.sleep(10)
    if out.with_suffix('.json').exists(): raise RuntimeError('Preserve the existing audit; choose a new output name')
    meta=json.loads((root/'metadata.json').read_text()); lengths=np.load(root/'lengths.npy',mmap_mode='r')
    offsets=np.load(root/'offsets.npy',mmap_mode='r'); first=np.load(root/'first_targets.npy',mmap_mode='r')
    hashes=np.load(root/'content_hash64.npy',mmap_mode='r'); unique,multiplicity=np.unique(hashes,return_counts=True)
    tokens=np.memmap(root/('labels.bin' if meta['stage']=='sft' else 'tokens.bin'),mode='r',dtype='<i2' if meta['stage']=='sft' else '<u2')
    report={'corpus':str(root),'stage':meta['stage'],'sequence':a.sequence,'batch_size':a.batch,'accumulation_steps':a.accumulation,
            'cache_metadata':meta,'unique_hashes':len(unique),'extra_exact_duplicates':len(hashes)-len(unique),
            'max_multiplicity':int(multiplicity.max()),'splits':{}}
    for split in ['train','val']:
        rows=np.load(root/(split+'.npy'),mmap_mode='r'); original=len(rows)
        expected_val=hashes[rows]%1000<1
        if (split=='val' and not expected_val.all()) or (split=='train' and expected_val.any()):
            raise ValueError(f'Unexpected content-hash assignment in {split}')
        del expected_val
        if meta['stage']=='sft': rows=rows[first[rows]<a.sequence]
        supervised=nonpad=0
        for i,row in enumerate(rows):
            start,end=map(int,offsets[row:row+2])
            if meta['stage']=='sft':
                end=min(end,start+a.sequence)
                supervised+=int(np.count_nonzero(tokens[start+1:end]!=-100)); nonpad+=end-start
            else:
                end=min(end,start+a.sequence-2)
                supervised+=int(np.count_nonzero(tokens[start:end]!=0))+1; nonpad+=end-start+2
            if (i+1)%500000==0: print(json.dumps({'split':split,'rows_audited':i+1,'supervised_tokens':supervised}),flush=True)
        report['splits'][split]=dict(original_rows=original,rows=len(rows),removed_zero_target_rows=original-len(rows),
                                     supervised_tokens=supervised,nonpad_positions=nonpad,
                                     positional_padding_fraction=1-nonpad/(len(rows)*a.sequence),
                                     optimizer_steps=math.ceil(len(rows)/(a.batch*a.accumulation)))
    report['exact_duplicate_cross_split']=False
    report['split_assignment_verified']=True
    report['limitation']='Content-hash split prevents identical records crossing train/val; no semantic decontamination claim. Padding estimate counts positions before literal PAD token handling.'
    report['script_sha256']=SCRIPT_SHA256
    report['time']=time.time(); out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    train=report['splits']['train']; val=report['splits']['val']
    lines=[f'# {meta["stage"].upper()} 全量token cache审计','',f'来源：`{root}`；完整原始数据校验与分词信息包含在同名JSON中。','',
           f'序列长度{a.sequence}，microbatch {a.batch}，累积{a.accumulation}；一epoch {train["optimizer_steps"]:,}次optimizer更新。','',
           f'训练{train["rows"]:,}条，验证{val["rows"]:,}条；该长度下训练有效监督token **{train["supervised_tokens"]:,}**。',
           f'截断后无监督标签的训练行{train["removed_zero_target_rows"]:,}；训练位置padding估计{train["positional_padding_fraction"]:.2%}。','',
           f'完整cache额外完全重复记录{report["extra_exact_duplicates"]:,}，最大重复次数{report["max_multiplicity"]}。相同内容hash固定分在同侧；未做近重复语义去污。','',
           '| 截断长度 | 被截断记录比例 | 输入token保留比例 | 监督token保留比例 | 零监督行 |','|---|---:|---:|---:|---:|']
    for length,stats in meta['length_coverage'].items():
        lines.append(f'| {length} | {stats["cut_rows"]/meta["rows"]:.2%} | {stats["tokens_retained"]/meta["full_input_tokens"]:.2%} | {stats["supervised_retained"]/meta["full_supervised_tokens"]:.2%} | {stats["zero_supervision_rows"]:,} |')
    lines+=['','以上保留比例来自全部cache（含held-out）；精确训练token预算另外逐行检查labels。有效标签数、输入非PAD数和固定padding后的名义token数不能混为吞吐口径。','',
            'epoch只表示遍历次数，不自动代表收敛；后续仍需验证曲线、生成与数据错误分析。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cache_metadata'}),flush=True)

if __name__=='__main__': main()
