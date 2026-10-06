#!/usr/bin/env python3
"""Capture an immutable training checkpoint at the requested review deadline.

This is a CPU-only evidence capture, not a training stop or a quality approval.
The trainer writes latest_resume.pt through atomic replacement. Hard-linking its
current inode before reading keeps model/optimizer/cursor internally consistent.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.pipeline import sha256


def capture(plan, out):
    import torch
    torch.set_num_threads(2)
    stages = [s for s in plan['stages'] if s['kind'] == 'train']
    candidates = [s for s in stages if
                  (ROOT / s['completion']['run_dir'] / 'checkpoints/latest_resume.pt').is_file()]
    if not candidates:
        raise RuntimeError('No actual trained recovery checkpoint available; do not fabricate delivery')
    selected = candidates[-1]
    run = ROOT / selected['completion']['run_dir']
    out.mkdir(parents=True, exist_ok=False)
    recovery = out / 'training_resume.pt'
    os.link(run / 'checkpoints/latest_resume.pt', recovery)
    saved = torch.load(recovery, map_location='cpu', mmap=True, weights_only=False)
    provenance = json.loads((run / 'provenance.json').read_text())
    if (saved['step'] <= 0 or saved['trained_tokens'] <= 0
            or saved['config'] != selected['completion']['config_contract']
            or provenance['parameter_count'] != 205623072
            or saved['source_sha256'] != provenance['source_sha256']):
        raise RuntimeError('Recovery is not a trained Hybrid checkpoint')
    weights = out / 'hybrid_205m_fp16.pth'
    torch.save({k: v.half() for k, v in saved['model'].items()}, weights)
    checkpoint_eval = run / 'evaluation' / f'step_{saved["step"]:07d}.json'
    if checkpoint_eval.is_file():
        shutil.copy2(checkpoint_eval, out / 'checkpoint_evaluation.json')
    (out / 'config.json').write_text(json.dumps(saved['config'], ensure_ascii=False, indent=2) + '\n')
    shutil.copytree(run / 'source', out / 'source')
    shutil.copy2(run / 'provenance.json', out / 'provenance.json')
    transfer = ROOT / plan['stages'][0]['out'] / 'transfer.json'
    shutil.copy2(transfer, out / 'transfer.json')
    stage_states = {}
    for stage in stages:
        p = ROOT / stage['completion']['run_dir'] / 'status.json'
        if p.exists():
            from lab.live_status import read_live_json
            stage_states[stage['id']] = read_live_json(p)
    receipt = dict(captured_local=datetime.now().astimezone().isoformat(),
                   purpose='Deadline review snapshot; training continues. Not a completed-model claim.',
                   run_dir=str(run.relative_to(ROOT)), step=saved['step'],
                   epoch=saved['epoch'], cursor=saved['cursor'],
                   stage_trained_tokens=saved['trained_tokens'],
                   expected_stage_steps=selected['completion']['expected_steps'],
                   model_source_sha256=saved['source_sha256'], stage_states=stage_states,
                   inference_precision='FP16 export of FP32 training parameters; export not independently re-evaluated',
                   checkpoint_evaluation_available=checkpoint_eval.is_file(),
                   evaluation_scope='If present, evaluation is for the same-step FP32 training model with BF16 autocast, not the rounded inference export.',
                   files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in [recovery, weights]})
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    (out / 'README.md').write_text(
        '# 指定时间的 Hybrid 训练快照\n\n'
        '**这是中间训练结果，不是完整模型训练完成证明。后台正式训练继续。**\n\n'
        f'捕获时间：{receipt["captured_local"]}。阶段：`{selected["id"]}`，'
        f'该阶段第 {saved["step"]:,} / {receipt["expected_stage_steps"]:,} 步，'
        f'{saved["trained_tokens"]:,} 个有效训练 token（仅此阶段，不包含基座历史）。\n\n'
        '`hybrid_205m_fp16.pth` 是完整模型参数导出；`training_resume.pt` 保留FP32模型、AdamW、RNG和cursor。'
        '导出保留完整205M结构；它不是官方权重改名。官方预训练初始化来源和新增模块迁移见 `transfer.json`。\n\n'
        '如果存在 `checkpoint_evaluation.json`，其中验证/生成对应相同步数的训练模型；FP16导出未单独重评。'
        '不能把其他步数的最好分数归给这一份快照。`receipt.json` 保存权重SHA与实际阶段状态。\n\n'
        '完整配置和源码快照在本目录；持续更新的日志、曲线和后续权重见模型的 `runs/` 及项目根README。\n')
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:
        raise ValueError('Deadline must include an explicit timezone')
    print(json.dumps(dict(event='waiting', deadline=a.deadline, out=a.out, pid=os.getpid())), flush=True)
    while True:
        remaining = deadline.timestamp() - time.time()
        if remaining <= 0:
            break
        time.sleep(min(30, remaining))
    plan = json.loads((ROOT / a.plan).read_text())
    receipt = capture(plan, ROOT / a.out)
    print(json.dumps(dict(event='captured', step=receipt['step'], out=a.out), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
