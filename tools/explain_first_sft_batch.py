#!/usr/bin/env python3
"""Reconstruct the real first SFT batch on CPU, without a model or updates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.data import EpochBatchSampler, TokenCorpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    assert cfg['stage'] == 'sft' and cfg['accumulation_steps'] == 1
    corpus = TokenCorpus(ROOT / cfg['corpus'], 'train', cfg['seq_len'], 'sft')
    order = next(iter(EpochBatchSampler(len(corpus), cfg['batch_size'], cfg['seed'])))
    pairs = [corpus[i] for i in order]
    ids = torch.stack([p[0] for p in pairs])
    labels = torch.stack([p[1] for p in pairs])
    first_metric = next(json.loads(line) for line in (ROOT / cfg['run_dir'] / 'metrics.jsonl').open()
                        if json.loads(line).get('event') == 'train')
    count = int((labels[:, 1:] != -100).sum())
    assert first_metric['step'] == 1 and first_metric['valid_tokens'] == count
    assert first_metric['nominal_tokens'] == ids.numel()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'upstream/model')
    examples = []
    for i in range(2):
        row = int(corpus.rows[order[i]])
        starts, ends = int(corpus.offsets[row]), int(corpus.offsets[row + 1])
        examples.append(dict(batch_position=i, dataset_index=order[i], cache_row=row,
                             raw_line=row + 1, untruncated_tokens=ends - starts,
                             input_ids=ids[i].tolist(), labels=labels[i].tolist(),
                             rendered_prefix=tokenizer.decode(ids[i][ids[i] != 0], skip_special_tokens=False),
                             first_64_positions=[dict(position=j, token_id=int(ids[i, j]),
                                                      piece=tokenizer.decode([int(ids[i, j])]),
                                                      label=int(labels[i, j]),
                                                      is_next_token_target=j > 0 and int(labels[i, j]) != -100)
                                                 for j in range(64)]))
    report = dict(config=args.config, config_sha256=hashlib.sha256((ROOT / args.config).read_bytes()).hexdigest(),
                  input_shape=list(ids.shape), labels_shape=list(labels.shape),
                  logits_shape=[cfg['batch_size'], cfg['seq_len'], 6400],
                  supervised_tokens=count, nominal_positions=ids.numel(),
                  first_real_update=first_metric, shuffled_dataset_indices=order,
                  raw_line_note='Valid because this official file had zero rejected or empty records.',
                  examples=examples, scope='CPU reconstruction and matching real first-step counts; no GPU update or fabricated logits')
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'source_explanation.py')
    (out / 'batch.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 真实Hybrid第一个batch：从文本到目标标签', '',
             f'对应正式配置 `{args.config}`。仅CPU重建，未新增训练；顺序来自真实seed和sampler。', '',
             f'输入和labels均为 `{list(ids.shape)}`；logits按模型定义为 `{report["logits_shape"]}`。',
             f'本批{ids.numel():,}个输入位置，实际有{count:,}个next-token监督标签，与正式第1步metrics一致。', '',
             '`input_ids`包含用户、助手、特殊token与padding；`labels=-100`的位置不进入CE。',
             '`logits[:, :-1]`预测`labels[:, 1:]`，因此位置j的非忽略label由位置j-1的logits预测；没有把输入原位置直接当成答案。',
             '先对有效assistant目标求CE均值，再加MoE辅助损失；此配置累积1，一批就是一次optimizer更新。', '',
             '下面列出第一条记录的前64个位置；完整两条的ID、labels和可读模板在`batch.json`。', '',
             '| 位置j | token ID | 解码片段（JSON转义） | label | 参与next-token目标 |',
             '|---:|---:|---|---:|---|']
    for item in examples[0]['first_64_positions']:
        piece = json.dumps(item['piece'], ensure_ascii=False).replace('|', '&#124;').replace('`', '&#96;')
        lines.append(f'| {item["position"]} | {item["token_id"]} | {piece} | {item["label"]} | {item["is_next_token_target"]} |')
    lines += ['', '单token的解码可能显示替换符，因为一个Unicode字符可能跨token；这不自动证明原始文本损坏。应比较完整序列解码与原始模板。', '',
              '调试时先检查有效标签数、user/assistant跨度、右移位置、pad ID和上下文截断；如果所有labels为-100，均值CE可能没有定义，不能仅靠降低学习率处理。', '',
              '第一步loss/aux/grad_norm与资源数值完整保存在JSON；不同batch的loss不能代替同一验证集的质量比较。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, first_step=1, supervised_tokens=count, shape=list(ids.shape), out=str(out))))


if __name__ == '__main__':
    main()
