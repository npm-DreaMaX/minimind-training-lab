#!/usr/bin/env python3
"""Check one immutable trained Hybrid checkpoint's CPU cache semantics.

Uses FP32, the existing PyTorch CPU path and no GPU. This is not a FLA/GPU
precision validation and not an inference-quality benchmark.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, candidate):
    difference = reference.float() - candidate.float()
    return dict(relative_rms=float(difference.square().mean().sqrt() /
                                  reference.float().square().mean().sqrt().clamp_min(1e-12)),
                max_absolute=float(difference.abs().max()),
                argmax_mismatches=int((reference.argmax(-1) != candidate.argmax(-1)).sum()),
                compared_positions=reference.shape[-2],
                finite=bool(torch.isfinite(candidate).all()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tokens', type=int, default=32)
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'source_check.py')
    os.link(run / 'checkpoints/latest_resume.pt', out / 'training_resume.pt')
    torch.set_num_threads(2)
    saved = torch.load(out / 'training_resume.pt', map_location='cpu', mmap=True, weights_only=False)
    cfg = saved['config']
    source = ROOT / cfg['model_file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == saved['source_sha256']
    spec = importlib.util.spec_from_file_location('trained_cache_cpu', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MiniMindForCausalLM(module.MiniMindConfig(**cfg['model'])).eval()
    model.load_state_dict(saved['model'], strict=True)
    step = saved['step']
    del saved
    gc.collect()
    assert sum(p.numel() for p in model.parameters()) == 205623072
    assert all(p.device.type == 'cpu' for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'upstream/model')
    prompts = ['你好，请介绍一下自己。', '为什么天空是蓝色的？', '计算 17 加 25。']
    records = []
    started = time.time()
    with torch.inference_mode():
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                                                      tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).input_ids
            prefix_length = ids.shape[1]
            cached = model(ids, use_cache=True)
            cached_logits = [cached.logits]
            past = cached.past_key_values
            generated = []
            for _ in range(args.tokens):
                token = cached.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(token[0, 0]))
                ids = torch.cat([ids, token], dim=1)
                cached = model(token, past_key_values=past, use_cache=True)
                cached_logits.append(cached.logits)
                past = cached.past_key_values
                if generated[-1] == tokenizer.eos_token_id:
                    break
            # Exactly the same input sequence in one causal forward; avoids
            # confusing floating-point differences with diverging input tokens.
            full = model(ids, use_cache=False).logits
            metrics = compare(full, torch.cat(cached_logits, dim=1))
            record = dict(prompt=prompt, generated_ids=generated,
                          cpu_greedy_text=tokenizer.decode(generated, skip_special_tokens=False),
                          prefix_tokens=prefix_length, sequence_tokens=ids.shape[1], **metrics)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del full, cached, cached_logits, past, ids
        # Explicitly straddle the default128-token chunk boundary using a
        # deterministic real-language input, with126 tokens already cached.
        text = '模型通过比较预测和答案来更新参数，并保存训练状态。' * 16
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[:, :137]
        assert ids.shape[1] == 137
        prefill = model(ids[:, :126], use_cache=True)
        pieces = [prefill.logits]
        past = prefill.past_key_values
        for i in range(126, ids.shape[1]):
            part = model(ids[:, i:i + 1], past_key_values=past, use_cache=True)
            pieces.append(part.logits)
            past = part.past_key_values
        boundary = compare(model(ids, use_cache=False).logits, torch.cat(pieces, dim=1))
    # Predefined diagnostic tolerances; report even if they fail.
    passed = all(v['finite'] and v['relative_rms'] < 1e-3 and v['max_absolute'] < .02
                 for v in records + [boundary])
    report = dict(step=step, parameters=205623072, device='cpu', precision='FP32, no autocast',
                  scope='Trained-weight CPU cached recurrence vs same-sequence full forward; does not validate GPU FLA or BF16 kernels',
                  tolerance=dict(relative_rms=1e-3, max_absolute=.02), passed=passed,
                  cases=records, chunk_boundary=boundary, elapsed_seconds=time.time() - started,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, step=step, boundary=boundary, seconds=report['elapsed_seconds'])), flush=True)
    if not passed:
        raise RuntimeError('CPU trained-cache diagnostic exceeded the predefined tolerance; evidence retained')


if __name__ == '__main__':
    main()
