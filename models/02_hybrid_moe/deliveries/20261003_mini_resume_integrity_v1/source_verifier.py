#!/usr/bin/env python3
"""CPU integrity audit of one immutable real Hybrid optimizer checkpoint.

This does not simulate resumption or prove numerical equivalence of a next step.
The hard link remains stable when the trainer atomically replaces latest_resume.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'source_verifier.py')
    checkpoint = out / 'training_resume.pt'
    os.link(run / 'checkpoints/latest_resume.pt', checkpoint)
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = json.loads((run / 'config.json').read_text())
    provenance = json.loads((run / 'provenance.json').read_text())
    assert saved['config'] == cfg
    assert saved['step'] > 0 and saved['trained_tokens'] > 0
    assert set(saved['rng']) == {'python', 'numpy', 'torch', 'cuda'}
    model_path = ROOT / cfg['model_file']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == saved['source_sha256'] == provenance['source_sha256']
    spec = importlib.util.spec_from_file_location('verified_hybrid', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.device('meta'):
        model = module.MiniMindForCausalLM(module.MiniMindConfig(**cfg['model']))
    expected = model.state_dict()
    assert expected.keys() == saved['model'].keys()
    count = sum(p.numel() for p in model.parameters())
    assert count == 205623072 == provenance['trainable_parameters']
    init = torch.load(ROOT / cfg['init_weight'], map_location='cpu', mmap=True, weights_only=True)
    changed = {'linear': [], 'inherited': []}
    for name, tensor in saved['model'].items():
        assert tensor.shape == expected[name].shape, name
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all(), name
        if not torch.equal(tensor, init[name]):
            changed['linear' if '.linear_attn.' in name else 'inherited'].append(name)
    assert changed['linear'] and changed['inherited']
    assert torch.equal(saved['model']['lm_head.weight'], saved['model']['model.embed_tokens.weight'])
    param_ids = [i for group in saved['optimizer']['param_groups'] for i in group['params']]
    state = saved['optimizer']['state']
    assert set(param_ids) == set(state)
    assert len(param_ids) == len(list(model.parameters()))
    moment_numel = 0
    for param, identifier in zip(model.parameters(), param_ids):
        values = state[identifier]
        assert int(values['step']) == saved['step']
        for key in ['exp_avg', 'exp_avg_sq']:
            tensor = values[key]
            assert tensor.shape == param.shape
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        moment_numel += values['exp_avg'].numel()
    assert moment_numel == count
    report = dict(passed=True, scope='CPU shape/dtype/finite/optimizer/RNG/lineage audit; not a next-update resume-equivalence test or quality approval',
                  run=args.run, time=time.time(), step=saved['step'], epoch=saved['epoch'], cursor=saved['cursor'],
                  trained_tokens=saved['trained_tokens'], parameters=count, model_keys=len(expected),
                  optimizer_parameter_states=len(state), moment_parameters=moment_numel,
                  changed_tensor_names=changed, bytes=checkpoint.stat().st_size,
                  source_sha256=saved['source_sha256'])
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'changed_tensor_names'}), flush=True)


if __name__ == '__main__':
    main()
