#!/usr/bin/env python3
"""Check TF32 projection matmuls against an independent strict FP32 recurrence.

FLA remains TF32x3 in both kernel paths. No model, data or train budget is reduced.
This numerical pilot alone does not approve a formal training recipe.
"""
import copy
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['TRITON_F32_DEFAULT'] = 'tf32x3'
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_hybrid_precision_gpu import sequential, error


def main():
    out = ROOT / 'models/02_hybrid_moe/runs/torch_tf32_precision_v1'
    out.mkdir(parents=True, exist_ok=False)
    lock = (ROOT / 'runs/local_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(5.3 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    path = ROOT / 'models/02_hybrid_moe/src/model_hybrid.py'
    spec = importlib.util.spec_from_file_location('tf32_hybrid', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    chunk = m.chunk_gated_delta_rule
    records = []; comparisons = []
    try:
        for seed in [20261007, 20261008]:
            torch.manual_seed(seed)
            template = m.GatedDeltaNet(m.MiniMindConfig(hidden_size=768, require_fla=True), 0).cuda()
            x = torch.randn(1, 257, 768, device='cuda'); probe = torch.randn_like(x)
            results = {}
            for name, kernel, tf32 in [('independent_strict', sequential, False), ('fla_strict', chunk, False), ('fla_torch_tf32', chunk, True)]:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                m.chunk_gated_delta_rule = kernel
                layer = copy.deepcopy(template); inp = x.detach().clone().requires_grad_()
                started = time.time(); y, _ = layer(inp); (y * probe).mean().backward(); torch.cuda.synchronize()
                results[name] = dict(output=y.detach().cpu(), input=inp.grad.detach().cpu(), gradients={n: p.grad.detach().cpu() for n, p in layer.named_parameters()})
                row = dict(seed=seed, variant=name, seconds=time.time() - started)
                records.append(row); print(json.dumps(row), flush=True)
                del layer, inp, y; gc.collect(); torch.cuda.empty_cache()
            for name in ['fla_strict', 'fla_torch_tf32']:
                a, b = results['independent_strict'], results[name]
                row = dict(seed=seed, candidate=name, output=error(a['output'], b['output']), input_gradient=error(a['input'], b['input']), gradients={n: error(v, b['gradients'][n]) for n, v in a['gradients'].items()})
                row['passed'] = row['output']['finite'] and row['output']['relative_rms'] < .02 and row['input_gradient']['finite'] and row['input_gradient']['relative_rms'] < .03 and all(v['finite'] and v['relative_rms'] < .03 for v in row['gradients'].values())
                comparisons.append(row); print(json.dumps(row), flush=True)
            del template, x, probe, results; gc.collect(); torch.cuda.empty_cache()
        assert all(r['passed'] for r in comparisons), 'Preset output/gradient gates failed'
    finally:
        (out / 'report.json').write_text(json.dumps(dict(records=records, comparisons=comparisons, passed=len(comparisons) == 4 and all(r['passed'] for r in comparisons), scope='Two full-width initializations; output <2%, input and every parameter gradient <3%; not yet a full-model training/throughput/resume check.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
