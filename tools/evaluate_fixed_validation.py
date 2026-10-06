#!/usr/bin/env python3
"""Score immutable exported weights on the training recipe's fixed held-out set.

This is evaluation only. Official reference weights may have seen these records;
the result is a transfer diagnostic, not an uncontaminated benchmark.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Model structure/source')
    parser.add_argument('--recipe', required=True, help='Corpus, length, fixed validation seed/count')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    recipe = json.loads((ROOT / args.recipe).read_text())
    assert cfg['stage'] == recipe['stage'] == 'sft'
    os.environ['TRITON_F32_DEFAULT'] = recipe.get('triton_f32_default', 'ieee')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from lab.data import TokenCorpus

    with (ROOT / 'runs/local_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT / args.out
        out.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(4)
        torch.manual_seed(20261008)
        torch.backends.cuda.matmul.allow_tf32 = False
        model_path = ROOT / cfg['model_file']
        spec = importlib.util.spec_from_file_location('validation_model', model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.MiniMindForCausalLM(module.MiniMindConfig(**cfg['model']))
        weight = ROOT / args.weights
        before_sha = sha(weight)
        state = torch.load(weight, map_location='cpu', mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        assert sha(weight) == before_sha, 'Weights changed during load'
        model = model.cuda().eval()
        val = TokenCorpus(ROOT / recipe['corpus'], 'val', recipe['seq_len'], 'sft')
        count = min(len(val), recipe.get('validation_samples', 512))
        seed = recipe.get('validation_seed', recipe['seed'] + 1000003)
        selection = torch.randperm(len(val), generator=torch.Generator().manual_seed(seed))[:count].tolist()
        total_loss, tokens = 0., 0
        started = time.perf_counter()
        with torch.inference_mode():
            for x, labels in DataLoader(Subset(val, selection), batch_size=recipe['batch_size'], num_workers=0):
                n = int((labels[:, 1:] != -100).sum())
                if not n:
                    continue
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(x.cuda(), labels=labels.cuda())
                value = result.loss.item()
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite validation loss')
                total_loss += value * n
                tokens += n
        assert tokens > 0
        report = dict(complete=True, validation_ce=total_loss / tokens, validation_rows=count,
                      validation_tokens=tokens, validation_selection_sha256=hashlib.sha256(
                          np.asarray(selection, dtype='<u8').tobytes()).hexdigest(),
                      config=args.config, recipe=args.recipe, weights=args.weights,
                      weights_sha256=before_sha, model_sha256=sha(model_path),
                      config_sha256=sha(ROOT / args.config), recipe_sha256=sha(ROOT / args.recipe),
                      corpus_metadata_sha256=sha(ROOT / recipe['corpus'] / 'metadata.json'),
                      script_sha256=sha(Path(__file__)), parameters=sum(p.numel() for p in model.parameters()),
                      parameter_dtype='float32', autocast='bfloat16', pytorch_tf32=False,
                      triton_f32_default=os.environ['TRITON_F32_DEFAULT'],
                      seconds=time.perf_counter() - started, time=time.time(),
                      scope='Same seeded subset and token-weighted assistant CE; no aux loss. '
                            'Exported FP16 weights loaded into FP32. Official base may have seen held-out records.')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
