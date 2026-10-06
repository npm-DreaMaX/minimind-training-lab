#!/usr/bin/env python3
"""Bounded, real-data full-Omni GPU checks in the authorized remote workspace.

No learned weights are used. A passed test proves only this T2A/resume path,
not A2A/I2T readiness, long-run stability, learned quality, or reserved capacity.
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    import torch
    import numpy as np
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not str(ROOT).startswith('/new_data/REMOTE_USER/minimind/'):
        raise RuntimeError('This tool is reserved for the authorized remote readiness workspace')
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'source.py')
    query = ['nvidia-smi', f'--id={a.gpu}', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']
    device = subprocess.check_output(query, text=True).strip()
    fields = [v.strip() for v in device.split(',')]
    if int(fields[3]) > 512 or int(fields[4]) < 15000 or int(fields[5]) != 0:
        raise RuntimeError(f'Selected GPU is not idle with sufficient memory: {device}')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpu), 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4'}
    base = dict(purpose='preflight', device='cuda', model=dict(hidden_size=768, num_hidden_layers=8, use_moe=True, dropout=.1),
                modality='t2a', mode='all', corpus='artifacts/omni_validation/t2a_shards_v1',
                train_indices_override=[0, 7, 8, 9, 10, 12, 32, 44], max_length=1536,
                batch_size=1, accumulation_steps=2, epochs=2, max_steps=5, learning_rate=1e-4,
                warmup_steps=1, seed=20261006, cpu_threads=4, num_workers=2, deterministic=True,
                checkpointing=True, eval_interval=2, save_interval=2, validation_samples=3,
                log_interval=1, allocator_gib=13.5, inspect_first_batches=2, optimizer_state_offload=False)
    configs = {}
    for name in ['continuous', 'resumed', 'resource_B4_recompute']:
        config = {**base, 'run_dir': str((out / name).relative_to(ROOT))}
        if name.startswith('resource'):
            config.pop('train_indices_override')
            config.update(model=dict(hidden_size=768, num_hidden_layers=8, use_moe=True),
                          batch_size=4, accumulation_steps=4, max_steps=12, epochs=1,
                          learning_rate=1e-5, warmup_steps=1, eval_interval=50, save_interval=50)
        path = out / f'{name}.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        configs[name] = path
    record = dict(scope='Full 314887938 parameter random-initialized Omni; real small T2A fixture; no frozen input encoders',
                  gpu_snapshot=device, started=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  commands=[], status='running')
    try:
        for i, (name, flags) in enumerate([('continuous', []), ('resumed', ['--stop-after-step', '1']),
                                         ('resumed', ['--resume']), ('resource_B4_recompute', [])]):
            command = [sys.executable, '-u', '-B', '-m', 'lab.omni_train', '--config', str(configs[name]), *flags]
            print(json.dumps(dict(event='command_start', command=command, time=time.time())), flush=True)
            started = time.monotonic()
            with (out / f'command_{i}.log').open('x') as log:
                result = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    result.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(result.pid, signal.SIGTERM)
                    try:
                        result.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(result.pid, signal.SIGKILL)
                        result.wait()
                    raise
            record['commands'].append(dict(command=command, returncode=result.returncode, seconds=time.monotonic()-started))
            if result.returncode:
                raise RuntimeError(f'{name} failed; retained command_{i}.log')
        import torch
        states = [torch.load(out / n / 'checkpoints/latest_resume.pt', map_location='cpu', mmap=True, weights_only=False)
                  for n in ['continuous', 'resumed']]
        keys = ['model', 'optimizer', 'epoch', 'cursor', 'step', 'text_tokens', 'audio_tokens', 'best', 'rng']
        record['resume_checks'] = {k: equal(states[0][k], states[1][k]) for k in keys}
        for name in configs:
            provenance = json.loads((out / name / 'provenance.json').read_text())
            status = json.loads((out / name / 'status.json').read_text())
            record.setdefault('run_checks', {})[name] = dict(full_parameter_count=provenance['parameters'] == 314887938,
                                                            status=status['status'], step=status['step'])
        assert all(record['resume_checks'].values()), record['resume_checks']
        assert all(v['full_parameter_count'] and v['status'] == 'complete' for v in record['run_checks'].values())
        assert states[0]['step'] == 5 and states[0]['epoch'] >= 1
        record['status'] = 'passed'
    except Exception as exc:
        import traceback
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished'] = time.time()
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
