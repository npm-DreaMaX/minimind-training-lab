#!/usr/bin/env python3
"""Persistent, fail-closed stage execution after completed formal dependencies.

This orchestrates existing trainers. It neither edits recipes on failure nor
calls a completed optimizer budget a successful model. Source/config fingerprints
are checked again after every wait. Completed stages can be revalidated on restart.
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.live_status import read_live_json
from lab.pipeline import assert_pinned, check_completed, sha256


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    assert_pinned(ROOT, plan['fingerprints'])
    if args.check_only:
        print(json.dumps(dict(passed=True, plan=args.plan, stages=len(plan['stages']),
                             scope='Pinned sources/configs only; dependencies and GPU are not yet ready')))
        return
    out = ROOT / plan['controller_dir']
    if out.exists() and not args.resume:
        raise RuntimeError('Existing controller evidence requires --resume')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'plan.json').exists():
        if json.loads((out / 'plan.json').read_text()) != plan:
            raise RuntimeError('Controller plan changed; preserve and version it')
    else:
        write_json(out / 'plan.json', plan)
    shutil.copy2(__file__, out / f'source_{time.time_ns()}.py')
    stopped = False
    child = None
    def request_stop(*_):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def record(event, **fields):
        value = dict(event=event, time=time.time(), pid=os.getpid(), **fields)
        with (out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        write_json(out / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    def wait_dependencies(dependencies):
        evidence = []
        for dep in dependencies:
            record('waiting_dependency', run_dir=dep['run_dir'])
            while not stopped:
                status = read_live_json(ROOT / dep['run_dir'] / 'status.json', missing_ok=True)
                if status and status.get('status') == 'failed':
                    raise RuntimeError(f'Dependency failed: {dep["run_dir"]}')
                if status and status.get('status') == 'complete':
                    evidence.append(check_completed(ROOT, dep))
                    break
                time.sleep(15)
            if stopped:
                return None
        assert_pinned(ROOT, plan['fingerprints'])
        return evidence
    def execute(command, label, env):
        nonlocal child
        # Acquire/release the actual project GPU lock: parent process IDs become
        # stale across resume. Each trainer/evaluator takes the same lock itself.
        with (ROOT / 'runs/local_gpu.lock').open('a') as gpu_lock:
            while not stopped:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(5)
            if stopped:
                return
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        if shutil.disk_usage(ROOT).free < plan.get('min_free_gib', 15) * 1024**3:
            raise RuntimeError('Insufficient checkpoint headroom; no evidence deleted')
        if plan.get('remote_gpu') is not None:
            from server_gpu_guard import wait_for_idle
            wait_for_idle(plan['remote_gpu'], lambda s: record('gpu_observation', **s))
        log = out / f'{label}_{time.time_ns()}.log'
        record('command_start', command=command, log=str(log.relative_to(ROOT)))
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        record('command_exit', command=command, returncode=code)
        if code and not stopped:
            raise RuntimeError(f'Command failed; retained {log}')
    try:
        record('started', plan=args.plan)
        for stage in plan['stages']:
            evidence = wait_dependencies(stage.get('dependencies', []))
            if stopped:
                break
            write_json(out / f'{stage["id"]}_lineage.json', evidence)
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **stage.get('environment', {})}
            if plan.get('remote_gpu') is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(plan['remote_gpu'])
            if stage['kind'] == 'transfer':
                destination = ROOT / stage['out']
                if (destination / 'transfer.json').exists():
                    report = json.loads((destination / 'transfer.json').read_text())
                    if (report['source_sha256'] != sha256(ROOT / stage['source'])
                            or report['output_sha256'] != sha256(destination / 'initialized_fp32.pth')):
                        raise RuntimeError('Existing transfer does not match lineage/weights')
                else:
                    execute([sys.executable, '-u', '-B', 'tools/transfer_ar_checkpoint.py',
                             '--kind', stage['model_kind'], '--source', stage['source'],
                             '--out', stage['out'], '--seed', str(stage['seed'])], stage['id'], env)
            elif stage['kind'] == 'train':
                config = json.loads((ROOT / stage['config']).read_text())
                if config.get('triton_f32_default'):
                    env['TRITON_F32_DEFAULT'] = config['triton_f32_default']
                if config.get('allocator_conf'):
                    env['PYTORCH_ALLOC_CONF'] = config['allocator_conf']
                run = ROOT / config['run_dir']
                status = read_live_json(run / 'status.json', missing_ok=True) or {}
                if status.get('status') == 'failed':
                    raise RuntimeError(f'Failed stage needs diagnosis: {run}')
                if status.get('status') != 'complete':
                    command = [sys.executable, '-u', '-B', '-m', stage['module'], '--config', stage['config']]
                    if (run / 'metrics.jsonl').exists():
                        if not (run / 'checkpoints/latest_resume.pt').exists():
                            raise RuntimeError('Partial stage without recovery checkpoint; preserve for diagnosis')
                        command.append('--resume')
                    execute(command, stage['id'], env)
                if stopped:
                    break
                completed = check_completed(ROOT, stage['completion'])
                write_json(out / f'{stage["id"]}_completed.json', completed)
                if stage.get('text_probes'):
                    dest = run / 'evaluation/final_selected_gpu_probes_v1'
                    if not (dest / 'report.json').exists():
                        execute([sys.executable, '-u', '-B', 'tools/evaluate_text_probes.py',
                                 '--config', stage['config'], '--weights', str((run / 'checkpoints/best_validation.pth').relative_to(ROOT)),
                                 '--stage', config['stage'], '--out', str(dest.relative_to(ROOT)), '--device', 'cuda'],
                                stage['id'] + '_generation', env)
                    report = json.loads((dest / 'report.json').read_text())
                    if not report.get('complete'):
                        raise RuntimeError('Incomplete generation suite; retained partial output')
                for modality in stage.get('omni_probes', []):
                    dest = run / f'evaluation/final_selected_{modality}_gpu_probes_v1'
                    if not (dest / 'report.json').exists():
                        execute([sys.executable, '-u', '-B', 'tools/evaluate_omni_probes.py',
                                 '--weights', str((run / 'checkpoints/best_validation.pth').relative_to(ROOT)),
                                 '--modality', modality, '--out', str(dest.relative_to(ROOT)), '--device', 'cuda'],
                                stage['id'] + '_' + modality + '_generation', env)
                    if not json.loads((dest / 'report.json').read_text()).get('complete'):
                        raise RuntimeError('Incomplete multimodal generation; retained partial output')
            else:
                raise ValueError(stage['kind'])
            if stopped:
                break
            record('stage_budget_complete', stage=stage['id'], quality_review_pending=True)
        record('paused' if stopped else 'pipeline_budget_complete', quality_review_pending=True)
    except Exception as exc:
        record('failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
