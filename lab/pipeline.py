"""Budget and lineage checks for staged experiments. No torch import or GPU work."""
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_completed(root, dependency):
    """A trainer's 'complete' alone does not prove the agreed budget was used."""
    run = Path(root) / dependency['run_dir']
    status = json.loads((run / 'status.json').read_text())
    if status.get('status') != 'complete':
        raise ValueError(f'Stage not complete: {run}')
    config = json.loads((run / 'config.json').read_text())
    provenance = json.loads((run / 'provenance.json').read_text())
    for key, expected in dependency.get('config_contract', {}).items():
        if config.get(key) != expected:
            raise ValueError(f'Configuration contract mismatch: {run}: {key}')
    for key, expected in dependency.get('provenance_contract', {}).items():
        if provenance.get(key) != expected:
            raise ValueError(f'Provenance contract mismatch: {run}: {key}')
    expected = dependency['expected_steps']
    if status.get('step') != expected or provenance.get('total_steps') != expected:
        raise ValueError(f'Incomplete/wrong update budget: {run}')
    if status.get('trained_tokens', 0) < dependency.get('min_trained_tokens', 0):
        raise ValueError(f'Incomplete token budget: {run}')
    field = 'selection_score' if 'selection_score' in status['validation'] else 'validation_ce'
    if not math.isfinite(status['validation'][field]):
        raise ValueError(f'Nonfinite final validation: {run}')
    if not status['validation'].get('final'):
        raise ValueError(f'Missing full final validation: {run}')
    best = json.loads((run / 'checkpoints/best_validation.json').read_text())
    if not math.isfinite(best[field]) or best['step'] <= 0:
        raise ValueError(f'No trained checkpoint improved initial validation: {run}')
    if best['step'] > status['step']:
        raise ValueError(f'Invalid selected checkpoint: {run}')
    weight = run / 'checkpoints/best_validation.pth'
    if not weight.is_file() or not (run / 'checkpoints/latest_resume.pt').is_file():
        raise ValueError(f'Missing learned weights or resume state: {run}')
    return dict(run_dir=dependency['run_dir'], completed_status=status,
                selection=best, selected_weight_sha256=sha256(weight),
                quality='Numerical/budget checks only; generation and task quality require analysis')


def assert_pinned(root, fingerprints):
    for name, expected in fingerprints.items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f'Pinned pipeline input changed: {name}')
