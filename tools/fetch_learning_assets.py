#!/usr/bin/env python3
"""Fetch explicitly requested public learning assets with complete SHA256 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'weight': {
        'url': 'https://github.com/npm-DreaMaX/minimind-training-lab/releases/download/v1.0-learning/hybrid_moe_205m_best_validation.pth',
        'path': 'models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/checkpoints/best_validation.pth',
        'sha256': '1fc422dfac3b784e319676e251a58fdb1eb0eea83def7936f6be4578a4cac896',
        'bytes': 421139037,
    },
    'raw-data': {
        'url': 'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/312afb4f76391145c6902f765bb51691c09a12f5/sft_t2t_mini.jsonl',
        'path': 'data/raw/minimind/sft_t2t_mini.jsonl',
        'sha256': 'abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518',
        'bytes': 1739201170,
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    item = ASSETS[name]
    target = ROOT / item['path']
    if target.exists():
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise RuntimeError(f'Existing file differs; refusing to replace: {target}')
        print(f'Already verified: {target}')
        if name == 'raw-data':
            target.with_name(target.name + '.verified.json').write_text(json.dumps({'bytes':item['bytes'],'sha256':item['sha256'],'url':item['url'],'verified':True},indent=2)+'\n')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.download.tmp')
    if temporary.exists():
        raise RuntimeError(f'Prior partial download retained: {temporary}. Inspect/remove it before retrying.')
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'MiniMind-learning-assets'})
    h = hashlib.sha256()
    total = 0
    last_report = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as out:
        while True:
            block = response.read(8 * 1024**2)
            if not block:
                break
            out.write(block)
            h.update(block)
            total += len(block)
            if total - last_report >= 64 * 1024**2:
                print(f'{name}: {total / 1024**2:.0f} / {item["bytes"] / 1024**2:.0f} MiB', flush=True)
                last_report = total
    if total != item['bytes'] or h.hexdigest() != item['sha256']:
        raise RuntimeError(f'Size/SHA256 mismatch; partial file retained at {temporary}')
    temporary.replace(target)
    if name == 'raw-data':
        target.with_name(target.name + '.verified.json').write_text(json.dumps({'bytes':total,'sha256':h.hexdigest(),'url':item['url'],'verified':True},indent=2)+'\n')
    print(f'Verified: {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=ASSETS)
    args = parser.parse_args()
    fetch(args.asset)
