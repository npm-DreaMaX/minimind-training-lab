#!/usr/bin/env python3
"""Auditable formal subset of complete SFT records, with no target truncation.

Length selection is a declared distribution change, not a quality filter or an
equivalent replacement for training the entire source corpus. Shared arrays stay
immutable; only sample indices and metadata are new.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train-rows', type=int, default=100000)
    p.add_argument('--seq-len', type=int, default=768)
    p.add_argument('--min-targets', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261003)
    a = p.parse_args()
    source = Path(a.source).resolve(); out = Path(a.out)
    if out.exists():
        raise RuntimeError('Preserve existing data artifacts; choose a new version')
    meta = json.loads((source / 'metadata.json').read_text())
    assert meta['stage'] == 'sft' and meta['limit'] == 0 and meta['bad_rows'] == 0
    offsets = np.load(source / 'offsets.npy', mmap_mode='r')
    lengths = np.diff(offsets)
    hashes = np.load(source / 'content_hash64.npy', mmap_mode='r')
    first = np.load(source / 'first_targets.npy', mmap_mode='r')
    labels = np.memmap(source / 'labels.bin', dtype='<i2', mode='r')
    rng = np.random.default_rng(a.seed)
    report = {}
    selected = {}
    started = time.time()
    for split in ['train', 'val']:
        rows = np.load(source / f'{split}.npy', mmap_mode='r')
        eligible = rows[(lengths[rows] <= a.seq_len) & (lengths[rows] >= a.min_targets + 1)
                        & (first[rows] < lengths[rows])]
        _, unique_indices = np.unique(hashes[eligible], return_index=True)
        pool = eligible[unique_indices]
        order = rng.permutation(len(pool)) if split == 'train' else np.arange(len(pool))
        chosen = []; supervision = []; rejected = 0
        for i in order:
            row = int(pool[i]); lo, hi = map(int, offsets[row:row + 2])
            # Model predicts positions 1..T-1: count exactly the trainer's labels.
            count = int(np.count_nonzero(labels[lo + 1:hi] != -100))
            if count < a.min_targets:
                rejected += 1
                continue
            chosen.append(row); supervision.append(count)
            if split == 'train' and len(chosen) == a.train_rows:
                break
        if split == 'train' and len(chosen) != a.train_rows:
            raise RuntimeError('Not enough complete, supervised, distinct training records')
        if not chosen:
            raise RuntimeError('Empty split')
        chosen = np.asarray(chosen, dtype='<u8')
        report[split] = dict(source_rows=len(rows), length_eligible_rows=len(eligible),
                             content_hash_unique_pool=len(pool), selected_rows=len(chosen),
                             checked_but_rejected_low_target_rows=rejected,
                             input_tokens=int(lengths[chosen].sum()),
                             supervised_tokens=int(sum(supervision)),
                             min_supervised_tokens=min(supervision),
                             length_percentiles={str(q): float(np.percentile(lengths[chosen], q)) for q in [0, 50, 90, 100]},
                             input_padding_fraction=1 - int(lengths[chosen].sum()) / (len(chosen) * a.seq_len),
                             targets_truncated=0)
        selected[split] = np.sort(chosen)
    if np.intersect1d(hashes[selected['train']], hashes[selected['val']]).size:
        raise RuntimeError('Train/validation content overlap')
    out.mkdir(parents=True)
    for name in ['tokens.bin', 'labels.bin', 'offsets.npy', 'first_targets.npy', 'lengths.npy', 'content_hash64.npy']:
        if (source / name).is_file():
            os.link(source / name, out / name)
    for split, rows in selected.items():
        np.save(out / f'{split}.npy', rows)
    metadata = dict(stage='sft', scope='Formal finite-budget complete-record subset; not the official mini file or a full-source epoch',
                    source=str(source), source_metadata_sha256=sha(source / 'metadata.json'),
                    tokenizer_sha256=meta['tokenizer_sha256'], augmentation=meta['augmentation'],
                    seed=a.seed, seq_len=a.seq_len, min_targets=a.min_targets,
                    train_rows=len(selected['train']), val_rows=len(selected['val']),
                    selection='Uniform without replacement from deduplicated length-eligible training records, then reject fewer than min_targets. Validation uses all eligible held-out records.',
                    deduplication='Existing 64-bit content hashes; not semantic deduplication or a proof against hash collisions.',
                    selection_bias='Only complete dialogs fitting the context window; long-dialog/tool-chain capabilities are not established by this experiment.',
                    counts=report, split='Inherited original content-hash split; explicit selected hash disjointness checked',
                    index_sha256={s: sha(out / f'{s}.npy') for s in selected},
                    storage='Read-only hardlinks of existing full cache; no repeated large arrays',
                    elapsed_seconds=time.time() - started)
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
