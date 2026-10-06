#!/usr/bin/env python3
"""Small, traceable real input fixtures and pinned encoders for remote preflight.

The corpus is explicitly a fixture. Remapped row numbers change augmentation
seed assignment; these runs are not paired population throughput comparisons.
"""
import hashlib, json, shutil, sys, tarfile, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.omni_table import ShardedOmniTable


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    dest = ROOT / 'artifacts/omni_validation/server_inputs_v1'
    dest.mkdir(exist_ok=False)
    rng = np.random.default_rng(20261013)
    selections = {}
    for modality in ['a2a', 'i2t']:
        source = ROOT / f'data/processed/sft_{modality}_shards_v1'
        split = source / ('audit_v1' if modality == 'a2a' else 'split_image_v2')
        train = np.load(split / ('train.npy' if modality == 'a2a' else 'visual_train.npy'), mmap_mode='r')
        val = np.load(source / 'evaluation_audio_unseen_v2/val.npy', mmap_mode='r') if modality == 'a2a' else np.load(split/'visual_val.npy', mmap_mode='r')
        train_rows = rng.choice(train, size=48, replace=False).tolist()
        if modality == 'a2a':
            tail = json.loads((ROOT/'reports/omni_a2a_batches_L3072_v1/report.json').read_text())['long_gpu_preflight_rows']
            assert np.isin(tail, train).all()
            train_rows = list(dict.fromkeys(train_rows + tail))
        val_rows = rng.choice(val, size=16, replace=False).tolist()
        rows = train_rows + val_rows
        assert len(set(rows)) == len(rows)
        backend = ShardedOmniTable([source], cache_size=2)
        with pa.memory_map(str(backend.shards[0]), 'r') as mm:
            schema = ipc.open_file(mm).schema
        payload = [{c:backend[c][row].as_py() for c in backend.column_names} for row in rows]
        table = pa.Table.from_pylist(payload, schema=schema)
        out = dest / modality
        out.mkdir(); (out/'audit_v1').mkdir()
        with pa.OSFile(str(out/'shard_000000.arrow'), 'wb') as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table.combine_chunks(), max_chunksize=len(rows))
        pq.write_table(table.slice(0, 1), out/'schema_sample.parquet')
        np.save(out/'audit_v1/train.npy', np.arange(len(train_rows), dtype='<u8'))
        np.save(out/'audit_v1/val.npy', np.arange(len(train_rows), len(rows), dtype='<u8'))
        index = dict(scope='fixture', rows=len(rows), columns=backend.column_names,
                     shards=[dict(file='shard_000000.arrow', rows=len(rows), bytes=(out/'shard_000000.arrow').stat().st_size)],
                     source_corpus=str(source.relative_to(ROOT)), source_index_sha256=sha(source/'index.json'),
                     raw_rows=rows, limitations='Reindexed real records; augmentation seed identity differs from full source; only short hardware/path validation')
        (out/'index.json').write_text(json.dumps(index, indent=2)+'\n')
        meta=dict(rows=len(rows), train_rows=len(train_rows), val_rows=len(val_rows), scope='fixture', source_split=str(split.relative_to(ROOT)))
        (out/'audit_v1/metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
        selections[modality] = index
        # Check serialized scalars against the original source, including raw bytes.
        replay = ShardedOmniTable([out])
        assert all(replay[c][i].as_py() == backend[c][row].as_py() for i,row in enumerate(rows) for c in backend.column_names)
        del payload, table, backend, replay
    files = [p for p in dest.rglob('*') if p.is_file()]
    frozen = json.loads((ROOT/'data/manifests/omni_frozen_models.json').read_text())
    for name in ['SenseVoiceSmall','siglip2-base-p32-256-ve']:
        for filename, entry in frozen[name]['files'].items():
            path = ROOT/'data/frozen_models'/name/filename
            if filename == '.gitattributes':
                continue
            assert path.stat().st_size == entry['bytes']
            if entry['sha256']:
                assert sha(path) == entry['sha256']
            files.append(path)
    files.extend([ROOT/'models/03_omni_moe/src/omni_dataset_role_aware.py', ROOT/'data/manifests/omni_frozen_models.json'])
    report=dict(scope='Actual bounded audio and visual input fixtures; pinned frozen encoders; no formal dataset replacement',
                selections=selections, files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
                script_sha256=sha(Path(__file__)),time=time.time())
    report_path=ROOT/'reports/server_omni_inputs_bundle_v1.json'
    if report_path.exists():raise RuntimeError('Preserve earlier manifest')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    archive=ROOT/'artifacts/server_omni_inputs_v1.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for p in files+[report_path]:tar.add(p,arcname=p.relative_to(ROOT))
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))


if __name__=='__main__':main()
