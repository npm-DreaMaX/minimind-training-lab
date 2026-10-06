#!/usr/bin/env python3
"""Stream Parquet to bounded Arrow IPC shards and losslessly store codes as uint16."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--out',required=True)
    p.add_argument('--allow-fixture',action='store_true'); p.add_argument('--batch-rows',type=int,default=128)
    p.add_argument('--shard-rows',type=int,default=4096); p.add_argument('--shard-mib',type=int,default=64)
    a=p.parse_args(); raw=Path(a.raw); out=Path(a.out)
    verified=Path(str(raw)+'.verified.json')
    if not a.allow_fixture and not verified.exists(): raise RuntimeError('Full Parquet requires a verified download')
    if not raw.is_file(): raise FileNotFoundError(raw)
    if (out/'index.json').exists(): raise RuntimeError('Already complete; do not overwrite a corpus')
    if out.exists() and any(out.iterdir()): raise RuntimeError('Preserve incomplete shards before retrying')
    out.mkdir(parents=True,exist_ok=True)
    source=pq.ParquetFile(raw,memory_map=True,pre_buffer=False)
    start=time.time(); pending=[]; nrows=nbytes=total=0; shards=[]; first=True; columns=None
    def flush():
        nonlocal pending,nrows,nbytes,total
        if not pending: return
        table=pa.Table.from_batches(pending).combine_chunks()
        free=shutil.disk_usage(out).free
        if free < table.nbytes+20*1024**3:
            raise RuntimeError(f'Insufficient disk headroom for decoded shards: free={free}, next_table={table.nbytes}; preserve 20GiB for active training/checkpoints, partial conversion retained')
        path=out/f'shard_{len(shards):06d}.arrow'
        with pa.OSFile(str(path)+'.tmp','wb') as f:
            with ipc.new_file(f,table.schema) as writer: writer.write_table(table,max_chunksize=table.num_rows)
        Path(str(path)+'.tmp').replace(path)
        total+=table.num_rows; shards.append({'file':path.name,'rows':table.num_rows,'bytes':path.stat().st_size})
        progress={'rows':total,'shards':len(shards),'written_bytes':sum(s['bytes'] for s in shards),'disk_free':shutil.disk_usage(out).free,'elapsed_s':time.time()-start}
        (out/'progress.json').write_text(json.dumps(progress)+'\n')
        if len(shards)%10==0: print(json.dumps(progress),flush=True)
        pending=[]; nrows=nbytes=0
    for batch in source.iter_batches(batch_size=a.batch_rows,use_threads=False):
        arrays=[]; fields=[]
        for field,array in zip(batch.schema,batch.columns):
            if field.name in ['answer_audios','ref_audios']:
                flat=array
                while pa.types.is_list(flat.type) or pa.types.is_large_list(flat.type): flat=pc.list_flatten(flat)
                extrema=pc.min_max(flat).as_py()
                if extrema['min'] is not None and (extrema['min']<0 or extrema['max']>=2048):
                    raise ValueError(f'Unexpected raw codec IDs: {field.name} {extrema}')
                dtype=pa.list_(pa.list_(pa.uint16())) if field.name=='answer_audios' else pa.list_(pa.uint16())
                array=array.cast(dtype,safe=True); field=field.with_type(dtype)
            arrays.append(array); fields.append(field)
        batch=pa.RecordBatch.from_arrays(arrays,schema=pa.schema(fields))
        if first:
            pq.write_table(pa.Table.from_batches([batch.slice(0,1)]),out/'schema_sample.parquet')
            columns=batch.schema.names; first=False
        if pending and (nrows+batch.num_rows>a.shard_rows or nbytes+batch.nbytes>a.shard_mib*1024**2): flush()
        pending.append(batch); nrows+=batch.num_rows; nbytes+=batch.nbytes
    flush()
    assert total==source.metadata.num_rows and total>0,(total,source.metadata.num_rows)
    meta={'raw':str(raw),'source_verification':json.loads(verified.read_text()) if verified.exists() else None,
          'scope':'fixture' if a.allow_fixture else 'full verified corpus','rows':total,'columns':columns,'shards':shards,
          'code_storage':'uint16, range checked 0..2047, exact integer conversion; all other values preserved',
          'elapsed_s':time.time()-start,'preprocessor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    temporary=out/'index.json.tmp'; temporary.write_text(json.dumps(meta,indent=2)+'\n'); temporary.replace(out/'index.json')
    print(json.dumps({k:v for k,v in meta.items() if k!='shards'}),flush=True)

if __name__=='__main__': main()
