#!/usr/bin/env python3
"""Read pinned remote Parquet footers with bounded ranges; do not load full tables."""
import concurrent.futures,io,json,time,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data/manifests/minimind_omni_full.json').read_text())

class RemoteFooter(io.RawIOBase):
    def __init__(self,name,size):
        self.name=name; self.size=size; self.pos=0; self.bytes_read=0; self.cache={}
        self.url=f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{name}?download=true"
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset)
        if not 0<=self.pos<=self.size: raise ValueError('Invalid seek')
        return self.pos
    def read(self,n=-1):
        if n<0 or n>64*1024**2: raise ValueError('Only bounded footer reads are authorized by this inspector')
        n=min(n,self.size-self.pos); result=[]
        while n:
            take=min(n,4*1024**2); start=self.pos; end=start+take-1; key=(start,end)
            if key not in self.cache:
                for attempt in range(5):
                    try:
                        request=urllib.request.Request(self.url+f'&footer_offset={start}&n={take}',headers={'Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(request,timeout=60) as r:
                            assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
                            content=r.read(take+1)
                        if len(content)!=take: raise IOError('Truncated footer range')
                        self.cache[key]=content; self.bytes_read+=take; break
                    except Exception:
                        if attempt==4: raise
                        time.sleep(2+attempt)
            result.append(self.cache[key]); self.pos+=take; n-=take
        return b''.join(result)

def inspect(item):
    name,info=item; remote=RemoteFooter(name,info['bytes']); parquet=pq.ParquetFile(remote); meta=parquet.metadata
    columns={}; max_group=0
    for index in range(meta.num_row_groups):
        group=meta.row_group(index); max_group=max(max_group,group.total_byte_size)
        for c in range(group.num_columns):
            column=group.column(c); key=column.path_in_schema
            entry=columns.setdefault(key,dict(physical_type=column.physical_type,encoded_values=0,encoded_uncompressed_bytes=0,compressed_bytes=0,
                                              null_values_recorded=0,groups_with_null_statistics=0,numeric_min=None,numeric_max=None))
            entry['encoded_values']+=column.num_values
            entry['encoded_uncompressed_bytes']+=column.total_uncompressed_size
            entry['compressed_bytes']+=column.total_compressed_size
            stats=column.statistics
            if stats is not None and stats.null_count is not None:
                entry['null_values_recorded']+=stats.null_count; entry['groups_with_null_statistics']+=1
            if stats is not None and stats.has_min_max and column.physical_type in ['INT32','INT64','FLOAT','DOUBLE']:
                entry['numeric_min']=stats.min if entry['numeric_min'] is None else min(entry['numeric_min'],stats.min)
                entry['numeric_max']=stats.max if entry['numeric_max'] is None else max(entry['numeric_max'],stats.max)
    result={'name':name,'revision':manifest['revision'],'rows':meta.num_rows,'row_groups':meta.num_row_groups,
            'schema':str(parquet.schema_arrow),'columns':columns,'largest_row_group_encoded_bytes':max_group,
            'footer_network_bytes':remote.bytes_read,'file_bytes':info['bytes'],
            'limits':'Remote footer only, not whole-file checksum validation. Encoded uncompressed sizes and nested num_values are not exact Arrow RAM or non-null audio-code counts.'}
    print(json.dumps(result),flush=True); return name,result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(3) as pool: results=dict(pool.map(inspect,manifest['files'].items()))
    (ROOT/'reports/omni_parquet_footers.json').write_text(json.dumps(results,indent=2)+'\n')
