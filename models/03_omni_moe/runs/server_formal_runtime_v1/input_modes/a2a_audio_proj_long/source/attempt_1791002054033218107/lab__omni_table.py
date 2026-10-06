"""Random access to bounded, memory-mapped Arrow shards; original scalar API."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

class ShardedOmniTable:
    def __init__(self,roots,cache_size=8):
        self.shards=[]; self.ends=[]; self.column_names=[]; self.cache=OrderedDict(); self.cache_size=cache_size
        total=0
        for root in roots:
            root=Path(root); meta=json.loads((root/'index.json').read_text())
            for name in meta['columns']:
                if name not in self.column_names: self.column_names.append(name)
            for shard in meta['shards']:
                self.shards.append(root/shard['file']); total+=shard['rows']; self.ends.append(total)
        self.total=total

    def __len__(self): return self.total

    def __getitem__(self,column):
        if column not in self.column_names: raise KeyError(column)
        return ShardedColumn(self,column)

    def scalar(self,column,row):
        if row<0: row+=self.total
        if not 0<=row<self.total: raise IndexError(row)
        shard=bisect_right(self.ends,row); offset=row-(self.ends[shard-1] if shard else 0)
        if shard not in self.cache:
            mapping=pa.memory_map(str(self.shards[shard]),'r'); reader=ipc.open_file(mapping)
            if reader.num_record_batches!=1: raise RuntimeError('Each audited shard must contain one record batch')
            batch=reader.get_batch(0); self.cache[shard]=(mapping,reader,batch)
            while len(self.cache)>self.cache_size:
                _,(old_map,old_reader,old_batch)=self.cache.popitem(last=False)
                del old_reader,old_batch; old_map.close()
        else: self.cache.move_to_end(shard)
        batch=self.cache[shard][2]; index=batch.schema.get_field_index(column)
        return pa.scalar(None) if index<0 else batch.column(index)[offset]

    def __getstate__(self):
        state=self.__dict__.copy(); state['cache']=OrderedDict(); return state

class ShardedColumn:
    def __init__(self,table,name): self.table,self.name=table,name
    def __getitem__(self,row): return self.table.scalar(self.name,row)
