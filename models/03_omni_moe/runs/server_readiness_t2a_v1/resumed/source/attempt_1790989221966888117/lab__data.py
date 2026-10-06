"""Memory-mapped token corpus; official per-record loss semantics, explicit cropping."""
from pathlib import Path
import json
import numpy as np
import torch


class TokenCorpus:
    def __init__(self, root, split, seq_len, stage):
        self.root=Path(root)
        self.meta=json.loads((self.root/'metadata.json').read_text())
        self.tokens=np.memmap(self.root/'tokens.bin',dtype='<u2',mode='r')
        self.offsets=np.load(self.root/'offsets.npy',mmap_mode='r')
        self.rows=np.load(self.root/(split+'.npy'),mmap_mode='r')
        self.original_rows=len(self.rows)
        if stage=='sft':
            first_targets=np.load(self.root/'first_targets.npy',mmap_mode='r')
            self.rows=self.rows[first_targets[self.rows]<seq_len]
        self.labels=(np.memmap(self.root/'labels.bin',dtype='<i2',mode='r') if stage=='sft' else None)
        self.seq_len=seq_len
        self.stage=stage

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=int(self.rows[index]); start,end=map(int,self.offsets[row:row+2])
        if self.stage=='pretrain':
            # Same BOS/text/EOS and prefix truncation as official PretrainDataset.
            ids=np.concatenate(([1],self.tokens[start:min(end,start+self.seq_len-2)],[2])).astype(np.int64)
            labels=ids.copy()
            labels[ids==0]=-100
        else:
            end=min(end,start+self.seq_len)
            ids=self.tokens[start:end].astype(np.int64)
            labels=self.labels[start:end].astype(np.int64)
        pad=self.seq_len-len(ids)
        if pad:
            ids=np.pad(ids,(0,pad),constant_values=0)
            labels=np.pad(labels,(0,pad),constant_values=-100)
        return torch.from_numpy(ids),torch.from_numpy(labels)


class EpochBatchSampler:
    """Shuffle using a private generator; resume directly at next unseen sample."""
    def __init__(self,length,batch_size,seed,epoch=0,start=0):
        self.length,self.batch_size,self.seed,self.epoch,self.start=length,batch_size,seed,epoch,start

    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        order=torch.randperm(self.length,generator=g)
        for i in range(self.start,self.length,self.batch_size):
            yield order[i:i+self.batch_size].tolist()

    def __len__(self):
        return (self.length-self.start+self.batch_size-1)//self.batch_size
