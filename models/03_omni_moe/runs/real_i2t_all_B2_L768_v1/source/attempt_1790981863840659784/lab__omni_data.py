"""Reuse the pinned official Omni Dataset logic with a bounded storage backend."""
import importlib.util
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from lab.omni_table import ShardedOmniTable
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_omni_dataset',ROOT/'sources/minimind-o/dataset/omni_dataset.py')
official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
_role_aware=None

def role_aware_dataset():
    global _role_aware
    if _role_aware is None:
        spec=importlib.util.spec_from_file_location('role_aware_omni_dataset',ROOT/'models/03_omni_moe/src/omni_dataset_role_aware.py')
        _role_aware=importlib.util.module_from_spec(spec); spec.loader.exec_module(_role_aware)
    return _role_aware.OmniDataset

class StreamingOmniDataset(official.OmniDataset):
    def __init__(self,data_path,*args,**kwargs):
        roots=[Path(path.strip()) for path in str(data_path).split(',')]
        # The original constructor only initializes token IDs and augmentation
        # settings; one real schema sample avoids materializing the full corpus.
        super().__init__(str(roots[0]/'schema_sample.parquet'),*args,**kwargs)
        self.table=ShardedOmniTable(roots)


class ReproducibleOmniDataset(StreamingOmniDataset):
    """Per-record, per-epoch augmentation; independent of worker prefetch/resume.

    This preserves official augmentation operations and distributions, but
    changes their random-number assignment. It is a recorded recipe choice.
    """
    def __init__(self,*args,rows=None,seed=20261006,image_placeholder_policy='official',**kwargs):
        if image_placeholder_policy not in ['official','user_only']: raise ValueError('Unknown image placeholder policy')
        self.image_placeholder_policy=image_placeholder_policy
        super().__init__(*args,**kwargs)
        self.rows=rows; self.seed=seed

    def create_chat_prompt(self,*args,**kwargs):
        if self.image_placeholder_policy=='user_only':
            return role_aware_dataset().create_chat_prompt(self,*args,**kwargs)
        return super().create_chat_prompt(*args,**kwargs)

    def __len__(self): return len(self.rows) if self.rows is not None else super().__len__()

    def __getitem__(self,index):
        epoch,position=index if isinstance(index,tuple) else (0,index)
        row=int(self.rows[position]) if self.rows is not None else int(position)
        seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{epoch}:{row}'.encode()).digest()[:8],'little')
        python_rng=random.getstate(); numpy_rng=np.random.get_state(); cpu_rng=torch.get_rng_state()
        try:
            random.seed(seed); np.random.seed(seed%2**32)
            # Do not call torch.manual_seed(): it also changes CUDA RNG in the
            # main process when num_workers=0. Augmentation uses CPU only.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if self.image_placeholder_policy=='user_only': return role_aware_dataset().__getitem__(self,row)
            return super().__getitem__(row)
        finally:
            random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(cpu_rng)


class EpochTaggedSampler:
    def __init__(self,sampler,epoch): self.sampler,self.epoch=sampler,epoch
    def __iter__(self):
        for batch in self.sampler: yield [(self.epoch,row) for row in batch]
    def __len__(self): return len(self.sampler)
