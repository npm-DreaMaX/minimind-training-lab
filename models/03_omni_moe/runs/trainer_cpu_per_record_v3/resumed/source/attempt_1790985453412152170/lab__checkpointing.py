"""Activation recomputation without changing parameter names or model source.

Non-reentrant checkpointing retains support for tuple outputs and frozen modules
whose inputs receive gradients from a projector. Dropout RNG is preserved.
"""
from functools import wraps
import torch
from torch.utils.checkpoint import checkpoint

def checkpoint_blocks(blocks):
    for block in blocks:
        if getattr(block,'_lab_checkpointed',False): continue
        original=block.forward
        def make_forward(module,forward):
            @wraps(forward)
            def wrapped(*args,**kwargs):
                if not module.training or not torch.is_grad_enabled(): return forward(*args,**kwargs)
                if kwargs.get('use_cache',False): raise ValueError('Training activation checkpointing requires use_cache=False')
                return checkpoint(forward,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return wrapped
        block.forward=make_forward(block,original)
        block._lab_checkpointed=True
