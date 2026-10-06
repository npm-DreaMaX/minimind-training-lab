"""Pinned official collator; import from its source without executing the CLI."""
import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def official_modules():
    # The upstream trainer imports `model.*` and `trainer.*` by package name.
    # This process is dedicated to Omni; never import another checkout as model.
    sys.path.insert(0,str(ROOT/'sources/minimind-o'))
    import model.model_omni as omni
    spec=importlib.util.spec_from_file_location('official_omni_trainer',ROOT/'sources/minimind-o/trainer/train_sft_omni.py')
    trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    return omni,trainer.omni_collate_fn

def to_device(batch,device):
    def move(value):
        if isinstance(value,dict): return {key:move(x) for key,x in value.items()}
        return value.to(device,non_blocking=True) if hasattr(value,'to') else value
    return tuple(move(value) for value in batch)
