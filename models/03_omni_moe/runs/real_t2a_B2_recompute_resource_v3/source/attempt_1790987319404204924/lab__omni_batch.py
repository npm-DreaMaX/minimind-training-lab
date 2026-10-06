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

def supervision_statistics(batch):
    """Observe CPU labels before transfer; no RNG or change to loss reduction."""
    ids,text,audio,_,lengths,_,_=batch
    text_count=(text!=-100).sum(1)
    audio_count=(audio!=-100).sum(2)
    stops=(audio==2050).sum(2)
    audio_active=audio_count>0
    return dict(samples=ids.shape[0],zero_supervision_samples=int(((text_count==0)&(audio_count.sum(1)==0)).sum()),
                audio_samples_with_missing_stop=int((audio_active&(stops==0)).any(1).sum()),
                active_audio_heads=int(audio_active.sum()),audio_heads_missing_stop=int((audio_active&(stops==0)).sum()),
                audio_marker_tokens=int((ids[:,8]==16).sum()),image_marker_tokens=int((ids[:,8]==12).sum()),
                max_fbank_frames=int(lengths.max()) if lengths is not None and lengths.numel() else 0)
