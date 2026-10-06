#!/usr/bin/env python3
"""Profile the real frozen SenseVoice encoder on repository audio, with no GPU use."""
import importlib,json,sys,time,types
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('omni_encoder_profile'); package.__path__=[str(ROOT/'sources/minimind-o/model')]
sys.modules[package.__name__]=package
module=importlib.import_module(package.__name__+'.model_omni')
torch.set_num_threads(4)
weights=ROOT/'data/frozen_models/SenseVoiceSmall'; assert (weights/'verified.json').exists()
start=time.time(); encoder,processor=module.MiniMindOmni.load_sensevoice(str(weights))
assert encoder is not None and all(not p.requires_grad for p in encoder.parameters())
record={'scope':'Real repository evaluation audio, frozen encoder on CPU only; no LLM update',
        'parameters':sum(p.numel() for p in encoder.parameters()),'load_seconds':time.time()-start,
        'threads':4,'dtype':str(next(encoder.parameters()).dtype),'samples':[]}
files=sorted((ROOT/'sources/minimind-o/dataset/eval_omni').glob('audio-zh-*.mp3'))[:3]
with torch.inference_mode():
    for i,path in enumerate(files):
        wav,sr=sf.read(path); wav=wav.mean(axis=1) if wav.ndim>1 else wav
        if sr!=16000: wav=librosa.resample(wav.astype(np.float32),orig_sr=sr,target_sr=16000)
        wav=wav.astype(np.float32); duration=len(wav)/16000
        before=time.perf_counter(); inputs=processor(wav,sampling_rate=16000); frontend_s=time.perf_counter()-before
        length=inputs.attention_mask.sum(dim=-1)
        before=time.perf_counter(); features,out_len=encoder(inputs.input_features,length); encode_s=time.perf_counter()-before
        assert torch.isfinite(features).all()
        fp16_relative_rms=float((features-features.half().float()).square().mean().sqrt()/features.square().mean().sqrt())
        row={'file':str(path.relative_to(ROOT)),'audio_seconds':duration,'fbank_shape':list(inputs.input_features.shape),
             'feature_shape':list(features.shape),'valid_frames':int(length[0]),'encoder_output_lengths':out_len.tolist(),
             'frontend_seconds':frontend_s,'encoder_seconds':encode_s,'encoder_real_time_factor':encode_s/duration,
             'fp16_cache_rounding_relative_rms':fp16_relative_rms,'first_sample_includes_warmup':i==0}
        record['samples'].append(row); print(json.dumps(row),flush=True)
out=ROOT/'models/03_omni_moe/runs/audio_encoder_cpu'; out.mkdir(parents=True,exist_ok=True)
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
