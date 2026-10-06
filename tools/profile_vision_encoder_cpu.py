#!/usr/bin/env python3
"""Verify the exact frozen vision checkpoint and real repository images on CPU."""
import json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import SiglipVisionModel,SiglipImageProcessor
ROOT=Path(__file__).resolve().parents[1]; torch.set_num_threads(4)
path=ROOT/'data/frozen_models/siglip2-base-p32-256-ve'; assert (path/'verified.json').exists()
started=time.time(); model,info=SiglipVisionModel.from_pretrained(path,torch_dtype=torch.float32,output_loading_info=True)
assert not any(info.values()),info
model.eval(); model.requires_grad_(False); processor=SiglipImageProcessor.from_pretrained(path)
report=dict(scope='Frozen encoder only, real repository evaluation images, CPU; no multimodal LLM trained',
            parameters=sum(p.numel() for p in model.parameters()),loading_info=info,load_seconds=time.time()-started,threads=4,samples=[])
with torch.inference_mode():
    for image_path in sorted((ROOT/'sources/minimind-o/dataset/eval_omni').glob('image-*.jpg'))[:3]:
        image=Image.open(image_path).convert('RGB'); inputs=processor(images=image,return_tensors='pt')
        started=time.perf_counter(); output=model(**inputs).last_hidden_state; seconds=time.perf_counter()-started
        assert list(output.shape)==[1,64,768] and torch.isfinite(output).all()
        record=dict(file=str(image_path.relative_to(ROOT)),input_shape=list(inputs.pixel_values.shape),output_shape=list(output.shape),seconds=seconds)
        report['samples'].append(record); print(json.dumps(record),flush=True)
out=ROOT/'models/03_omni_moe/runs/vision_encoder_cpu'; out.mkdir(parents=True,exist_ok=True)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
