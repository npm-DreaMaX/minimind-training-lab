#!/usr/bin/env python3
"""Retain pinned Omni stream outputs, exact Mimi codes and unclipped float WAV.

Uses the official sampling generator. Does not call it greedy, replace invalid
codec IDs silently, or equate teacher-forced CE with audible generation quality.
"""
import argparse,contextlib,fcntl,gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoTokenizer,MimiModel
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.omni_batch import official_modules
from lab.omni_data import official as official_dataset

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(8*1024**2): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--out',required=True)
    p.add_argument('--modality',choices=['t2a','a2a','i2t'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--suite',default='evaluation/omni_probes_v1.json'); p.add_argument('--reference',action='store_true')
    p.add_argument('--case',help='One case, retaining its original modality-local seed index')
    p.add_argument('--budget-floor',type=int,default=0)
    a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve prior outputs; choose a versioned destination')
    if a.device=='cuda':
        lock=(ROOT/'runs/local_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True); (out/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    suite=json.loads((ROOT/a.suite).read_text()); cases=[x for x in suite['cases'] if x['modality']==a.modality]
    if not cases: raise ValueError('Empty probe suite')
    expected_cases=sum(a.case is None or x['id']==a.case for x in cases)
    if not expected_cases: raise ValueError('Requested case is absent')
    started=time.perf_counter(); module,_=official_modules()
    audio_path=str(ROOT/'data/frozen_models/SenseVoiceSmall') if a.modality=='a2a' else '/nonexistent/no_audio_encoder_needed'
    vision_path=str(ROOT/'data/frozen_models/siglip2-base-p32-256-ve') if a.modality=='i2t' else None
    model=module.MiniMindOmni(module.OmniConfig(hidden_size=768,num_hidden_layers=8,use_moe=True),audio_encoder_path=audio_path,vision_model_path=vision_path)
    if a.modality=='a2a' and model.audio_encoder is None: raise RuntimeError('Missing actual audio encoder')
    if a.modality=='i2t' and model.vision_encoder is None: raise RuntimeError('Missing actual vision encoder')
    weight=ROOT/a.weights; weight_sha=sha(weight)
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True); model.load_state_dict(state,strict=True); del state; gc.collect()
    if sha(weight)!=weight_sha: raise RuntimeError('Weight changed during load')
    model.to(a.device).eval()
    for encoder in [model.audio_encoder,model.vision_encoder]:
        if encoder is not None: encoder.to(a.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
    codec,info=MimiModel.from_pretrained(ROOT/'data/frozen_models/mimi',torch_dtype=torch.float32,output_loading_info=True)
    if info['missing_keys'] or info['unexpected_keys'] or info['mismatched_keys']: raise RuntimeError(f'Mimi loading mismatch: {info}')
    codec.to(a.device).eval(); records=[]
    protocol=dict(device=a.device,parameters=sum(p.numel() for p in model.parameters()),parameter_dtype='FP32',
                  autocast='BF16 generation only' if a.device=='cuda' else 'disabled',codec_dtype='FP32',
                  text_sampling='official multinomial',temperature=.7,top_p=.85,repetition_penalty=1.,
                  audio_temperature=.2,audio_top_k=50,audio_recent_repetition_penalty=1.05,
                  seed=a.seed,open_thinking=False,reference_voice=None,use_cache=True,budget_floor=a.budget_floor,selected_case=a.case,
                  scope='Official pretrained reference only' if a.reference else 'Evaluation of explicitly supplied project checkpoint')
    try:
        for index,case in enumerate(cases):
            if a.case is not None and case['id']!=a.case: continue
            seed=a.seed+index; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            budget=max(case['max_new_tokens'],a.budget_floor)
            extra={}; assets={}; prompt=case.get('prompt',''); shapes={}
            if a.modality=='a2a':
                audio=ROOT/case['audio']; assets[case['audio']]=sha(audio)
                mel,length=official_dataset.OmniDataset.process_audio(str(audio),model.audio_processor)
                extra.update(audio_inputs=mel.unsqueeze(0).to(a.device),audio_lens=torch.tensor([length],device=a.device))
                prompt=model.config.audio_special_token*max(1,int(length)); shapes.update(fbank=list(mel.shape),valid_frames=int(length))
            if a.modality=='i2t':
                path=ROOT/case['image']; assets[case['image']]=sha(path)
                with Image.open(path) as img: pixels=model.vision_processor(images=img.convert('RGB'),return_tensors='pt')
                extra['pixel_values']={k:v.to(a.device) for k,v in pixels.items()}
                prompt+='\n\n'+model.config.image_special_token*model.config.image_token_len
                shapes['pixels']={k:list(v.shape) for k,v in pixels.items()}
            rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
            ids=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to(a.device)
            frames=[]; text_ids=[]; yielded=0; generation_start=time.perf_counter()
            with torch.inference_mode(),(torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext()):
                stream=model.generate(ids,eos_token_id=tokenizer.eos_token_id,max_new_tokens=budget,temperature=.7,top_p=.85,
                                      stream=True,return_audio_codes=True,open_thinking=False,rp=1.,use_cache=True,**extra)
                for text,frame in stream:
                    yielded+=1
                    if text is not None: text_ids=text[0].cpu().tolist()
                    if frame is not None:
                        if len(frame)!=8 or any(not 0<=int(c)<2048 for c in frame): raise RuntimeError(f'Invalid emitted Mimi frame: {frame}')
                        frames.append(frame)
            generation_seconds=time.perf_counter()-generation_start
            code_array=np.asarray(frames,dtype=np.uint16).reshape(-1,8); np.save(out/(case['id']+'_codes.npy'),code_array)
            audio_metrics=dict(frames=len(frames),empty_audio=not bool(frames))
            if frames:
                codes=torch.as_tensor(code_array.astype(np.int64).T.copy(),device=a.device).unsqueeze(0)
                decode_start=time.perf_counter()
                with torch.inference_mode(): waveform=codec.decode(codes).audio_values.squeeze().float().cpu().numpy()
                if not np.isfinite(waveform).all(): raise RuntimeError('Nonfinite decoded waveform')
                wav=out/(case['id']+'.wav'); sf.write(wav,waveform,codec.config.sampling_rate,subtype='FLOAT')
                audio_metrics.update(wav=str(wav.relative_to(ROOT)),sampling_rate=codec.config.sampling_rate,seconds=len(waveform)/codec.config.sampling_rate,
                                     peak_amplitude=float(np.abs(waveform).max()),rms=float(np.sqrt(np.mean(waveform**2))),
                                     fraction_abs_over_one=float(np.mean(np.abs(waveform)>1)),decode_seconds=time.perf_counter()-decode_start)
            row=dict(id=case['id'],case=case,seed=seed,assets=assets,input_shapes=shapes,input_ids=ids[0].cpu().tolist(),rendered_input=rendered,
                     text_ids=text_ids,text=tokenizer.decode(text_ids,skip_special_tokens=True),text_with_special_tokens=tokenizer.decode(text_ids,skip_special_tokens=False),
                     text_has_eos=tokenizer.eos_token_id in text_ids,yielded_steps=yielded,generation_seconds=generation_seconds,
                     max_new_tokens=budget,reached_step_budget=yielded>=budget,audio=audio_metrics)
            records.append(row)
            with (out/'outputs.jsonl').open('a') as stream: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(dict(id=case['id'],text=row['text'],audio=audio_metrics,reached_step_budget=row['reached_step_budget']),ensure_ascii=False),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n'); raise
    finally:
        report=dict(complete=len(records)==expected_cases,cases=len(records),modality=a.modality,suite_scope=suite['scope'],protocol=protocol,
                    weights=a.weights,weights_sha256=weight_sha,model_sha256=sha(ROOT/'sources/minimind-o/model/model_omni.py'),
                    tokenizer_sha256=sha(ROOT/'upstream/model/tokenizer.json'),suite_sha256=sha(ROOT/a.suite),script_sha256=sha(Path(__file__)),
                    seconds=time.perf_counter()-started,time=time.time(),note='No automatic semantic/audio correctness score; retain raw outputs for separate analysis')
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
