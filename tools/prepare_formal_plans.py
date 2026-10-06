#!/usr/bin/env python3
"""Materialize explicit full-data budgets; never start training or overwrite plans."""
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.pipeline import sha256


def write(path, value):
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dep(run, steps, rows, params, minimum_tokens=0, **contract):
    return dict(run_dir=run, expected_steps=steps, min_trained_tokens=minimum_tokens,
                config_contract=contract,
                provenance_contract=dict(train_rows=rows, parameter_count=params))


def main():
    pre_corpus = 'data/processed/pretrain_t2t_full_v1'
    sft_corpus = 'data/processed/sft_t2t_full_v1'
    pre_rows = len(np.load(ROOT / pre_corpus / 'train.npy', mmap_mode='r'))
    targets = np.load(ROOT / sft_corpus / 'first_targets.npy', mmap_mode='r')
    sft_rows = np.load(ROOT / sft_corpus / 'train.npy', mmap_mode='r')
    sft_count = int((targets[sft_rows] < 1536).sum())
    expected_tokens = json.loads((ROOT/'reports/pretrain_data_audit.json').read_text())['recipes']['512']['estimated_supervised_train_tokens']
    baseline_pre = dep('models/01_moe/runs/pretrain_full_v1', math.ceil(pre_rows/144), pre_rows, 198416640,
                       int(expected_tokens*.999), corpus=pre_corpus, stage='pretrain', epochs=1, seq_len=512)
    baseline_sft = dep('models/01_moe/runs/sft_full_v1', math.ceil(sft_count/16), sft_count, 198416640,
                       corpus=sft_corpus, stage='sft', epochs=1, seq_len=1536)
    shared = dict(model=dict(hidden_size=768,num_hidden_layers=8,use_moe=True), strict_init=True,
                  epochs=1,num_workers=2,cpu_threads=4,deterministic=False,validation_samples=512,
                  warmup_steps=100,eval_interval=1000,save_interval=1000,log_interval=50,
                  milestone_interval=10000,early_checkpoint_steps=[10,100],plot_on_save=True,
                  allocator_conf='expandable_segments:True',allocator_gib=5.3,purpose='formal')
    stages = []
    transfer = 'models/02_hybrid_moe/runs/transfer_formal_v1'
    stages.append(dict(id='transfer_hybrid', kind='transfer', model_kind='hybrid', out=transfer,
                       source=baseline_pre['run_dir']+'/checkpoints/best_validation.pth', seed=20261015,
                       dependencies=[baseline_pre,baseline_sft]))
    paths = []
    for family, directory, params in [('hybrid','models/02_hybrid_moe',205623072),('control','models/01_moe',198416640)]:
        previous = transfer+'/initialized_fp32.pth' if family=='hybrid' else baseline_pre['run_dir']+'/checkpoints/best_validation.pth'
        for phase in ['adapt','continue','sft']:
            sft = phase=='sft'
            run_id = f'{family}_{phase}_full_v1'
            config_path = f'{directory}/configs/{run_id}.json'
            config = dict(shared, model_file=('models/02_hybrid_moe/src/model_hybrid.py' if family=='hybrid' else 'upstream/model/model_minimind.py'),
                          run_dir=f'{directory}/runs/{run_id}', init_weight=previous,
                          stage='sft' if sft else 'pretrain', corpus=sft_corpus if sft else pre_corpus,
                          batch_size=4,accumulation_steps=4 if sft else 6,seq_len=1536 if sft else 512,
                          checkpointing=sft,learning_rate=1e-5 if sft else 1e-4,
                          seed={'adapt':20261015,'continue':20261016,'sft':20261004}[phase],
                          validation_seed=20261004+1000003 if sft else 20261003+1000003,
                          recipe_status='Authorized full architecture/data experiment. Adapt is a bounded transfer stage, not model completion. Matched control uses the same sample order, budget, microbatch, LR, scheduler and checkpoint selection policy; frozen/new parameter sets differ by design.')
            if family=='hybrid':
                config['model']={**shared['model'],'require_fla':True}
                config.update(triton_f32_default='tf32x3',train_only_linear=phase=='adapt')
            if phase=='adapt':
                config.update(max_steps=5000,eval_interval=500,save_interval=500,milestone_interval=5000)
            count = sft_count if sft else pre_rows
            steps = config.get('max_steps') or math.ceil(count/(config['batch_size']*config['accumulation_steps']))
            completed = dep(config['run_dir'],steps,count,params,
                            int(expected_tokens*.999) if phase=='continue' else 0,
                            **config)
            # Config is kept verbatim in the contract, including all budget fields.
            completed['config_contract']=config
            stage=dict(id=run_id,kind='train',module='lab.train',config=config_path,completion=completed,
                       text_probes=phase!='adapt',dependencies=[])
            stages.append(stage);write(config_path,config);paths.append(config_path)
            # The full adaptation budget reaches both continuation branches;
            # selecting an earlier adaptation checkpoint would change exposure.
            previous=config['run_dir']+('/checkpoints/model_step_0005000.pth' if phase=='adapt' else '/checkpoints/best_validation.pth')
    source_paths=['lab/pipeline.py','lab/train.py','lab/data.py','lab/transfer.py','lab/checkpointing.py','lab/live_status.py',
                  'tools/run_formal_pipeline.py','tools/transfer_ar_checkpoint.py','tools/evaluate_text_probes.py','evaluation/text_probes_v1.json',
                  'upstream/model/model_minimind.py','upstream/model/tokenizer.json','models/02_hybrid_moe/src/model_hybrid.py']
    language=dict(version=1,controller_dir='runs/formal_hybrid_and_control_v1',min_free_gib=20,stages=stages,
                  fingerprints={name:sha256(ROOT/name) for name in paths+source_paths},
                  semantics='Pipeline completion means planned budgets and numerical contracts passed, not quality approved. Hybrid and AR control each receive 120000 adaptation examples plus one full extra pretrain epoch and one full SFT epoch.')
    write('plans/formal_hybrid_and_control_v1.json',language)

    omni_sources=['lab/omni_train.py','lab/omni_data.py','lab/omni_table.py','lab/omni_batch.py','lab/omni_loss.py',
                  'models/03_omni_moe/src/omni_dataset_role_aware.py','sources/minimind-o/model/model_omni.py',
                  'sources/minimind-o/model/model_minimind.py','sources/minimind-o/dataset/omni_dataset.py',
                  'sources/minimind-o/trainer/train_sft_omni.py','tools/monitor_host_resources.py','tools/server_gpu_guard.py',
                  'tools/evaluate_omni_probes.py','evaluation/omni_probes_v1.json','sources/minimind-o/trainer/trainer_utils.py']
    stages=[dict(id='transfer_omni',kind='transfer',model_kind='omni',out='models/03_omni_moe/runs/transfer_formal_v1',
                 source=baseline_sft['run_dir']+'/checkpoints/best_validation.pth',seed=20261017,dependencies=[baseline_sft])]
    previous=stages[0]['out']+'/initialized_fp32.pth'; paths=[]; budget=[]
    specs=[('t2a','all',6,5e-4,1536,16),('a2a','audio_proj',1,5e-4,3072,2),('a2a','all',3,5e-5,3072,2),
           ('i2t','vision_proj',1,5e-5,768,16),('i2t','all',1,5e-6,768,4),('a2a','all',1,5e-6,3072,2),('i2t','vision_proj',1,5e-6,768,16)]
    for number,(modality,mode,epochs,lr,length,batch) in enumerate(specs,1):
        name=f'omni_{number:02d}_{modality}_{mode}_full_v1'; corpus=f'data/processed/sft_{modality}_shards_v1'
        split='split_image_v2' if modality=='i2t' else 'audit_v1'
        index_name='visual_train.npy' if mode=='vision_proj' else 'train.npy'
        n=len(np.load(ROOT/corpus/split/index_name,mmap_mode='r'));steps=math.ceil(n/128)*epochs
        config=dict(run_dir=f'models/03_omni_moe/runs/{name}',purpose='formal',device='cuda',model=shared['model'],
                    modality=modality,mode=mode,corpus=corpus,init_weight=previous,max_length=length,batch_size=batch,
                    accumulation_steps=128//batch,epochs=epochs,learning_rate=lr,warmup_steps=100,seed=20261017+number,
                    cpu_threads=4,num_workers=2,prefetch_factor=1,deterministic=False,checkpointing=True,allocator_gib=13.5,
                    allocator_conf='expandable_segments:True',optimizer_state_offload=False,eval_interval=250,save_interval=250,
                    milestone_interval=5000,early_checkpoint_steps=[10,100],validation_samples=256,log_interval=10,inspect_first_batches=2,
                    recipe_status='Full pinned author seven-stage epochs/LRs and effective sample batch128. Single-GPU accumulation is not identical to four GPU microbatch32 MoE routing. Our longer speech contexts, holdout splits, warmup, loss-preserving harness and stage checkpoint selection are explicit recipe differences.')
        if modality=='a2a':config['audio_encoder']='data/frozen_models/SenseVoiceSmall'
        if modality=='i2t':config.update(vision_encoder='data/frozen_models/siglip2-base-p32-256-ve',split_subdir=split,
                                       train_index_name=index_name,val_index_name='visual_val.npy' if mode=='vision_proj' else 'val.npy',
                                       image_placeholder_policy='user_only')
        path=f'models/03_omni_moe/configs/{name}.json';write(path,config);paths.append(path)
        completion=dict(run_dir=config['run_dir'],expected_steps=steps,config_contract=config,
                        provenance_contract=dict(parameters=314887938,train_rows=n,purpose='formal'))
        probes=['t2a'] + (['a2a'] if number>=2 else []) + (['i2t'] if number>=4 else [])
        stages.append(dict(id=name,kind='train',module='lab.omni_train',config=path,completion=completion,dependencies=[],omni_probes=probes))
        previous=config['run_dir']+'/checkpoints/best_validation.pth'
        budget.append(dict(stage=name,rows=n,epochs=epochs,sample_presentations=n*epochs,updates=steps))
    write('plans/formal_omni_server_v1.json',dict(version=1,controller_dir='runs/formal_omni_server_v1',remote_gpu=2,min_free_gib=30,
          prerequisites=['Independent project Python/runtime and full-data hashes verified','Actual B16 accumulation8 T2A update check passed','Completed formal language SFT and immutable uploaded lineage','Idle GPU checked again at stage start'],
          stages=stages,budget=budget,fingerprints={name:sha256(ROOT/name) for name in paths+source_paths+omni_sources},
          semantics='Prepared plan; launch only after deployment and resource gates. Budget completion is not multimodal evaluation completion.'))
    print(json.dumps(dict(language_stages=len(language['stages']),pretrain_rows=pre_rows,sft_rows=sft_count,omni_budget=budget),indent=2))


if __name__=='__main__':main()
