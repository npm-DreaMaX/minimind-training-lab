#!/usr/bin/env python3
"""Read-only learning exercises for the completed Hybrid conversion experiment.

No optimizer, CUDA training, resume, or checkpoint writes are exposed here.
Trace computes CPU gradients on a disposable model, without updating parameters.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1'
CFG = ROOT / 'models/02_hybrid_moe/configs/hybrid_sft_official_mini_2ep_v1.json'
WEIGHT = RUN / 'checkpoints/best_validation.pth'


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics():
    with (RUN / 'metrics.jsonl').open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_summary():
    rows = [r for r in metrics() if r.get('event') == 'train']
    assert [r['step'] for r in rows] == list(range(1, 113083))
    assert sum(r['valid_tokens'] for r in rows) == 735833934
    emit({'实验': '官方198M SFT权重转换后的205M Hybrid，两轮全参数SFT；没有Hybrid预训练',
          '状态': json.loads((RUN / 'status.json').read_text()),
          '更新连续': True, '监督token精确累计': 735833934,
          '更新计算小时': sum(r['step_seconds'] for r in rows) / 3600,
          '参数': 205623072, '质量': '量化评估与原始生成样例见reports/hybrid_final_review_20261006.md',
          '权重': str(WEIGHT.relative_to(ROOT)),
          '本工具': '只读，未启动训练'})


def replay(step):
    row = None
    for r in metrics():
        if r.get('event') == 'train' and r['step'] == step:
            row = r
            break
    if row is None:
        raise ValueError(f'No recorded training update {step}')
    evaluation = None
    for p in sorted((RUN / 'evaluation').glob('step_*.json')):
        r = json.loads(p.read_text())
        if r['step'] <= step:
            evaluation = {'path': str(p.relative_to(ROOT)), 'step': r['step'],
                          'ce': r['validation_ce'], 'rows': r['validation_rows'],
                          'generation': [{'prompt': x['prompt'], 'continuation': x['continuation']}
                                         for x in r.get('generation', [])]}
    emit({'真实更新': row, '此前最近保存的评估': evaluation,
          '解释': {'loss': 'ce_loss + aux_loss', 'grad_norm': '裁剪前范数；阈值1',
                   'cursor': '当前epoch已消费的训练记录数；epoch从0开始',
                   'valid_tokens': '本次有效右移标签数，不是B*T',
                   'cuda_peak_allocated': '本次进程启动以来高水位，不是每步重置',
                   'generation': '来自评估保存步，不能假装每一步都有生成样例'},
          '这是历史重放': True})


def sample(index, length):
    from lab.data import TokenCorpus
    from transformers import AutoTokenizer
    cfg = json.loads(CFG.read_text())
    corpus = ROOT / cfg['corpus']
    if not (corpus / 'metadata.json').is_file():
        bundled = ROOT / 'data/examples/hybrid_sample_51023.json'
        if not bundled.is_file() or index != 51023 or length != 768:
            raise SystemExit('Full corpus is absent. The public bundle supports --index 51023; see SHARING_GUIDE.md for full-data download/tokenization.')
        import torch
        item = json.loads(bundled.read_text())
        assert item['dataset_index'] == index and item['seq_len'] == length
        ds = SimpleNamespace(rows={index:item['cache_row']}, source='bundled_real_formal_sample')
        tok = AutoTokenizer.from_pretrained(ROOT / 'upstream/model', local_files_only=True)
        return cfg, ds, tok, torch.tensor(item['input_ids'],dtype=torch.long), torch.tensor(item['labels'],dtype=torch.long)
    ds = TokenCorpus(ROOT / cfg['corpus'], 'train', length, 'sft')
    ds.source = 'full_formal_token_cache'
    if not 0 <= index < len(ds):
        raise ValueError('Dataset index outside this length-filtered dataset')
    x, y = ds[index]
    tok = AutoTokenizer.from_pretrained(ROOT / 'upstream/model', local_files_only=True)
    return cfg, ds, tok, x, y


def show_sample(index):
    cfg, ds, tok, x, y = sample(index, 768)
    emit({'dataset_index': index, 'cache_row': int(ds.rows[index]),
          'data_source': ds.source,
          'shape': list(x.shape), 'valid_shifted_labels': int((y[1:] != -100).sum()),
          'nonpadding_input': int((x != 0).sum()),
          'rendered_text': tok.decode(x[x != 0].tolist(), skip_special_tokens=False),
          'first_80_positions': [{'j': j, 'input_id': int(x[j]), 'label': int(y[j]),
                                 'token': tok.decode([int(x[j])]),
                                 'predicted_by_logit_position': j-1 if j > 0 and y[j] != -100 else None}
                                for j in range(min(80, len(x)))],
          '注意': '单token显示替换符不一定是乱码；完整序列解码才用于判断文本。'})


def load_cpu():
    if not WEIGHT.is_file():
        raise SystemExit('Final weight is absent. Download and verify it with: python tools/fetch_learning_assets.py weight')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    cfg = json.loads(CFG.read_text())
    path = ROOT / cfg['model_file']
    spec = importlib.util.spec_from_file_location('study_hybrid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This entry point is CPU-only; native recurrence was already used here.
    # Do not require CUDA/FLA merely to load it on a reader's CPU machine.
    cpu_config = {**cfg['model'], 'require_fla': False}
    model = module.MiniMindForCausalLM(module.MiniMindConfig(**cpu_config))
    state = torch.load(WEIGHT, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    assert sum(p.numel() for p in model.parameters()) == 205623072
    tok = AutoTokenizer.from_pretrained(ROOT / 'upstream/model', local_files_only=True)
    return torch, model, tok


def destination(path):
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    # Student exercises can only create new learning artifacts.
    if not out.is_relative_to(ROOT / '学习手册' / '实操输出'):
        raise ValueError('Output must be a new folder under 学习手册/实操输出')
    out.mkdir(parents=True, exist_ok=False)
    return out


def trace(args):
    outdir = destination(args.out)
    before = sha(WEIGHT)
    torch, model, tok = load_cpu()
    # Index is always resolved at formal T768; then crop this one learning input.
    cfg, ds, _, x, y = sample(args.index, 768)
    x, y = x[:args.length].unsqueeze(0), y[:args.length].unsqueeze(0)
    if not (y[:, 1:] != -100).any():
        raise ValueError('Prefix has no assistant targets. Increase --length; do not train all-ignore labels.')
    model.train()  # Enables the actual aux loss. Formal config dropout is zero.
    shapes = {}
    embedding = []
    def hook(name):
        def record(_module, _args, result):
            tensor = result[0] if isinstance(result, tuple) else result
            if torch.is_tensor(tensor):
                shapes[name] = list(tensor.shape)
                if name == 'embedding':
                    tensor.retain_grad()
                    embedding.append(tensor)
        return record
    model.model.embed_tokens.register_forward_hook(hook('embedding'))
    for i, block in enumerate(model.model.layers):
        block.register_forward_hook(hook(f'block_{i+1}_{block.layer_type}'))
        block.mlp.gate.register_forward_hook(hook(f'router_{i+1}'))
    result = model(x, labels=y)
    result.logits.retain_grad()
    logits = result.logits[:, :-1, :].contiguous()
    labels = y[:, 1:].contiguous()
    manual = torch.nn.functional.cross_entropy(logits.reshape(-1,6400), labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(manual, result.loss, atol=1e-6, rtol=1e-6)
    # Isolate the CE path: aux also depends on prefix hidden states, so the
    # combined gradient alone cannot establish influence through context.
    ce_embedding_gradient = torch.autograd.grad(result.loss, embedding[0], retain_graph=True)[0].detach()
    embedding[0].grad = None
    result.logits.grad = None
    (result.loss + result.aux_loss).backward()
    parameter_gradients = {n: float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
    assert all(math.isfinite(v) for v in parameter_gradients.values())
    positions = []
    for j in range(x.shape[1]):
        entry = {'position':j, 'input_id':int(x[0,j]), 'label':int(y[0,j]),
                 'piece':tok.decode([int(x[0,j])]),
                 'embedding_output_gradient_norm':float(embedding[0].grad[0,j].norm()),
                 'embedding_output_ce_only_gradient_norm':float(ce_embedding_gradient[0,j].norm())}
        if j > 0 and y[0,j] != -100:
            v = result.logits[0,j-1].detach().float()
            target = int(y[0,j])
            entry.update(prediction_position=j-1, target_probability=float(v.softmax(-1)[target]),
                         token_ce=float(-v.log_softmax(-1)[target]), predicted_id=int(v.argmax()))
        positions.append(entry)
    after = sha(WEIGHT)
    assert before == after
    report = {'scope':'CPU full205M forward/backward, no optimizer/update; prefix is a learning probe, not formal training or quality evaluation',
              'device':'cpu', 'precision':'float32', 'model_parameters':205623072,
              'dataset_index_at_formal_T768':args.index, 'cache_row':int(ds.rows[args.index]),
              'data_source':ds.source, 'cpu_cuda_dependency_guard_disabled':True,
              'input_shape':list(x.shape), 'logits_shape':list(result.logits.shape),
              'valid_labels':int((labels != -100).sum()), 'ce':float(result.loss.detach()),
              'manual_ce':float(manual.detach()), 'aux_loss':float(result.aux_loss.detach()),
              'global_gradient_norm_before_clipping':math.sqrt(sum(v*v for v in parameter_gradients.values())),
              'shapes':shapes, 'gradient_norm_by_parameter':parameter_gradients,
              'positions':positions,
              'prefix_ce_only_nonzero_positions':sum(p['label']==-100 and p['embedding_output_ce_only_gradient_norm']>0 for p in positions),
              'weight_sha256_before':before, 'weight_sha256_after':after,
              'weight_unchanged':True, 'optimizer_steps':0,
              'script_sha256':sha(__file__), 'model_sha256':sha(ROOT/cfg['model_file'])}
    (outdir/'trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    emit({k:v for k,v in report.items() if k not in ['positions','gradient_norm_by_parameter','shapes']})


def infer(args):
    outdir = destination(args.out)
    before = sha(WEIGHT)
    torch, model, tok = load_cpu()
    model.eval()
    rendered = tok.apply_chat_template([{'role':'user','content':args.prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(rendered, return_tensors='pt', add_special_tokens=False).input_ids
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tok.eos_token_id)
    continuation = output[0,ids.shape[1]:].tolist()
    assert sha(WEIGHT) == before
    report = {'prompt':args.prompt,'rendered_input':rendered,'text':tok.decode(continuation,skip_special_tokens=False),
              'output_ids':continuation,'ended_with_eos':bool(continuation and continuation[-1]==tok.eos_token_id),
              'truncated':len(continuation)==args.max_new_tokens and continuation[-1]!=tok.eos_token_id,
              'device':'cpu','parameter_dtype':'float32','do_sample':False,'max_new_tokens':args.max_new_tokens,
              'checkpoint_sha256':before,'checkpoint_unchanged':True,'training_steps':0}
    (outdir/'generation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    emit(report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command', required=True)
    sub.add_parser('summary', help='Audit all recorded training updates and tokens')
    r=sub.add_parser('replay', help='Read one historical update and its preceding evaluation')
    r.add_argument('--step',type=int,required=True)
    s=sub.add_parser('sample', help='Decode a real training record and masks')
    s.add_argument('--index',type=int,default=51023)
    t=sub.add_parser('trace', help='CPU full-model forward/backward without any optimizer')
    t.add_argument('--index',type=int,default=51023)
    t.add_argument('--length',type=int,default=128,choices=[128,256,512,768])
    t.add_argument('--out',required=True)
    g=sub.add_parser('infer',help='CPU inference with the actual final checkpoint')
    g.add_argument('--prompt',required=True)
    g.add_argument('--max-new-tokens',type=int,default=64,choices=range(1,257))
    g.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='summary': read_summary()
    elif a.command=='replay': replay(a.step)
    elif a.command=='sample': show_sample(a.index)
    elif a.command=='trace': trace(a)
    elif a.command=='infer': infer(a)


if __name__=='__main__':
    main()
