#!/usr/bin/env python3
"""Bounded full-size MiniMind MoE training smoke test, not a quality experiment.

No dataset download, model architecture changes, external tracking, or compilation.
Synthetic token batches are fixed across hosts. The official model file is imported
unchanged. A separate, explicit FP32 model+AdamW resume check tests this harness.
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

import torch
import transformers


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(tensors):
    h = hashlib.sha256()
    for key, t in tensors:
        h.update(str(key).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def opt_digest(opt):
    return digest((f"{i}:{k}", value) for i, state in enumerate(opt.state.values())
                  for k, value in sorted(state.items()) if isinstance(value, torch.Tensor))


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(), device_free_bytes=free,
                device_total_bytes=total, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def step(model, opt, ids, labels, instrument=False):
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    phases = {"before_forward": memory()} if instrument else {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, labels=labels)
        loss = out.loss + out.aux_loss
    if instrument:
        phases["after_forward"] = memory()
    loss.backward()
    if instrument:
        phases["after_backward"] = memory()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    router_grad = model.model.layers[0].mlp.gate.weight.grad.norm()
    # Small before/after sample verifies the optimizer actually changes parameters.
    probe = model.model.embed_tokens.weight.view(-1)[:4096]
    before = probe.detach().clone()
    opt.step()
    delta = (probe.detach() - before).abs().max()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    values = dict(seconds=seconds, ce_loss=out.loss.item(), aux_loss=out.aux_loss.item(),
                  loss=loss.item(), grad_norm=grad_norm.item(), router_grad_norm=router_grad.item(),
                  parameter_sample_max_update=delta.item(),
                  logits_shape=list(out.logits.shape), logits_dtype=str(out.logits.dtype))
    if not all(torch.isfinite(torch.tensor(values[k])).item()
               for k in ("loss", "grad_norm", "router_grad_norm")):
        raise RuntimeError(f"Nonfinite result: {values}")
    if delta.item() == 0 or router_grad.item() == 0:
        raise RuntimeError(f"Missing optimizer/router update: {values}")
    if instrument:
        phases["after_optimizer"] = memory()
        values["memory_phases"] = phases
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--physical-gpu", default="0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--resume-check", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--expected-parameters",type=int,default=198416640)
    p.add_argument("--require-fla",action="store_true")
    p.add_argument("--linear-precision",choices=["float32","bfloat16"],default="float32")
    p.add_argument("--triton-precision",choices=["ieee","tf32","tf32x3"])
    p.add_argument("--allocator-gib",type=float)
    args = p.parse_args()
    if args.triton_precision: os.environ['TRITON_F32_DEFAULT']=args.triton_precision
    root = Path(__file__).resolve().parents[1]
    (root / "runs").mkdir(exist_ok=True)
    gpu_lock = (root / "runs/local_gpu.lock").open("a")
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "args": vars(args), "pid": os.getpid(),
              "hostname": platform.node(), "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_build": torch.version.cuda, "started_unix": time.time(),
              "model_sha256": hashlib.sha256(Path(args.model_file).read_bytes()).hexdigest(),
              "scope": "Synthetic fixed tokens; full configured architecture; tiny training verification, not learned quality"}
    write_json(outdir / "result.json", result)
    telemetry = open(outdir / "gpu.csv", "w")
    monitor = subprocess.Popen(["nvidia-smi", "-i", args.physical_gpu,
        "--query-gpu=timestamp,index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv", "-lms", "500"], stdout=telemetry, stderr=subprocess.STDOUT)
    try:
        torch.set_num_threads(4)
        if args.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261003)
        torch.cuda.manual_seed_all(20261003)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA/BF16 unavailable")
        if args.allocator_gib:
            torch.cuda.set_per_process_memory_fraction(args.allocator_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
        result["gpu"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        result["memory_before_model"] = memory()
        spec = importlib.util.spec_from_file_location("official_minimind", args.model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=True,require_fla=args.require_fla,
                                      linear_precision=args.linear_precision)
        model = module.MiniMindForCausalLM(config).cuda().train()
        count = sum(p.numel() for p in model.parameters())
        if count != args.expected_parameters:
            raise AssertionError(f"Unexpected model size: {count}")
        result["parameter_count"] = count
        result["parameter_bytes"] = tensor_bytes(model.parameters())
        result["parameter_dtype"] = str(next(model.parameters()).dtype)
        result["initial_parameter_probe_sha256"] = digest([(n, v.view(-1)[:64]) for n,v in model.named_parameters()])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        generator = torch.Generator().manual_seed(731)
        ids = torch.randint(3, config.vocab_size, (args.batch_size,args.seq_len), generator=generator).cuda()
        ids[:, 0] = 1
        ids[:, -1] = 2
        labels = ids.clone()
        result["input_sha256"] = digest([("input_ids",ids)])
        result["input_shape"] = list(ids.shape)
        result["labels_shape"] = list(labels.shape)
        result["supervised_tokens_per_step"] = args.batch_size * (args.seq_len-1)
        result["optimizer"] = {"name":"AdamW", "lr":1e-4, "betas":[0.9,0.999], "weight_decay":0.01, "foreach":False}
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for i in range(args.warmup + args.steps):
            row = step(model,opt,ids,labels,instrument=(i==0))
            row.update(step=i, warmup=i<args.warmup)
            rows.append(row)
            with open(outdir / "steps.jsonl", "a") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        result["memory_after_steps"] = memory()
        result["gradient_bytes"] = tensor_bytes(p.grad for p in model.parameters() if p.grad is not None)
        result["optimizer_tensor_bytes"] = tensor_bytes(v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
        times = [r["seconds"] for r in rows if not r["warmup"]]
        result["seconds_per_step_median"] = statistics.median(times)
        result["input_tokens_per_second"] = args.batch_size * args.seq_len * len(times) / sum(times)
        result["supervised_tokens_per_second"] = args.batch_size * (args.seq_len-1) * len(times) / sum(times)
        result["timed_steps"] = len(times)
        result["forward_backward_optimizer_checks"] = "passed"
        if args.resume_check:
            checkpoint = outdir / "smoke_resume.pt"
            opt.zero_grad(set_to_none=True)
            state = {"model":model.state_dict(),"optimizer":opt.state_dict(),
                     "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                     "completed_steps":len(rows),"config":config.to_dict(),"purpose":"hardware smoke only"}
            saved_model_hash = digest(model.state_dict().items())
            saved_opt_hash = opt_digest(opt)
            t = time.perf_counter()
            torch.save(state,checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
            save_seconds = time.perf_counter()-t
            del state
            expected = step(model,opt,ids,labels)
            expected_hash = digest(model.state_dict().items())
            expected_opt_hash = opt_digest(opt)
            del opt,model
            gc.collect()
            torch.cuda.empty_cache()
            loaded = torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
            model = module.MiniMindForCausalLM(config).cuda().train()
            model.load_state_dict(loaded["model"],strict=True)
            opt = torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
            opt.load_state_dict(loaded["optimizer"])
            torch.set_rng_state(loaded["cpu_rng"])
            torch.cuda.set_rng_state(loaded["cuda_rng"])
            del loaded
            gc.collect()
            restored_model_exact = saved_model_hash == digest(model.state_dict().items())
            restored_optimizer_exact = saved_opt_hash == opt_digest(opt)
            actual = step(model,opt,ids,labels)
            actual_hash = digest(model.state_dict().items())
            actual_opt_hash = opt_digest(opt)
            checks = {"restored_model_bitwise_equal":restored_model_exact,
                      "restored_optimizer_bitwise_equal":restored_optimizer_exact,
                      "next_loss_equal":expected["loss"]==actual["loss"],
                      "all_model_tensors_bitwise_equal":expected_hash==actual_hash,
                      "all_optimizer_tensors_bitwise_equal":expected_opt_hash==actual_opt_hash}
            result["resume"] = {"checks":checks,"passed":all(checks.values()),
                                "checkpoint_bytes":checkpoint.stat().st_size,
                                "save_seconds":save_seconds,"expected_next":expected,"resumed_next":actual,
                                "model_sha256":actual_hash,"optimizer_sha256":actual_opt_hash}
            if not all(checks.values()):
                raise AssertionError(f"Resume comparison failed: {checks}")
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"],flush=True)
        if torch.cuda.is_initialized():
            result["memory_at_failure"] = memory()
        raise
    finally:
        result["finished_unix"] = time.time()
        write_json(outdir / "result.json",result)
        monitor.terminate()
        try: monitor.wait(timeout=3)
        except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
        telemetry.close()
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
