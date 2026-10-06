#!/usr/bin/env python3
"""Save a live AR run, execute bounded exclusive-GPU checks, then resume it.

SIGTERM/SIGINT to this controller cancels the window without restarting training.
An individual check failing is retained and does not abandon the formal run.
"""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json
CANCEL=False
CHILD=None

def cancel(*_):
    global CANCEL
    CANCEL=True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)

def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'

def main():
    global CHILD
    p=argparse.ArgumentParser(); p.add_argument('--pid',type=int,required=True); p.add_argument('--out',required=True)
    p.add_argument('--checks',help='Optional JSON list of named bounded commands and dependencies')
    a=p.parse_args(); out=ROOT/a.out
    if out.exists(): raise RuntimeError('Preserve previous window evidence')
    out.mkdir(parents=True); shutil.copy2(__file__,out/'source_controller.py')
    tasks=json.loads((ROOT/a.checks).read_text()) if a.checks else None
    if tasks is not None: (out/'checks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    events=(out/'events.jsonl').open('a',buffering=1)
    def log(event,**fields):
        row=dict(event=event,time=time.time(),**fields); events.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
    signal.signal(signal.SIGTERM,cancel); signal.signal(signal.SIGINT,cancel)
    cfg='models/01_moe/configs/pretrain_full_v1.json'; run=ROOT/'models/01_moe/runs/pretrain_full_v1'
    proc=Path(f'/proc/{a.pid}')
    command=proc.joinpath('cmdline').read_bytes().split(b'\0')
    if b'lab.train' not in command or cfg.encode() not in command or proc.joinpath('cwd').resolve()!=ROOT:
        raise RuntimeError('PID does not identify the intended formal training process')
    old_status=read_live_json(run/'status.json'); assert old_status['status']=='training',old_status
    normal_env=os.environ.copy(); normal_env.pop('PYTORCH_ALLOC_CONF',None)
    test_env={**normal_env,'PYTORCH_ALLOC_CONF':'expandable_segments:True'}
    main_python='/home/USER/.venvs/minimind-lab/bin/python'
    omni_python='/home/USER/.venvs/minimind-omni/bin/python'
    interrupted=False; completed=[]
    def check(name,argv,timeout=600,omni=False):
        global CHILD
        if CANCEL: return False
        cmd=[omni_python if omni else main_python,'-u','-B',*argv]
        log('check_start',name=name,command=cmd,timeout_seconds=timeout)
        with (out/f'{name}.log').open('x') as stream:
            CHILD=subprocess.Popen(cmd,cwd=ROOT,env=test_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic(); timed_out=False
            while CHILD.poll() is None:
                if CANCEL or time.monotonic()-started>timeout:
                    timed_out=not CANCEL
                    os.killpg(CHILD.pid,signal.SIGTERM)
                    deadline=time.monotonic()+30
                    while CHILD.poll() is None and time.monotonic()<deadline: time.sleep(1)
                    # Kill the isolated process group, including workers and telemetry.
                    try: os.killpg(CHILD.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    CHILD.wait(); break
                time.sleep(1)
            record=dict(name=name,returncode=CHILD.returncode,timed_out=timed_out,seconds=time.monotonic()-started)
            completed.append(record); log('check_end',**record); CHILD=None
        return record['returncode']==0 and not timed_out and not CANCEL
    try:
        log('save_requested',pid=a.pid,previous_status=old_status,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        os.kill(a.pid,signal.SIGTERM); interrupted=True
        deadline=time.monotonic()+180
        while alive(a.pid):
            if CANCEL: return
            if time.monotonic()>deadline: raise TimeoutError('Training did not save and exit; it was not forcibly killed')
            time.sleep(1)
        status=read_live_json(run/'status.json'); assert status['status']=='paused',status
        with (ROOT/'runs/local_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(lock,fcntl.LOCK_UN)
        log('saved_and_released',status=status)
        if tasks is not None:
            passed={}
            for task in tasks:
                if any(not passed.get(key,False) for key in task.get('requires',[])):
                    log('check_skipped',name=task['name'],reason='Required earlier check did not pass'); passed[task['name']]=False
                    continue
                passed[task['name']]=check(task['name'],task['args'],timeout=task.get('timeout',600),omni=task.get('omni',False))
            return
        weight=str((run/'checkpoints'/f"model_step_{status['step']:07d}.pth").relative_to(ROOT))
        check('ar_profile',['tools/profile_training_step.py','--config',cfg,'--weight',weight,'--out','models/01_moe/runs/profile_real_update_v1'])
        precision_ok=check('hybrid_precision',['tools/validate_hybrid_precision_gpu.py','--out','models/02_hybrid_moe/runs/precision_gpu_v1'])
        for suffix in ['', '_bf16_projection'] if precision_ok else ['']:
            for phase in ['real_pretrain_B2_T512','real_sft_B1_T1536']:
                check(phase+suffix,['-m','lab.train','--config',f'models/02_hybrid_moe/configs/{phase}{suffix}.json'])
        if precision_ok:
            check('hybrid_bf16_resume',['tools/hardware_smoke.py','--model-file','models/02_hybrid_moe/src/model_hybrid.py',
                  '--out','models/02_hybrid_moe/runs/resume_bf16_projection_v1','--expected-parameters','205623072',
                  '--require-fla','--linear-precision','bfloat16','--batch-size','1','--seq-len','128',
                  '--warmup','1','--steps','2','--resume-check','--deterministic','--allocator-gib','5.3'])
        check('omni_real_t2a',['-m','lab.omni_train','--config','models/03_omni_moe/configs/real_t2a_B1_T1536.json'],omni=True)
        check('omni_full_resume',['tools/validate_omni_trainer_cpu.py','--full-gpu','--out','models/03_omni_moe/runs/trainer_gpu_resume_v1'],timeout=900,omni=True)
    except Exception as exc:
        log('window_error',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if interrupted and not CANCEL and not alive(a.pid):
            status=read_live_json(run/'status.json')
            if status['status']=='paused':
                # Fail closed if any unrelated GPU job claimed the project slot.
                with (ROOT/'runs/local_gpu.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(lock,fcntl.LOCK_UN)
                cmd=[main_python,'-u','-B','-m','lab.train','--config',cfg,'--resume']
                with (ROOT/'logs/pretrain_full_v1.log').open('a') as stream:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=normal_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                log('formal_resume_started',pid=child.pid,command=cmd,checkpoint_step=status['step'])
                (out/'resumed_process.json').write_text(json.dumps(dict(pid=child.pid,command=cmd,time=time.time()),indent=2)+'\n')
            else: log('formal_resume_withheld',status=status)
        log('window_finished',cancelled=CANCEL,checks=completed)
        events.close()

if __name__=='__main__': main()
