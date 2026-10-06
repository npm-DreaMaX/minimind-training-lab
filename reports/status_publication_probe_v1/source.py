#!/usr/bin/env python3
"""Reproduce live atomic-replace reads on project mount and Linux /tmp; no GPU."""
import argparse,json,threading,time,tempfile
from pathlib import Path

def trial(root,iterations):
    path=root/'status.json';temp=root/'status.json.tmp';path.write_text(json.dumps({'step':-1}))
    stop=threading.Event();counts=dict(reads=0,missing=0,invalid=0,writer_failures=0);examples=[];started=time.perf_counter()
    def writer():
        try:
            for i in range(iterations):
                temp.write_text(json.dumps({'step':i,'payload':'x'*256}));temp.replace(path)
        except Exception as exc:
            counts['writer_failures']+=1;examples.append(dict(writer_error=repr(exc)))
        finally:stop.set()
    thread=threading.Thread(target=writer);thread.start()
    while not stop.is_set():
        try:json.loads(path.read_text())
        except FileNotFoundError:
            counts['missing']+=1
            if len(examples)<20:examples.append(dict(error='FileNotFoundError',seconds=time.perf_counter()-started))
        except json.JSONDecodeError:
            counts['invalid']+=1
            if len(examples)<20:examples.append(dict(error='JSONDecodeError',seconds=time.perf_counter()-started))
        counts['reads']+=1
    thread.join();return dict(**counts,iterations=iterations,seconds=time.perf_counter()-started,examples=examples)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,default=2000);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise RuntimeError('Preserve existing probe')
    out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes());mounted=out/'mounted';mounted.mkdir()
    report={'mounted':trial(mounted,a.iterations)}
    with tempfile.TemporaryDirectory(prefix='minimind-status-',dir='/tmp') as directory:report['linux_tmp']=trial(Path(directory),a.iterations)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
