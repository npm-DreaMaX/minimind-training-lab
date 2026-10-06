#!/usr/bin/env python3
"""Exercise recovery and fail-closed behavior without touching training files."""
import argparse,json,sys,tempfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'reader_source.py').write_bytes((ROOT/'lab/live_status.py').read_bytes())
    events=[];results={}
    with tempfile.TemporaryDirectory(prefix='reader-',dir=out) as directory:
        path=Path(directory)/'status.json'
        assert read_live_json(path,missing_ok=True) is None
        results['expected_not_started']=True
        for case in ['missing','invalid']:
            if case=='invalid':path.write_text('{')
            def publish():
                time.sleep(.08);path.with_suffix('.tmp').write_text('{"step":42}')
                path.with_suffix('.tmp').replace(path)
            writer=threading.Thread(target=publish);writer.start()
            value=read_live_json(path,timeout=2,interval=.01,report=events.append)
            writer.join();assert value=={'step':42};results[case+'_recovered']=True
            path.unlink()
        started=time.monotonic()
        try:read_live_json(path,timeout=.1,interval=.01,report=events.append)
        except RuntimeError:results['persistent_missing_fails']=True
        else:raise AssertionError('A permanently missing state was hidden')
        results['timeout_seconds']=time.monotonic()-started
        path.write_text('{"step":-1}');done=threading.Event();failures=[]
        def racing_writer():
            try:
                for step in range(2000):
                    temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'step':step}));temp.replace(path)
            except Exception as exc:failures.append(repr(exc))
            finally:done.set()
        writer=threading.Thread(target=racing_writer);writer.start();reads=0
        while not done.is_set():
            value=read_live_json(path,timeout=2,interval=.001,report=events.append)
            assert isinstance(value['step'],int);reads+=1
        writer.join();assert not failures,failures
        assert read_live_json(path)['step']==1999
        results.update(concurrent_writes=2000,successful_reads=reads,concurrent_read_pass=True)
    report=dict(complete=True,results=results,retry_events=events)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
