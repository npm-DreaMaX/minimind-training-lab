#!/usr/bin/env python3
"""Launch only the verified, authorized remote branch; keep selected-GPU telemetry."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import assert_pinned,sha256


def main():
    assert str(ROOT)=='/new_data/REMOTE_USER/minimind/omni_formal_v1/project'
    assert sys.prefix=='/new_data/REMOTE_USER/minimind/envs/omni-formal'
    for name in ['reports/omni_formal_deployment_v2/remote_verified.json','runs/formal_runtime_preflight_v1/result.json']:
        if not json.loads((ROOT/name).read_text())['passed']:raise RuntimeError(f'Prerequisite failed: {name}')
    plan=json.loads((ROOT/'plans/formal_omni_server_v1.json').read_text());assert_pinned(ROOT,plan['fingerprints'])
    probe=json.loads((ROOT/'runs/formal_runtime_preflight_v1/t2a_B16_acc8/provenance.json').read_text())
    for name,digest in probe['fingerprints'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError(f'Runtime preflight source drift: {name}')
    out=ROOT/'runs/omni_formal_supervisor_v1';out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),prefix=sys.prefix))+'\n')
    with (out/'gpu_selected.csv').open('x') as gpu, (out/'pipeline.log').open('x') as log:
        monitor=subprocess.Popen(['nvidia-smi','--id=2','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv','-l','10'],stdout=gpu,stderr=subprocess.STDOUT)
        try:
            result=subprocess.run([sys.executable,'-u','-B','tools/run_formal_pipeline.py','--plan','plans/formal_omni_server_v1.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    (out/'exit.json').write_text(json.dumps(dict(returncode=result.returncode))+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
