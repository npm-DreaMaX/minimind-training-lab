"""Observe an idle GPU; this is neither a reservation nor permission to evict work."""
import subprocess,time


def is_idle(snapshot):
    return (snapshot['memory_used_mib']<=512 and snapshot['memory_free_mib']>=15000
            and snapshot['utilization_percent']==0 and not snapshot['compute_pids'])


def read_snapshot(index):
    fields=[v.strip() for v in subprocess.check_output(['nvidia-smi',f'--id={index}',
            '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    index_value,uuid,used,free,util=fields
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    pids=[]
    for line in processes.splitlines():
        values=[v.strip() for v in line.split(',')]
        if len(values)==2 and values[0]==uuid:pids.append(int(values[1]))
    return dict(index=int(index_value),uuid=uuid,memory_used_mib=int(used),memory_free_mib=int(free),utilization_percent=int(util),compute_pids=pids,time=time.time())


def wait_for_idle(index,emit,timeout=20):
    deadline=time.monotonic()+timeout;consecutive=0
    while True:
        snapshot=read_snapshot(index);emit(snapshot)
        consecutive=consecutive+1 if is_idle(snapshot) else 0
        if consecutive>=2:return snapshot
        if time.monotonic()>=deadline:raise RuntimeError(f'GPU did not become consistently idle; no process terminated: {snapshot}')
        time.sleep(1)
