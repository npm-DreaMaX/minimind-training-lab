import os,fcntl
from pathlib import Path
ROOT=Path('/mnt/d/minimind/reports/gpu_lock_pid_regression_v1/lock-kh0wbyc6')
gpu_lock=(ROOT/'runs/local_gpu.lock').open('a+')
fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu_lock.seek(0)
gpu_lock.truncate()
gpu_lock.write(str(os.getpid()))
gpu_lock.flush()
