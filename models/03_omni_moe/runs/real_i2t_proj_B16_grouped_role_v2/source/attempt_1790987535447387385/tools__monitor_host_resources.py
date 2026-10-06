#!/usr/bin/env python3
"""Low-overhead WSL host memory/paging observations for one training lifetime."""
import argparse,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pid',type=int,required=True); p.add_argument('--interval',type=float,default=2.)
    args=p.parse_args(); parent=Path(f'/proc/{args.pid}'); page_bytes=os.sysconf('SC_PAGE_SIZE')
    while parent.exists():
        try:
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
            status={line.split(':')[0]:line.split(':',1)[1].strip() for line in (parent/'status').read_text().splitlines()}
            vm={key:int(value) for key,value in (line.split() for line in Path('/proc/vmstat').read_text().splitlines())}
            if status['State'].startswith('Z'): break
            row=dict(time=time.time(),monotonic=time.monotonic(),pid=args.pid,host_total=mem['MemTotal'],host_available=mem['MemAvailable'],
                     host_swap_used=mem['SwapTotal']-mem['SwapFree'],host_shmem=mem['Shmem'],parent_rss=int(status['VmRSS'].split()[0])*1024,
                     parent_locked=int(status.get('VmLck','0 kB').split()[0])*1024,host_major_faults=vm['pgmajfault'],
                     host_swap_in_bytes=vm['pswpin']*page_bytes,host_swap_out_bytes=vm['pswpout']*page_bytes)
            print(json.dumps(row),flush=True)
        except FileNotFoundError: break
        time.sleep(args.interval)

if __name__=='__main__': main()
