#!/usr/bin/env python3
"""Refresh a small navigation report from actual run states; never infer success."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.live_status import read_live_json


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(value);tmp.replace(path)


def snapshot():
    priority=json.loads((ROOT/'plans/formal_hybrid_official_mini_2ep_v1.json').read_text())
    jobs=[(s['id'],s['completion']['run_dir'],s['completion']['expected_steps'])
          for s in priority['stages'] if s['kind']=='train']
    jobs += [('198M预训练','models/01_moe/runs/pretrain_full_v1',58752),
             ('198M SFT','models/01_moe/runs/sft_full_v1',318877)]
    for name in ['formal_hybrid_official_priority_v1','formal_hybrid_and_control_v1','formal_omni_server_v1']:
        plan=json.loads((ROOT/f'plans/{name}.json').read_text())
        jobs.extend((s['id'],s['completion']['run_dir'],s['completion']['expected_steps']) for s in plan['stages'] if s['kind']=='train')
    records=[]
    for name,path,total in jobs:
        status=read_live_json(ROOT/path/'status.json',missing_ok=True) or {}
        value=status.get('status','not_started')
        # Remote states are mirrors and may lag the server; local stale training
        # state is likewise evidence to check, not proof a process is alive.
        records.append(dict(name=name,path=path,status=value,step=status.get('step',0),total_steps=total,
                            progress=status.get('step',0)/total,
                            effective_tokens=status.get('tokens_trained',status.get('trained_tokens')),
                            last_record_unix=status.get('time'),validation=status.get('validation')))
    controls={}
    for name in ['formal_hybrid_official_mini_2ep_v1','formal_hybrid_official_priority_v1','formal_hybrid_and_control_v1','formal_omni_server_v1','omni_formal_deployment_v3','omni_formal_arming_v2','omni_delivery_bridge_v1']:
        controls[name]=read_live_json(ROOT/f'runs/{name}/status.json',missing_ok=True)
    now=time.time();report=dict(time=now,generated_local=datetime.now().astimezone().isoformat(),stages=records,controllers=controls,
                               scope='Budget state only. No automatic quality approval. Remote values are last mirrored snapshots.')
    atomic(ROOT/'reports/project_status.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    active = [row['name'] for row in records if row['status'] == 'training']
    phase = ('Hybrid官方mini两轮训练预算已完成，生成质量仍须独立审查。'
             if records[0]['status'] == 'complete'
             else '当前优先完成205M Hybrid官方mini完整两轮、113,082次更新。')
    phase += (' 当前报告为training的阶段：' + '、'.join(active) + '。') if active else ' 当前没有阶段报告为training。'
    phase += ' 后续顺序与recipe见[整体计划](../docs/experiment_plan.md)。'
    lines=['# 项目实时记录','',f'更新时间：{report["generated_local"]}。本页每60秒读取实际状态；远端数据可能因网络滞后。',
           '', phase,
           '', '**训练预算完成≠生成质量合格≠整个学习项目完成。** `not_started`表示正式阶段尚未开始，预检不计入。', '',
           '| 阶段 | 状态 | optimizer steps | 本阶段预算进度 |', '|---|---|---:|---:|']
    for row in records:
        lines.append(f'| [{row["name"]}](../{row["path"]}/) | {row["status"]} | {row["step"]:,} / {row["total_steps"]:,} | {100*row["progress"]:.2f}% |')
    lines+=['','## 接续与交付','']
    for name,status in controls.items():
        lines.append(f'- [{name}](../runs/{name}/)：{status.get("event",status.get("status")) if status else "尚无控制器状态"}')
    lines+=['','正式预算和时间解释：[执行方案](../docs/formal_execution_v1.md)。',
            '学习路径、模型目录、数据和debug入口：[根README](../README.md)。',
            '本页不能检测所有挂起/网络故障；请同时核对日志时间、controller错误与实际进程。','']
    atomic(ROOT/'reports/project_status.md','\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    lock=(ROOT/'runs/project_status.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(json.dumps(dict(pid=os.getpid(),watch=a.watch)),flush=True)
    while True:
        snapshot()
        if not a.watch:return
        time.sleep(60)


if __name__=='__main__':main()
