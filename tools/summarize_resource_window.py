#!/usr/bin/env python3
"""Summarize completed bounded runs; keep failures distinct from completed budgets."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def jsonlines(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--window',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    window=ROOT/a.window;out=ROOT/a.out
    if out.exists():raise RuntimeError('Preserve earlier resource report')
    events=jsonlines(window/'events.jsonl')
    if not events or events[-1]['event']!='window_finished':raise RuntimeError('Window not yet finished')
    ended={e['name']:e for e in events if e['event']=='check_end'};reports=[]
    for task in json.loads((window/'checks.json').read_text()):
        row=dict(name=task['name'],execution=ended.get(task['name'],{'not_executed':True}));args=task['args']
        if '--config' not in args:
            row['kind']='independent numerical/resume check';reports.append(row);continue
        cfg=json.loads((ROOT/args[args.index('--config')+1]).read_text());run=ROOT/cfg['run_dir']
        row.update(config=cfg,kind='bounded resource preflight')
        for name in ['status','provenance','failure']:
            if (run/f'{name}.json').exists():row[name]=json.loads((run/f'{name}.json').read_text())
        metrics=jsonlines(run/'metrics.jsonl');train=[r for r in metrics if r.get('event')=='train'];steady=[r for r in train if r['step']>2]
        if train:
            row['observed_updates']=len(train);row['peak_allocated_bytes']=max(r.get('cuda_peak_allocated',0) for r in train)
        if steady:
            seconds=sum(r['step_seconds'] for r in steady)
            key='valid_tokens' if 'valid_tokens' in steady[0] else 'text_labels'
            row['steady_after_first_two_updates']=dict(updates=len(steady),seconds=seconds,
                mean_step_seconds=seconds/len(steady),median_step_seconds=statistics.median(r['step_seconds'] for r in steady),
                effective_samples_per_update=cfg['batch_size']*cfg['accumulation_steps'],
                nominal_samples_per_second=len(steady)*cfg['batch_size']*cfg['accumulation_steps']/seconds,
                label_counter_key=key)
            if key=='valid_tokens':row['steady_after_first_two_updates']['valid_tokens_per_second']=sum(r[key] for r in steady)/seconds
            else:
                row['steady_after_first_two_updates']['mean_recorded_samples_per_second']=statistics.mean(r['samples_per_second'] for r in steady)
                row['steady_after_first_two_updates']['text_labels_per_second']=sum(r['text_labels'] for r in steady)/seconds
                row['steady_after_first_two_updates']['audio_labels_per_second_all_eight_heads']=sum(r['audio_labels'] for r in steady)/seconds
                row['data_counters']={key:sum(r.get('data',{}).get(key,0) for r in train) for key in ['samples','zero_supervision_samples','audio_samples_with_missing_stop','active_audio_heads','audio_heads_missing_stop']}
        hosts=jsonlines(run/'host.jsonl')
        if hosts:
            row['host']=dict(samples=len(hosts),minimum_available_bytes=min(r['host_available'] for r in hosts),
                maximum_swap_used_bytes=max(r['host_swap_used'] for r in hosts),maximum_parent_rss_bytes=max(r['parent_rss'] for r in hosts),
                global_swap_out_increment_bytes=hosts[-1]['host_swap_out_bytes']-hosts[0]['host_swap_out_bytes'],
                global_swap_in_increment_bytes=hosts[-1]['host_swap_in_bytes']-hosts[0]['host_swap_in_bytes'],
                caveat='Whole WSL host counters sampled every two seconds; parent RSS includes shared/file-backed pages; global swapping not exclusively attributable to this process')
        reports.append(row)
    report=dict(window=a.window,checks=reports,window_finished_event=events[-1],
        formal_resume=[e for e in events if e['event'].startswith('formal_resume')],
        scope='Short full-architecture checks from early AR transfer, not formal trained quality; source/config/data differences retained. First two updates excluded from steady timing. Timings exclude validation, saving and startup; failures remain failures.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],execution=r['execution'],steady=r.get('steady_after_first_two_updates'),peak=r.get('peak_allocated_bytes'),host=r.get('host')) for r in reports],indent=2))

if __name__=='__main__':main()
