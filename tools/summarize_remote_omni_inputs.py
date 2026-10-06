#!/usr/bin/env python3
"""Combine completed remote paths while preserving the earlier controller failure."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out
    if out.exists():raise RuntimeError('Keep earlier report')
    base=ROOT/'models/03_omni_moe/runs/server_readiness_inputs_v1';report=dict(scope='Bounded full315M hardware/path checks, not formally trained quality or population throughput',groups={},checks=[])
    for group in ['real_inputs_all_modes_v1','real_inputs_visual_v2']:
        source=json.loads((base/group/'result.json').read_text());report['groups'][group]=dict(controller_passed=source['passed'],started=source['started'],finished=source['finished'])
        for check in source['checks']:
            run=base/group/check['name'];cfg=json.loads((run/'config.json').read_text())
            metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            train=[m for m in metrics if m.get('event')=='train'];steady=train[2:]
            hosts=[json.loads(line) for line in (run/'host.jsonl').read_text().splitlines()]
            row=dict(name=check['name'],result=check['status'],run=str(run.relative_to(ROOT)),config=cfg,
                     parameters=check['provenance']['parameters'],trainable_parameters=check['provenance']['trainable_parameters'],
                     updates=len(train),steady_updates=len(steady),samples_per_second=sum(m['data']['samples'] for m in steady)/sum(m['step_seconds'] for m in steady),
                     peak_allocated_gib=max(m['cuda_peak_allocated'] for m in train)/1024**3,
                     minimum_host_available_gib=min(m['host_available'] for m in hosts)/1024**3,
                     nonzero_projector_gradient=check.get('nonzero_projector_gradient'),projector_gradients=check.get('projector_gradients'),
                     data={key:sum(m['data'][key] for m in train) for key in ['samples','zero_supervision_samples','audio_samples_with_missing_stop']},
                     maximum_observed_fbank_frames=max(m['data']['max_fbank_frames'] for m in train),
                     complete_status=json.loads((run/'status.json').read_text()))
            report['checks'].append(row)
    expected={'t2a_B16_recompute','a2a_all_long','a2a_audio_proj_long','i2t_all','i2t_vision_proj'}
    assert {r['name'] for r in report['checks']}==expected
    report['all_five_paths_passed']=all(r['result']=='passed' and r['parameters']==314887938 for r in report['checks'])
    assert report['all_five_paths_passed']
    report['controller_caveat']='v1 stopped at the pre-I2T idle-card guard; its three completed model checks remain passed. v2 ran only the two pending I2T modes; neither group alone represents all five paths.'
    report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],samples_per_second=r['samples_per_second'],peak_gib=r['peak_allocated_gib'],data=r['data']) for r in report['checks']],indent=2))


if __name__=='__main__':main()
