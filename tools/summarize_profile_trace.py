#!/usr/bin/env python3
"""Bounded-memory trace summary: keep kernel events separate from CPU ranges."""
import argparse,collections,json
from pathlib import Path

def events(path):
    decoder=json.JSONDecoder()
    with path.open() as stream:
        buffer=''
        while '"traceEvents": [' not in buffer:
            chunk=stream.read(65536)
            if not chunk: raise ValueError('No traceEvents array')
            buffer+=chunk
        buffer=buffer.split('"traceEvents": [',1)[1]
        while True:
            buffer=buffer.lstrip(' \r\n\t,')
            if buffer.startswith(']'): return
            try: event,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk=stream.read(65536)
                if not chunk: raise ValueError('Incomplete trace event')
                buffer+=chunk; continue
            yield event; buffer=buffer[end:]

def main():
    p=argparse.ArgumentParser(); p.add_argument('trace'); a=p.parse_args(); path=Path(a.trace)
    totals=collections.Counter(); counts=collections.Counter(); kernels=collections.Counter(); ranges=collections.defaultdict(list)
    for event in events(path):
        if event.get('ph')!='X': continue
        cat=event.get('cat',''); duration=event.get('dur',0)
        totals[cat]+=duration; counts[cat]+=1
        if cat=='kernel': kernels[event['name']]+=duration
        if cat=='user_annotation' and event['name'] in ['microbatch_forward','microbatch_backward','clip_and_optimizer','data_to_cuda']:
            ranges[event['name']].append(duration)
    report=dict(category_duration_us=dict(totals),category_count=dict(counts),
                cpu_ranges={key:dict(count=len(values),sum_us=sum(values)) for key,values in ranges.items()},
                top_kernels=[dict(name=name,duration_us=value,kernel_duration_fraction=value/totals['kernel']) for name,value in kernels.most_common(25)],
                caveats='Category durations are sums, not a partition of wall time. CPU operations nest, CUDA annotations overlap kernels, and concurrent kernels may overlap each other. Memory-allocation totals in the operator table are not peak live VRAM. Normal throughput is measured outside the profiler.')
    out=path.with_name('trace_summary.json'); out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='top_kernels'},indent=2))

if __name__=='__main__': main()
