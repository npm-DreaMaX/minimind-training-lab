#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/minimind
source env/activate.sh
printf 'Waiting for verified full pretraining corpus\n'
while [ ! -f data/raw/minimind/pretrain_t2t.jsonl.verified.json ]; do sleep 10; done
printf 'Verified corpus ready; preparing tokens\n'
python -u -B tools/prepare_tokens.py --raw data/raw/minimind/pretrain_t2t.jsonl --out data/processed/pretrain_t2t_full_v1 --stage pretrain > logs/prepare_pretrain_full.log 2>&1
python -B - <<'PY'
import json
m=json.load(open('data/processed/pretrain_t2t_full_v1/metadata.json'))
assert m['limit']==0 and m['bad_rows']==0 and m['train_rows']>100000 and m['val_rows']>100, m
print('Full corpus verified:',m['rows'],'rows',m['full_input_tokens'],'input tokens')
PY
printf 'Starting formal full-corpus MoE pretraining\n'
python -u -B -m lab.train --config models/01_moe/configs/pretrain_full_v1.json > logs/pretrain_full_v1.log 2>&1
python -B tools/plot_run.py models/01_moe/runs/pretrain_full_v1
