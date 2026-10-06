#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/minimind
source env/activate.sh
while [ ! -f data/raw/minimind/pretrain_t2t.jsonl.verified.json ]; do sleep 10; done
python -u -B tools/download_ranges.py sft_t2t.jsonl --workers 4 > logs/data_download_sft_ranges.log 2>&1
python -u -B tools/prepare_tokens.py --raw data/raw/minimind/sft_t2t.jsonl --out data/processed/sft_t2t_full_v1 --stage sft > logs/prepare_sft_full.log 2>&1
