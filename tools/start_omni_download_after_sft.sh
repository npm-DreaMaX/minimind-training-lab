#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/minimind
source env/activate.sh
while [ ! -f data/raw/minimind/sft_t2t.jsonl.verified.json ]; do sleep 10; done
for file in sft_t2a.parquet sft_a2a.parquet sft_i2t.parquet; do
  if [ ! -f "data/raw/omni/$file.verified.json" ]; then
    python -u -B tools/download_ranges.py "$file" --manifest data/manifests/minimind_omni_full.json --out-dir data/raw/omni --workers 4 > "logs/data_download_$file.log" 2>&1
  fi
done
