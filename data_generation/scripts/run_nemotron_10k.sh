#!/bin/bash
# Resumable: skips story_ids already written with a summary. Safe to re-run.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --no-sync python scripts/generate_neutral.py \
  --provider nvidia --api-key-env NEMOTRON_3_ULTRA_KEY \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --in data/final/clusters_corpus10k.jsonl \
  --out runs/nemotron/summaries_10k.jsonl \
  --workers 14 --rpm 40 --thinking-level HIGH \
  --temperature 1.0 --top-p 0.95 --max-output-tokens 16384 \
  --stop-after 5 \
  --set max_network_retries=3 \
  --set enforce_length=false \
  --set hard_max_ratio=5.0
