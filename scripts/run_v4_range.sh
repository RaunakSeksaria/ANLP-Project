#!/bin/bash
# v4 range-style neutral articles. Writes to its OWN file -- the v3 corpus at
# runs/nemotron/summaries_10k.jsonl is never touched. Resumable: skips story_ids
# already written with a neutral article, so it is safe to re-run at any time.
# To discard this variant entirely: rm runs/nemotron/neutral_v4_range.jsonl
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec .venv/bin/python3 scripts/generate_neutral.py \
  --provider nvidia --api-key-env NEMOTRON_3_ULTRA_KEY \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --in data/final/clusters_corpus10k.jsonl \
  --out runs/nemotron/neutral_v4_range.jsonl \
  --conflict-style range \
  --workers 14 --rpm 40 --thinking-level HIGH \
  --temperature 1.0 --top-p 0.95 --max-output-tokens 16384 \
  --stop-after 5 \
  --set max_network_retries=3 \
  --set enforce_length=false \
  --set hard_max_ratio=5.0
