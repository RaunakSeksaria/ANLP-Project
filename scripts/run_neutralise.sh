#!/bin/bash
# Single-article neutralisation for ONE stance.
#
#   scripts/run_neutralise.sh left  NEMOTRON_3_ULTRA_KEY
#   scripts/run_neutralise.sh right SOME_OTHER_KEY
#
# The two stances are fully independent: separate input file, separate output
# file, separate API key. Run them as two processes, on two machines if you
# like. Either can die without affecting the other.
#
# Resumable -- re-running skips story_ids already written with text.
# Removable -- rm runs/nemotron/neutralised_<stance>.jsonl discards that stance.
set -u
STANCE="${1:?usage: run_neutralise.sh <left|right> [API_KEY_ENV_VAR]}"
KEYVAR="${2:-NEMOTRON_3_ULTRA_KEY}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# data/ is gitignored, so a fresh clone has the inputs at the repo root.
IN="data/final/single_${STANCE}_10k.jsonl"
[ -f "$IN" ] || IN="single_${STANCE}_10k.jsonl"
mkdir -p runs/nemotron

exec .venv/bin/python3 scripts/generate_neutral.py \
  --task neutralise \
  --provider nvidia --api-key-env "$KEYVAR" \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --in  "$IN" \
  --out "runs/nemotron/neutralised_${STANCE}.jsonl" \
  --workers 14 --rpm 40 --thinking-level HIGH \
  --temperature 1.0 --top-p 0.95 --max-output-tokens 16384 \
  --stop-after 5 \
  --set max_network_retries=3 \
  --set enforce_length=false \
  --set hard_max_ratio=3.0
