#!/bin/bash
# Put the inputs generation_eval/ reads into generation_eval/data/ and external/.
# Needs dataset/raw/ first (splits/build.sh, or splits/download_hf.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RAW=../dataset/raw
for f in neutral_articles_10k.jsonl left_neutral_dedup.jsonl right_neutral_dedup.jsonl; do
    [ -f "$RAW/$f" ] || { echo "missing $RAW/$f: run splits/download_hf.py first"; exit 1; }
done
mkdir -p data/led_eval models external evaluation/scores jev/results   # last two: notebook outputs
ln -sf ../../dataset/raw/neutral_articles_10k.jsonl data/stories_10k.jsonl          # same file, older name
ln -sf ../../../dataset/raw/left_neutral_dedup.jsonl data/led_eval/left_neutral_dedup.jsonl
ln -sf ../../../dataset/raw/right_neutral_dedup.jsonl data/led_eval/right_neutral_dedup.jsonl
# 2,000 held-out test articles with the event's centre article as target (QA eval input).
git show 3de161f:dataset/llm_neutral_prepared/test/baseline.jsonl > data/own_test.jsonl
# DiverseSumm (QA coverage eval), pinned to the commit the evaluation used.
if [ ! -d external/DiverseSumm ]; then
    git clone -q https://github.com/salesforce/DiverseSumm external/DiverseSumm
    git -C external/DiverseSumm checkout -q 42f22e3
fi
echo "data ready: $(ls data data/led_eval | tr '\n' ' ')"
