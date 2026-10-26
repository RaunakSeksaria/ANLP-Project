#!/bin/bash
# Generate LED neutraliser outputs on the 2,000 held-out stories_10k test articles, then score them with Jev.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 led_eval/generate.py --set test --tokens-per-batch 4096 2>&1 | grep --line-buffered -vE "padded from|Warning"
echo "LED GENERATION DONE"
# Wait for the Nemotron Jev run so the two never hit the spend ledger at the same time.
while pgrep -f run_neutral.sh >/dev/null; do sleep 30; done
cd jev && python3 score_neutral.py led --cap-usd 2.60
echo "LED JEV DONE"
