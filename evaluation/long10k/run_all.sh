#!/bin/bash
# Score every model over all long_10k texts, one after another. Resumable: rerun to continue.
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in chunkybert baly_random baly_media baly_aa baly_outlet no_classifier; do
  python3 evaluation/long10k/score.py run --model $m 2>&1 | grep -vE "Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|^- UNEXPECTED|^$|HF_TOKEN"
done
echo "ALL DONE"
