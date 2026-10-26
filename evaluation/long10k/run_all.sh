#!/bin/bash
# Score both Nemotron rewrite sets with every model. Resumable: rerun to continue where it stopped.
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { python3 evaluation/long10k/score.py run --dataset $1 --model $2 2>&1 \
          | grep --line-buffered -vE "Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|^- UNEXPECTED|^$|HF_TOKEN" \
          | sed -u "s/^/[$1] /"; }
run short chunkybert; run short baly_random
for m in baly_media baly_aa baly_outlet no_classifier; do run long10k $m; done
for m in baly_media baly_aa baly_outlet no_classifier; do run short $m; done
echo "ALL DONE"
