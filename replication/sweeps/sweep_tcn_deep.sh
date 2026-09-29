#!/bin/bash
# TCN with a receptive field large enough to actually span a document:
# 8 dilated levels (dilations 1..128) -> RF 511 tokens, vs 31 at 4 levels.
cd "$(dirname "$0")/.."   # run from replication/
for v in "TCN no chunk, no attn" "TCN no chunk, attn" \
         "TCN with chunk, no attn" "TCN with chunk, attn"; do
  echo "=== $(date -u +%H:%M:%S) [1e-3 L8] $v ==="
  python3 run_variant.py --variant "$v" --data-csv ../data/article_bias_prediction.csv \
    --results-csv results/results_table8_lr1e-3.csv --learning-rate 1e-3 --tcn-levels 8 \
    --tag standalone-lr1e-3-L8 2>&1 | grep -E "^(RESULT|Traceback|RuntimeError|.*Error)" | tail -2
done
echo "SWEEP DONE"
