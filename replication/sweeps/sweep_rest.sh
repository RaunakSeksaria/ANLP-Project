#!/bin/bash
# Finish the two missing lr=2e-5 runs, then repeat all 8 at lr=1e-3.
cd "$(dirname "$0")/.."   # run from replication/
DATA=../data/article_bias_prediction.csv

for v in "GRU with chunk, no attn" "GRU with chunk, attn"; do
  echo "=== $(date -u +%H:%M:%S) [2e-5] $v ==="
  python3 run_variant.py --variant "$v" --data-csv "$DATA" --results-csv results/results_table8.csv \
    --tag standalone 2>&1 | grep -E "^(RESULT|Traceback|RuntimeError|.*Error)" | tail -2
done

for v in "TCN no chunk, no attn" "TCN no chunk, attn" \
         "GRU no chunk, no attn" "GRU no chunk, attn" \
         "TCN with chunk, no attn" "TCN with chunk, attn" \
         "GRU with chunk, no attn" "GRU with chunk, attn"; do
  echo "=== $(date -u +%H:%M:%S) [1e-3] $v ==="
  python3 run_variant.py --variant "$v" --data-csv "$DATA" --results-csv results/results_table8_lr1e-3.csv \
    --learning-rate 1e-3 --tag standalone-lr1e-3 2>&1 | grep -E "^(RESULT|Traceback|RuntimeError|.*Error)" | tail -2
done
echo "SWEEP DONE"
