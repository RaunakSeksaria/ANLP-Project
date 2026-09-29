#!/bin/bash
# Table 8 baselines, cheapest first: standalone embedding + GRU/TCN, one epoch each.
cd "$(dirname "$0")/.."   # run from replication/
DATA=../data/article_bias_prediction.csv
OUT=results/results_table8.csv
for v in "TCN no chunk, no attn" "TCN no chunk, attn" \
         "GRU no chunk, no attn" "GRU no chunk, attn" \
         "TCN with chunk, no attn" "TCN with chunk, attn" \
         "GRU with chunk, no attn" "GRU with chunk, attn"; do
  echo "=== $(date -u +%H:%M:%S) $v ==="
  python3 run_variant.py --variant "$v" --data-csv "$DATA" --results-csv "$OUT" --tag standalone 2>&1 \
    | grep -E "^(RESULT|\[|Traceback|RuntimeError|.*Error)" | tail -3
done
echo "SWEEP DONE"
