#!/bin/bash
# Jev on every neutral text. Cumulative spend cap $2.60 (user-approved). Resumable.
cd "$(dirname "$0")"
python3 score_neutral.py summaries --cap-usd 2.60
python3 score_neutral.py rewrites --dataset short --cap-usd 2.60
python3 score_neutral.py rewrites --dataset long10k --cap-usd 2.60
python3 score_neutral.py led --cap-usd 2.60
echo "JEV NEUTRAL DONE"
