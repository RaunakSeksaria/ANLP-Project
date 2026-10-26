#!/bin/bash
# Run several variants one after another (each waits for the previous to finish),
# so they don't compete for key 1's rate limit.
#
#   setsid nohup scripts/run_queue.sh <N> <variant> [variant ...] &
#   -> runs/experiments/queue.log
set -u
N="${1:?events per side}"; shift
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=runs/experiments/queue.log
for name in "$@"; do
  echo "[$(date '+%F %T')] start $name ($N/side)" >> "$LOG"
  scripts/run_variant.sh "$name" "$N" >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] $name refused -- skipping" >> "$LOG"; continue; }
  sleep 20
  while pgrep -f "experiments/$name/in_" > /dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] done  $name" >> "$LOG"
done
echo "[$(date '+%F %T')] queue finished" >> "$LOG"
