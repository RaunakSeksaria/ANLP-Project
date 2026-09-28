#!/bin/bash
STANCE="${1:?usage: watchdog_neutralise.sh <left|right> [API_KEY_ENV_VAR]}"
KEYVAR="${2:-NEMOTRON_3_ULTRA_KEY}"
# Keeps the Nemotron 10k run alive overnight.
#
# The generator is resumable -- it skips story_ids already written with a
# summary -- so restarting it is always safe and never duplicates work.
#
# Every 60s: if the run is not alive and the target is not met, record WHY it
# stopped and restart it. Each event is appended to watchdog.log with the record
# count and the tail of the run log, so the morning diagnosis does not depend on
# anyone having watched it happen.
#
#   start:  setsid nohup scripts/watchdog_neutralise.sh <stance> <KEYVAR> > /dev/null 2>&1 < /dev/null &
#   stop:   touch runs/nemotron/STOP_NEUTRALISE_${STANCE}   (or pkill -f watchdog.sh)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p runs/nemotron

OUT=runs/nemotron/neutralised_${STANCE}.jsonl
RUNLOG=runs/nemotron/run_neutralise_${STANCE}.log
WLOG=runs/nemotron/watchdog_neutralise_${STANCE}.log
STOP=runs/nemotron/STOP_NEUTRALISE_${STANCE}
TARGET=10000
MAX_RESTARTS=20
restarts=0

log() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }

log "watchdog started (pid $$), target $TARGET"

while true; do
  [ -f "$STOP" ] && { log "STOP file present -- watchdog exiting"; exit 0; }

  done_n=$(grep -c '"summary": "[^"]' "$OUT" 2>/dev/null || echo 0)
  total_n=$(wc -l < "$OUT" 2>/dev/null || echo 0)

  if [ "$done_n" -ge "$TARGET" ]; then
    log "TARGET REACHED: $done_n summaries in $total_n records -- watchdog exiting"
    exit 0
  fi

  # Match on this stance's output file: the other stance's generator also runs
  # scripts/generate_neutral.py and would otherwise make a dead run look alive.
  if pgrep -f "neutralised_${STANCE}.jsonl" > /dev/null; then
    sleep 60
    continue
  fi

  # --- not running: diagnose before restarting -------------------------
  restarts=$((restarts + 1))
  log "RUN NOT ALIVE. summaries=$done_n records=$total_n restart=$restarts/$MAX_RESTARTS"
  log "  last run.log lines:"
  tail -6 "$RUNLOG" 2>/dev/null | sed 's/^/    /' >> "$WLOG"

  # What kinds of failure are actually in the output file? A restart is only
  # worth doing if the failures look transient, and the morning diagnosis needs
  # the distribution, not just the last line.
  log "  failure reasons recorded so far:"
  grep -o '"error": "[^"]*"' "$OUT" 2>/dev/null \
    | sed 's/"error": "//; s/"$//' | cut -c1-60 | sort | uniq -c | sort -rn \
    | head -6 | sed 's/^/    /' >> "$WLOG"

  # A clean 'stopped early' means 5 consecutive failures -- the endpoint or
  # the key is unhealthy. Back off hard rather than hammering it.
  if tail -20 "$RUNLOG" 2>/dev/null | grep -q "STOPPED EARLY"; then
    # The run itself judged the endpoint unhealthy. Restarting re-runs the same
    # doomed work and writes a row per cluster; that produced 188k junk records
    # across 21 restarts once. Stop and let a human look.
    log "  reason: consecutive-failure limit tripped. NOT restarting."
    log "          Completed records are safe. Fix the endpoint/key, then re-run"
    log "          the runner script by hand."
    exit 2
  elif tail -20 "$RUNLOG" 2>/dev/null | grep -q "run aborted"; then
    log "  reason: FATAL (bad key / unknown model). Not retryable -- exiting."
    exit 1
  fi

  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    log "  giving up after $MAX_RESTARTS restarts"
    exit 1
  fi

  log "  restarting..."
  setsid nohup scripts/run_neutralise.sh "$STANCE" "$KEYVAR" >> "$RUNLOG" 2>&1 < /dev/null &
  sleep 90
done
