#!/bin/bash
# Outer supervisor for the two neutralise runs, for unattended overnight use.
#
# watchdog_neutralise.sh deliberately EXITS when the endpoint looks unhealthy
# (5 consecutive failures), rather than hammering it. That is right for a
# minute-scale loop, but overnight nobody is there to restart it. This loop
# checks every 30 min and relaunches a stance's watchdog if it has exited and
# the stance is unfinished -- a 30-minute cool-off is long enough for a rate
# limit to clear. Relaunches are capped per stance, so a dead key cannot loop.
#
#   start:  setsid nohup scripts/supervise_neutralise.sh > /dev/null 2>&1 < /dev/null &
#   stop:   touch runs/nemotron/STOP_SUPERVISE
#
# Log: runs/nemotron/supervise_neutralise.log

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p runs/nemotron

INTERVAL=1800
MAX_RELAUNCH=6
TARGET=10000
LOG=runs/nemotron/supervise_neutralise.log
declare -A KEY=([left]=NEMOTRON_3_ULTRA_KEY [right]=NEMOTRON_3_ULTRA_KEY_2)
declare -A RELAUNCH=([left]=0 [right]=0)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "supervisor started (pid $$), every ${INTERVAL}s"

while true; do
  [ -f runs/nemotron/STOP_SUPERVISE ] && { log "STOP_SUPERVISE present -- exiting"; exit 0; }
  finished=0
  status=""
  for s in left right; do
    out=runs/nemotron/neutralised_$s.jsonl
    n=$(grep -c '"summary": "[^"]' "$out" 2>/dev/null || echo 0)
    status+="$s=$n "
    if [ "$n" -ge "$TARGET" ]; then finished=$((finished + 1)); continue; fi
    [ -f runs/nemotron/STOP_NEUTRALISE_$s ] && continue
    pgrep -f "watchdog_neutralise.sh $s" > /dev/null && continue

    if [ "${RELAUNCH[$s]}" -ge "$MAX_RELAUNCH" ]; then
      log "$s: watchdog down, relaunch cap ($MAX_RELAUNCH) reached -- leaving it"
      continue
    fi
    RELAUNCH[$s]=$((RELAUNCH[$s] + 1))
    log "$s: watchdog not running at $n/$TARGET -- relaunch ${RELAUNCH[$s]}/$MAX_RELAUNCH"
    tail -3 runs/nemotron/watchdog_neutralise_$s.log 2>/dev/null | sed 's/^/    /' >> "$LOG"
    setsid nohup scripts/watchdog_neutralise.sh "$s" "${KEY[$s]}" > /dev/null 2>&1 < /dev/null &
  done
  log "status: $status"
  [ "$finished" -eq 2 ] && { log "both stances at target -- exiting"; exit 0; }
  sleep "$INTERVAL"
done
