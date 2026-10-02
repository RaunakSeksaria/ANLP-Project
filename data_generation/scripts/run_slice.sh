#!/bin/bash
# Neutralise one slice of articles on one API key, to its own output file.
#
#   scripts/run_slice.sh <stance> <KEY_ENV_VAR> <input.jsonl> <tag> [workers] [rpm]
#   -> runs/nemotron/extra_<stance>_<tag>.jsonl   (build_single_release.py merges these)
#
# Several slices can run at once -- on different keys, or on one key -- as long
# as their inputs are disjoint. Resumable: re-running skips ids already written.
# If the endpoint trips the consecutive-failure limit, wait COOLOFF seconds and
# retry, up to MAX_TRIES; stop early with   touch runs/nemotron/STOP_<tag>
set -u
STANCE="${1:?stance}" KEYVAR="${2:?key env var}" IN="${3:?input}" TAG="${4:?tag}"
WORKERS="${5:-8}" RPM="${6:-24}"
COOLOFF="${COOLOFF:-600}" MAX_TRIES="${MAX_TRIES:-12}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p runs/nemotron
OUT="runs/nemotron/extra_${STANCE}_${TAG}.jsonl"
LOG="runs/nemotron/run_extra_${STANCE}_${TAG}.log"
STOP="runs/nemotron/STOP_${TAG}"

remaining() {
  uv run --no-sync python - "$IN" "$OUT" <<'EOF'
import json, sys
want = {json.loads(l)["story_id"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
done = set()
try:
    for l in open(sys.argv[2], encoding="utf-8"):
        r = json.loads(l)
        if r.get("summary"):
            done.add(r["story_id"])
except FileNotFoundError:
    pass
print(len(want - done))
EOF
}

for try in $(seq 1 "$MAX_TRIES"); do
  [ -f "$STOP" ] && { echo "[$(date '+%F %T')] $STOP present -- exiting" >> "$LOG"; exit 0; }
  left=$(remaining)
  echo "[$(date '+%F %T')] try $try/$MAX_TRIES: $left remaining in $IN" >> "$LOG"
  [ "$left" -eq 0 ] && exit 0
  uv run --no-sync python scripts/generate_neutral.py --task neutralise \
    --provider nvidia --api-key-env "$KEYVAR" --model nvidia/nemotron-3-ultra-550b-a55b \
    --in "$IN" --out "$OUT" --workers "$WORKERS" --rpm "$RPM" --thinking-level HIGH \
    --temperature 1.0 --top-p 0.95 --max-output-tokens 16384 --stop-after 5 \
    --set max_network_retries=3 --set backoff_base=4 --set backoff_cap=120 \
    --set enforce_length=false --set hard_max_ratio=3.0 >> "$LOG" 2>&1
  if tail -20 "$LOG" | grep -q "run aborted"; then
    echo "[$(date '+%F %T')] FATAL (bad key / model) -- not retrying" >> "$LOG"; exit 1
  fi
  [ "$(remaining)" -eq 0 ] && exit 0
  sleep "$COOLOFF"
done
echo "[$(date '+%F %T')] gave up after $MAX_TRIES tries" >> "$LOG"
exit 1
