#!/bin/bash
# Run one generation-setting variant on the first N events of the fixed evaluation sample.
#
#   scripts/run_variant.sh <variant-name> <N>        e.g.  scripts/run_variant.sh think_off_t1.0 150
#
# The variant's settings come from experiments/variants.json (append-only: a new
# setting always gets a new name). Events are split between key 1 and key 3 (every
# 6th event to key 3); each slice runs via scripts/run_slice.sh, so it is resumable
# and retries after a cool-off. Outputs go to runs/experiments/<name>/ and are never
# overwritten: re-running resumes, and a changed spec under an existing name is refused.
# Raising N later (150 -> 500) only adds the new events.
set -u
NAME="${1:?variant name}" N="${2:?number of events per side}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
DIR="runs/experiments/$NAME"
mkdir -p "$DIR"
eval "$(.venv/bin/python3 - "$NAME" "$N" "$DIR" <<'PY'
import json, shlex, sys
name, n, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
specs = {v["name"]: v for v in json.load(open("experiments/variants.json"))}
if name not in specs:
    sys.exit(f"unknown variant {name!r}; add it to experiments/variants.json")
spec = {k: v for k, v in specs[name].items() if k != "note"}
snap = f"{d}/spec.json"
try:
    old = json.load(open(snap))
    if old != spec:
        sys.exit(f"spec for {name} changed since it was first run -- use a new name")
except FileNotFoundError:
    json.dump(spec, open(snap, "w"), indent=1)
for st in ("left", "right"):
    rows = [l for l in open(f"experiments/eval500/{st}.jsonl") if l.strip()][:n]
    open(f"{d}/in_{st}_k1.jsonl", "w").writelines(r for i, r in enumerate(rows) if i % 6 != 5)
    open(f"{d}/in_{st}_k3.jsonl", "w").writelines(r for i, r in enumerate(rows) if i % 6 == 5)
prompt_args = "" if spec["prompt_version"] == "n2" else f"--set neutralise_prompt={spec['prompt_version']}"
for k, v in (("TEMP", spec["temperature"]), ("TOP_P", spec["top_p"]), ("SEED", spec["seed"]),
             ("TOP_K", "" if spec["top_k"] is None else spec["top_k"]), ("THINK", spec["thinking"]),
             ("PROMPT_ARGS", prompt_args)):
    print(f"export {k}={shlex.quote(str(v))}")
PY
)" || exit 1
export OUTDIR="$DIR"
for st in left right; do
  setsid nohup scripts/run_slice.sh "$st" NEMOTRON_3_ULTRA_KEY   "$DIR/in_${st}_k1.jsonl" k1 4 12 >/dev/null 2>&1 </dev/null &
  setsid nohup scripts/run_slice.sh "$st" NEMOTRON_3_ULTRA_KEY_3 "$DIR/in_${st}_k3.jsonl" k3 1 2  >/dev/null 2>&1 </dev/null &
done
echo "launched $NAME on the first $N events per side -> $DIR"
