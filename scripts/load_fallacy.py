"""Tolerant loader for the logical-fallacy dataset.

`data/fallacy/logical-fallacy-dataset-v2.json` is NOT valid JSON. It looks
LLM-generated and is broken in at least two ways:

  1. ~19.3k records write the answer key as   "Answer: "A",   instead of
     "Answer": "A",
  2. many `output` / `Fallacy Deconstruction` values contain unescaped `"`
     characters, e.g.   "output":" "Flawed Premise: ...

So instead of json.loads() on the whole file we split it into record blocks
and pull the five known fields out with a regex, decoding each value
leniently. Result is cached as *.clean.jsonl (one valid JSON object per line).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/fallacy/logical-fallacy-dataset-v2.json"
CLEAN = ROOT / "data/fallacy/logical-fallacy-dataset-v2.clean.jsonl"

FIELDS = ("instruction", "question", "Answer", "Fallacy Deconstruction", "output")

# "Answer: "A"  ->  "Answer": "A"
BAD_ANSWER = re.compile(r'"Answer:\s*"([A-Z])"')
# start of any known field inside a record block
FIELD_START = re.compile(
    r'^[ \t]*"(instruction|question|Answer|Fallacy Deconstruction|output)"[ \t]*:',
    re.MULTILINE,
)
ESCAPES = {"\\n": "\n", "\\t": "\t", "\\r": "\r", '\\"': '"', "\\\\": "\\", "\\/": "/"}


def _decode(raw: str) -> str:
    """Turn a raw JSON-ish string literal into text, tolerating bad escapes."""
    raw = raw.strip().rstrip(",").strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]
    try:  # fast path: it was actually well-formed
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        pass
    out = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), raw)
    for a, b in ESCAPES.items():
        out = out.replace(a, b)
    return out


def parse(text: str) -> list[dict]:
    text = BAD_ANSWER.sub(r'"Answer": "\1"', text)
    # record boundaries: a line that is just  },  /  {  between objects
    blocks = re.split(r"\n\s*\},?\s*\n\s*\{\s*\n", text)
    records = []
    for block in blocks:
        starts = list(FIELD_START.finditer(block))
        if not starts:
            continue
        rec = {}
        for i, m in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(block)
            rec[m.group(1)] = _decode(block[m.end() : end])
        records.append(rec)
    return records


def load_records(force: bool = False) -> list[dict]:
    if CLEAN.exists() and not force:
        return [json.loads(l) for l in CLEAN.read_text(encoding="utf-8").splitlines() if l]
    records = parse(RAW.read_text(encoding="utf-8"))
    with CLEAN.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    import statistics
    from collections import Counter

    recs = load_records(force=True)
    print(f"records: {len(recs)}  ->  {CLEAN.relative_to(ROOT)}")

    keys = Counter(k for r in recs for k in r)
    print("\nfield presence:")
    for k, n in keys.most_common():
        print(f"  {k:24s} {n:6d}  ({n / len(recs):.1%})")

    print("\nAnswer:", Counter(r.get("Answer") for r in recs).most_common())
    print("distinct instructions:", len({r.get("instruction") for r in recs}))
    print("\ntop fallacy labels:")
    for lab, n in Counter(
        r.get("Fallacy Deconstruction", "")[:60] for r in recs
    ).most_common(10):
        print(f"  {n:5d}  {lab!r}")

    for field in ("question", "output"):
        lens = [len(r[field]) for r in recs if r.get(field)]
        print(
            f"\n{field}: n={len(lens)} chars min={min(lens)} "
            f"median={int(statistics.median(lens))} max={max(lens)}"
        )

    print("\n--- sample records ---")
    for r in recs[:2]:
        print(json.dumps(r, ensure_ascii=False, indent=2)[:1200])
        print()
