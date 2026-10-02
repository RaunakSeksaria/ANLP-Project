#!/usr/bin/env python3
"""Keep only articles that can anchor an event triplet.

An article is pairable when at least one of its events also has an article from
the other side, so a same-event opposite-side positive exists. The rest (all in
train) are dropped. Output is sorted by article id, so it does not depend on the
order of the input files; prepare_side_neutral.py reads it next.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO / "dataset/raw",
                        help="holds left_neutral_dedup.jsonl and right_neutral_dedup.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO / "dataset/pairable")
    args = parser.parse_args()

    rows = {}
    for side in ("left", "right"):
        with (args.raw_dir / f"{side}_neutral_dedup.jsonl").open(encoding="utf-8") as handle:
            rows[side] = [json.loads(line) for line in handle if line.strip()]
    sides_by_event = defaultdict(set)
    for side, records in rows.items():
        for r in records:
            for event in set(r["event_ids"]) | {r["story_id"]}:
                sides_by_event[event].add(side)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for side, records in rows.items():
        keep = [r for r in records
                if any(len(sides_by_event[e]) == 2 for e in set(r["event_ids"]) | {r["story_id"]})]
        keep.sort(key=lambda r: r[f"{side}_original"]["article_id"])
        with (args.output_dir / f"{side}_neutral_dedup.jsonl").open("w", encoding="utf-8") as handle:
            for r in keep:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[side] = {"kept": dict(Counter(r["split"] for r in keep)),
                         "dropped": len(records) - len(keep)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
