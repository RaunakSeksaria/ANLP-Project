#!/usr/bin/env python3
"""Check the rebuilt datasets before any training starts.

1. The ideology set (Manas's dataset/side_neutral/prepared) must match, byte for byte,
   the SHA-256s logged in his W&B runs (side-neutral-first-token-*-lambda035).
2. The publisher set must differ from it in the negatives only: identical baseline
   files, identical anchors / positives / targets, every negative from the anchor's
   publisher and from a different event group, and the expected file hashes.
Exits 1 on any failure; writes a JSON report.
"""

import argparse
import filecmp
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
ROWS = {"train": 16447, "validation": 1778, "test": 1775}
IDEOLOGY_SHA256 = {  # Manas's W&B run config `file_sha256`
    "train/baseline.jsonl": "d3c76a58f5112aba41a742aea77741625925afef23c0c9fca0be769faaf0254a",
    "train/triplets.jsonl": "070181a392216b9f4a6e050959b5138f80ed24ad1592fd61265256507777b90e",
    "validation/baseline.jsonl": "b029e73217fa30fb9fe4ebe073fb8f4ff4f0b25a775506f0796aa95c33b7ee31",
    "validation/triplets.jsonl": "31def1d58e4c3146e63304226c069dfbb2bc9db778402174ee0e8d71b1b3591d",
    "test/baseline.jsonl": "0f5ebc841217ff8f70d9b425dae9eb61baef80efe22d4c74d0619d5bce9ab169",
    "test/triplets.jsonl": "e35b52fc3fbfb63475da09b93bbba20090c5c80a3033600bdb08e5f2b11e15fe",
}
PUBLISHER_SHA256 = {  # same build, run on the laptop before submission
    "train/baseline.jsonl": IDEOLOGY_SHA256["train/baseline.jsonl"],
    "train/triplets.jsonl": "775dfdba003abed5de4bf3396be27701e0f83139d4c7298906ab381f8993d81a",
    "validation/baseline.jsonl": IDEOLOGY_SHA256["validation/baseline.jsonl"],
    "validation/triplets.jsonl": "104b019cd81ddc47b4316e245aa1c30f2f55da03166af9680cae82568223ac4e",
    "test/baseline.jsonl": IDEOLOGY_SHA256["test/baseline.jsonl"],
    "test/triplets.jsonl": "b95b685f584de399b24058acbd3ed7d4fea3ef5bd023cb2aa268bf7b0c073c1c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--ideology", type=Path, required=True)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    publisher_of = {}
    for side in ("left", "right"):
        for row in rows(args.raw / f"{side}_neutral_dedup.jsonl"):
            original = row[f"{side}_original"]
            publisher_of[original["article_id"]] = original["publisher"]

    report, failures = {"ideology_sha256": {}, "publisher_sha256": {}, "splits": {}}, []
    for name, expected in IDEOLOGY_SHA256.items():
        got = sha256(args.ideology / name)
        report["ideology_sha256"][name] = got
        if got != expected:
            failures.append(f"ideology {name}: {got} != Manas's {expected}")
    for name, expected in PUBLISHER_SHA256.items():
        got = sha256(args.publisher / name)
        report["publisher_sha256"][name] = got
        if got != expected:
            failures.append(f"publisher {name}: {got} != expected {expected}")

    for split in SPLITS:
        ideology = rows(args.ideology / split / "triplets.jsonl")
        publisher = rows(args.publisher / split / "triplets.jsonl")
        same_baseline = filecmp.cmp(args.ideology / split / "baseline.jsonl",
                                    args.publisher / split / "baseline.jsonl", shallow=False)
        same_rest = len(ideology) == len(publisher) and all(
            {k: v for k, v in a.items() if not k.startswith("negative_")}
            == {k: v for k, v in b.items() if not k.startswith("negative_")}
            for a, b in zip(ideology, publisher))
        same_publisher = sum(publisher_of[t["anchor_article_id"]] == publisher_of[t["negative_article_id"]]
                             for t in publisher)
        other_event = sum(t["negative_event_id"] != t["anchor_event_id"] for t in publisher)
        status = dict(Counter(t["positive_status"] for t in publisher))
        report["splits"][split] = {
            "rows": len(publisher), "baseline_identical": same_baseline,
            "anchors_positives_targets_identical": same_rest,
            "negative_same_publisher": same_publisher, "negative_other_event": other_event,
            "positive_status": status,
        }
        n = len(publisher)
        if n != ROWS[split]:
            failures.append(f"{split}: {n} rows, expected {ROWS[split]}")
        if not (same_baseline and same_rest and same_publisher == n and other_event == n):
            failures.append(f"{split}: publisher variant differs from the ideology set beyond its negatives")
        print(f"{split:10s} rows {n:6d} | baseline identical {same_baseline} | anchors/positives/targets "
              f"identical {same_rest} | same-publisher negatives {same_publisher}/{n} | "
              f"other-event negatives {other_event}/{n} | {status}")

    report["failures"] = failures
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("DATASET CHECK FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("dataset check passed: ideology set = Manas's (6/6 hashes), publisher set differs in negatives only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
