#!/usr/bin/env python3
"""Rewrite only the negatives of prepared triplets under a different policy.

Anchors, positives, targets and baseline files are copied unchanged, so a run on
the output differs from a run on the input in its negatives alone.

Policies (all keep the negative in the anchor's split and outside its event group):
  publisher                same publisher as the anchor (proposal variant a)
  ideology-other-publisher same ideology, different publisher
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_side_neutral import SPLITS, canonical, read_rows, validate_prepared, write_json, write_rows  # noqa: E402

NEGATIVE_FIELDS = (("event_id", "event_id"), ("article_id", "input_article_id"),
                   ("ideology", "input_ideology"), ("title", "input_title"),
                   ("text", "input_text"))
POLICIES = ("publisher", "ideology-other-publisher")


def publisher_index(left: Path, right: Path) -> dict[str, str]:
    index = {}
    for side, path in (("left", left), ("right", right)):
        for row in read_rows(path):
            original = row[f"{side}_original"]
            index[original["article_id"]] = original["publisher"]
    return index


def eligible(policy: str, anchor: dict, candidate: dict, publisher: dict[str, str]) -> bool:
    same_publisher = publisher[candidate["input_article_id"]] == publisher[anchor["input_article_id"]]
    if policy == "publisher":
        return same_publisher
    return candidate["input_ideology"] == anchor["input_ideology"] and not same_publisher


def rewrite_split(policy, baselines, triplets, publisher, rng):
    by_id = {row["input_article_id"]: row for row in baselines}
    pools = defaultdict(list)
    for row in sorted(baselines, key=lambda r: r["input_article_id"]):
        key = publisher[row["input_article_id"]] if policy == "publisher" else row["input_ideology"]
        pools[key].append(row)
    text = {row["input_article_id"]: canonical(row["input_text"]) for row in baselines}
    rewritten = []
    for triplet in triplets:
        anchor = by_id[triplet["anchor_article_id"]]
        positive = by_id[triplet["positive_article_id"]]
        key = publisher[anchor["input_article_id"]] if policy == "publisher" else anchor["input_ideology"]
        pool = [row for row in pools[key]
                if row["event_id"] != anchor["event_id"]
                and eligible(policy, anchor, row, publisher)
                and text[row["input_article_id"]] not in (text[anchor["input_article_id"]],
                                                          text[positive["input_article_id"]])]
        if not pool:
            raise ValueError(f"no {policy} negative for {triplet['triplet_id']}")
        negative = rng.choice(pool)
        updated = dict(triplet)
        for suffix, key_name in NEGATIVE_FIELDS:
            updated[f"negative_{suffix}"] = negative[key_name]
        rewritten.append(updated)
    return rewritten


def check(policy, base, out, publisher):
    """Everything but the negative must be byte-identical; negatives must obey the policy."""
    report = {}
    for split in SPLITS:
        before = read_rows(base / split / "triplets.jsonl")
        after = read_rows(out / split / "triplets.jsonl")
        if len(before) != len(after):
            raise ValueError(f"{split}: triplet count changed")
        same_pub = changed = 0
        for b, a in zip(before, after):
            if {k: v for k, v in b.items() if not k.startswith("negative_")} != \
               {k: v for k, v in a.items() if not k.startswith("negative_")}:
                raise ValueError(f"{split}: non-negative field changed in {b['triplet_id']}")
            if a["negative_event_id"] == a["anchor_event_id"]:
                raise ValueError(f"{split}: negative inside anchor event group")
            anchor_pub = publisher[a["anchor_article_id"]]
            negative_pub = publisher[a["negative_article_id"]]
            if policy == "publisher" and negative_pub != anchor_pub:
                raise ValueError(f"{split}: negative publisher differs")
            if policy == "ideology-other-publisher" and (
                    negative_pub == anchor_pub or a["negative_ideology"] != a["anchor_ideology"]):
                raise ValueError(f"{split}: negative breaks ideology-other-publisher")
            same_pub += negative_pub == anchor_pub
            changed += a["negative_article_id"] != b["negative_article_id"]
        if (base / split / "baseline.jsonl").read_bytes() != (out / split / "baseline.jsonl").read_bytes():
            raise ValueError(f"{split}: baseline changed")
        report[split] = {"triplets": len(after), "negative_same_publisher": same_pub,
                         "negatives_changed": changed}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=ROOT / "dataset/side_neutral_full/prepared")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="written as <output-dir>/prepared/<split>/{baseline,triplets}.jsonl")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--left", type=Path, default=ROOT / "left_neutral_dedup.jsonl")
    parser.add_argument("--right", type=Path, default=ROOT / "right_neutral_dedup.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    publisher = publisher_index(args.left, args.right)
    out = args.output_dir / "prepared"
    rng = random.Random(f"{args.seed}:{args.policy}")
    for split in SPLITS:
        baselines = read_rows(args.base / split / "baseline.jsonl")
        missing = [r["input_article_id"] for r in baselines if r["input_article_id"] not in publisher]
        if missing:
            raise ValueError(f"{split}: {len(missing)} articles have no publisher, e.g. {missing[0]}")
        triplets = read_rows(args.base / split / "triplets.jsonl")
        (out / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base / split / "baseline.jsonl", out / split / "baseline.jsonl")
        write_rows(out / split / "triplets.jsonl", rewrite_split(policy=args.policy, baselines=baselines,
                                                                triplets=triplets, publisher=publisher, rng=rng))
    counts = validate_prepared(out)
    report = check(args.policy, args.base, out, publisher)
    base_manifest = json.loads((args.base / "manifest.json").read_text())
    by_publisher = {split: dict(Counter(publisher[r["input_article_id"]]
                                        for r in read_rows(out / split / "baseline.jsonl")))
                    for split in SPLITS}
    write_json(out / "manifest.json", {
        **base_manifest,
        "negative_policy": {"publisher": "same publisher as the anchor, outside its connected event group, same split",
                            "ideology-other-publisher": "same ideology, different publisher, outside its connected event group, same split",
                            }[args.policy],
        "derived_from": {"base": str(args.base.relative_to(ROOT)) if args.base.is_relative_to(ROOT) else str(args.base),
                         "base_triplets_sha256": {s: hashlib.sha256((args.base / s / "triplets.jsonl").read_bytes()).hexdigest()
                                                  for s in SPLITS},
                         "unchanged": "baseline files and every non-negative triplet field"},
        "variant_seed": f"{args.seed}:{args.policy}",
        "counts": counts, "variant_check": report, "anchors_by_publisher": by_publisher,
    })
    print(json.dumps({"counts": counts, "variant_check": report}, indent=2))


if __name__ == "__main__":
    main()
