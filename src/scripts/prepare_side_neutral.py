#!/usr/bin/env python3
"""Prepare unique article-to-own-neutral pairs and event-matched triplets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
SCHEMA = "anchor_specific_neutral_v1"


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical(text):
    return " ".join(text.split())


def stats(lengths):
    values = sorted(lengths)
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": values[0], "max": values[-1],
        "p90": values[int((len(values) - 1) * .90)],
        "p95": values[int((len(values) - 1) * .95)],
        "p99": values[int((len(values) - 1) * .99)],
        "over_512": sum(x > 512 for x in values),
    }


def normalize(raw_by_side):
    records = []
    for side, rows in raw_by_side.items():
        for row in rows:
            original = row[f"{side}_original"]
            split = {"val": "validation"}.get(row["split"], row["split"])
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split}")
            article_id = original["article_id"]
            events = sorted(set(row["event_ids"]) | {row["story_id"]})
            record = {
                "example_id": f"{article_id}:{side}-to-{side}-neutral",
                "split": split, "event_id": "", "event_ids": events,
                "input_article_id": article_id, "input_ideology": side,
                "input_title": original["title"].strip(),
                "input_text": original["body"].strip(),
                "target_article_id": f"{article_id}:{side}-neutral",
                "target_title": "", "target_text": row[f"{side}_neutralised"].strip(),
            }
            if not record["input_text"] or not record["target_text"]:
                raise ValueError(f"Empty source or target: {article_id}")
            records.append(record)
    if len({r["input_article_id"] for r in records}) != len(records):
        raise ValueError("Duplicate article IDs in raw data")
    return records


def assign_components(records):
    """Keep shared events and duplicate source/target bodies within one split."""
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(records):
        keys = [("event", event) for event in row["event_ids"]]
        keys += [("text", canonical(row[field])) for field in ("input_text", "target_text")]
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    components = defaultdict(list)
    for index, row in enumerate(records):
        components[root(index)].append(row)
    for group in components.values():
        if len({r["split"] for r in group}) != 1:
            raise ValueError("Connected events/texts cross supplied splits; resolve leakage before preparing")
        component_id = "group:" + min(e for r in group for e in r["event_ids"])
        for row in group:
            row["event_id"] = component_id
    return {"count": len(components), "largest_articles": max(map(len, components.values()))}


def make_triplets(records, token_ids, seed):
    rng = random.Random(seed)
    by_event = defaultdict(list)
    pools = defaultdict(list)
    for row in records:
        for event in row["event_ids"]:
            by_event[event].append(row)
        pools[row["split"], row["input_ideology"]].append(row)
    triplets = []
    for anchor in records:
        candidates = {
            row["input_article_id"]: row
            for event in anchor["event_ids"] for row in by_event[event]
            if row["input_ideology"] != anchor["input_ideology"]
            and canonical(row["input_text"]) != canonical(anchor["input_text"])
            and token_ids[row["input_article_id"]] != token_ids[anchor["input_article_id"]]
        }
        if not candidates:
            raise ValueError(f"No valid opposite-side positive for {anchor['input_article_id']}")
        positive = rng.choice([candidates[k] for k in sorted(candidates)])
        pool = pools[anchor["split"], anchor["input_ideology"]]
        start = rng.randrange(len(pool))
        negative = None
        for offset in range(len(pool)):
            row = pool[(start + offset) % len(pool)]
            if row["event_id"] == anchor["event_id"]:
                continue
            if any(canonical(row["input_text"]) == canonical(x["input_text"])
                   or token_ids[row["input_article_id"]] == token_ids[x["input_article_id"]]
                   for x in (anchor, positive)):
                continue
            negative = row
            break
        if negative is None:
            raise ValueError("No valid same-side negative outside event group")
        triplet = {
            "triplet_id": anchor["input_article_id"] + ":anchor",
            "split": anchor["split"],
            **{key: anchor[key] for key in ("target_article_id", "target_title", "target_text")},
        }
        for prefix, row in (("anchor", anchor), ("positive", positive), ("negative", negative)):
            for suffix, key in (("event_id", "event_id"), ("article_id", "input_article_id"),
                                ("ideology", "input_ideology"), ("title", "input_title"),
                                ("text", "input_text")):
                triplet[f"{prefix}_{suffix}"] = row[key]
        triplets.append(triplet)
    return triplets


def validate_prepared(data_dir):
    """Validate exact anchor joins, original-event positives, negatives and leakage."""
    summaries, split_keys = {}, {}
    for split in SPLITS:
        baselines = read_rows(data_dir / split / "baseline.jsonl")
        triplets = read_rows(data_dir / split / "triplets.jsonl")
        index = {r["input_article_id"]: r for r in baselines}
        if not baselines or len(index) != len(baselines):
            raise ValueError(f"{split}: empty data or duplicate article IDs")
        if len({r["example_id"] for r in baselines}) != len(baselines):
            raise ValueError(f"{split}: duplicate example IDs")
        if len({r["triplet_id"] for r in triplets}) != len(triplets):
            raise ValueError(f"{split}: duplicate triplet IDs")
        if Counter(r["anchor_article_id"] for r in triplets) != Counter(index.keys()):
            raise ValueError(f"{split}: expected exactly one triplet per baseline anchor")
        keys = set()
        for r in baselines:
            if r["split"] != split or r["input_ideology"] not in ("left", "right"):
                raise ValueError("Invalid split or side")
            if not r["input_text"].strip() or not r["target_text"].strip() or not r["event_ids"]:
                raise ValueError("Empty source, target or event IDs")
            expected = f"{r['input_article_id']}:{r['input_ideology']}-neutral"
            if r["target_article_id"] != expected:
                raise ValueError("Target is not specific to the input article and side")
            keys.update(("event", e) for e in r["event_ids"])
            keys.add(("component", r["event_id"]))
            keys.add(("article", r["input_article_id"]))
            keys.update(("text", canonical(r[k])) for k in ("input_text", "target_text"))
        for t in triplets:
            if t["split"] != split:
                raise ValueError("Triplet split mismatch")
            a, p, n = [index[t[f"{role}_article_id"]] for role in ("anchor", "positive", "negative")]
            for role, row in (("anchor", a), ("positive", p), ("negative", n)):
                for suffix, key in (("event_id", "event_id"), ("ideology", "input_ideology"),
                                    ("title", "input_title"), ("text", "input_text")):
                    if t[f"{role}_{suffix}"] != row[key]:
                        raise ValueError(f"{role} metadata mismatch")
            if a["event_id"] != p["event_id"] or not set(a["event_ids"]) & set(p["event_ids"]):
                raise ValueError("Positive must share an actual original event with anchor")
            if a["input_ideology"] == p["input_ideology"] or a["input_ideology"] != n["input_ideology"]:
                raise ValueError("Invalid positive or negative ideology")
            if a["event_id"] == n["event_id"]:
                raise ValueError("Negative belongs to anchor's connected event group")
            if len({canonical(x["input_text"]) for x in (a, p, n)}) != 3:
                raise ValueError("Triplet contains duplicate bodies")
            if any(t[k] != a[k] for k in ("target_article_id", "target_title", "target_text")):
                raise ValueError("Triplet target differs from anchor-specific baseline target")
        split_keys[split] = keys
        summaries[split] = {
            "baseline_records": len(baselines), "triplet_records": len(triplets),
            "left": sum(r["input_ideology"] == "left" for r in baselines),
            "right": sum(r["input_ideology"] == "right" for r in baselines),
            "event_groups": len({r["event_id"] for r in baselines}),
        }
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if split_keys[a] & split_keys[b]:
                raise ValueError(f"Event/article/body leakage between {a} and {b}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=ROOT / "left_neutral_dedup.jsonl")
    parser.add_argument("--right", type=Path, default=ROOT / "right_neutral_dedup.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/side_neutral")
    parser.add_argument("--tokenizer", default=str(ROOT / "outputs/joint_mean_lambda01/best"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = {"left": read_rows(args.left), "right": read_rows(args.right)}
    records = normalize(raw)
    components = assign_components(records)
    sources = [f"{r['input_title']} {tokenizer.sep_token} {r['input_text']}"
               if r['input_title'] else r['input_text'] for r in records]
    source_ids = tokenizer(sources, truncation=False, padding=False)["input_ids"]
    targets = tokenizer([r["target_text"] for r in records], truncation=False, padding=False)["input_ids"]
    # Match the 512-token training encoding when rejecting degenerate triplets.
    truncated = tokenizer(sources, truncation=True, max_length=512, padding=False)["input_ids"]
    token_ids = {r["input_article_id"]: tuple(ids) for r, ids in zip(records, truncated)}
    triplets = make_triplets(records, token_ids, args.seed)
    eda = {"tokenizer": args.tokenizer, "token_length_includes_special_tokens": True,
           "source_format": "title + sep_token + body (or body if title empty)",
           "event_components": components, "sides": {}}
    for side in ("left", "right"):
        indices = [i for i, r in enumerate(records) if r["input_ideology"] == side]
        side_rows = [records[i] for i in indices]
        eda["sides"][side] = {
            "records": len(indices), "unique_article_ids": len({r["input_article_id"] for r in side_rows}),
            "unique_bodies": len({canonical(r["input_text"]) for r in side_rows}),
            "unique_targets": len({canonical(r["target_text"]) for r in side_rows}),
            "original_events": len({e for r in side_rows for e in r["event_ids"]}),
            "split_counts": dict(Counter(r["split"] for r in side_rows)),
            "empty_titles": sum(not r["input_title"] for r in side_rows),
            "unchanged_targets": sum(canonical(r["input_text"]) == canonical(r["target_text"]) for r in side_rows),
            "body_words": stats([len(r["input_text"].split()) for r in side_rows]),
            "target_words": stats([len(r["target_text"].split()) for r in side_rows]),
            "source_tokens": stats([len(source_ids[i]) for i in indices]),
            "target_tokens": stats([len(targets[i]) for i in indices]),
        }
    prepared = args.output_dir / "prepared"
    for split in SPLITS:
        write_rows(prepared / split / "baseline.jsonl", [r for r in records if r["split"] == split])
        write_rows(prepared / split / "triplets.jsonl", [t for t in triplets if t["split"] == split])
    summaries = validate_prepared(prepared)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in (args.left, args.right):
        destination = raw_dir / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        hashes[path.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {"schema": SCHEMA, "seed": args.seed, "source_sha256": hashes,
                "generation_target": "anchor's own side-specific neutralised text",
                "positive_policy": "opposite ideology sharing an original event_id",
                "negative_policy": "same ideology and split, outside connected event component",
                "split_policy": "preserve supplied splits; val renamed validation; fail on leakage",
                "counts": summaries}
    write_json(prepared / "manifest.json", manifest)
    write_json(args.output_dir / "eda.json", eda)
    write_json(args.output_dir / "validation_report.json", {"passed": True, "splits": summaries})
    print(json.dumps({"eda": eda, "prepared_counts": summaries}, indent=2))


if __name__ == "__main__":
    main()
