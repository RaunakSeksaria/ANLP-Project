#!/usr/bin/env python3
"""Prepare baseline and ideology-triplet JSONL files from AllSides events."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


SPLITS = ("train", "validation", "test")
ROLES = ("left", "center", "right")
SPLIT_ALIASES = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}


@dataclass(frozen=True)
class Article:
    article_id: str
    ideology: str
    title: str
    text: str


@dataclass(frozen=True)
class Event:
    event_id: str
    split: str
    left: Article
    center: Article
    right: Article


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_id(value: str) -> tuple[int, int | str]:
    """Sort numeric IDs numerically and all other IDs lexically."""
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def normalize_split(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return SPLIT_ALIASES.get(value.strip().lower())


def find_event_files(input_dir: Path) -> list[Path]:
    event_dir = input_dir / "events" if (input_dir / "events").is_dir() else input_dir
    files = sorted(event_dir.glob("*.json"), key=lambda path: stable_id(path.stem))
    if not files:
        raise ValueError(f"No event JSON files found in {event_dir}")
    return files


def find_articles_file(input_dir: Path) -> Path | None:
    for candidate in (input_dir / "articles.jsonl", input_dir.parent / "articles.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def load_split_index(path: Path | None) -> dict[str, str]:
    """Load the released article-to-split assignments, rejecting ambiguity."""
    if path is None:
        return {}
    index: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            article_id = clean_scalar(record.get("article_id"))
            split = normalize_split(record.get("split"))
            if not article_id or split is None:
                continue
            previous = index.setdefault(article_id, split)
            if previous != split:
                raise ValueError(f"Conflicting split assignments for article {article_id!r}")
    return index


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip() if isinstance(value, (str, int, float)) else ""


def recover_event_split(raw: Mapping[str, Any], split_index: Mapping[str, str], source: Path) -> str:
    candidates: set[str] = set()
    top_split = normalize_split(raw.get("split"))
    if top_split:
        candidates.add(top_split)
    for role in ROLES:
        article = raw.get(role)
        if not isinstance(article, Mapping):
            continue
        embedded = normalize_split(article.get("split"))
        if embedded:
            candidates.add(embedded)
        article_id = clean_scalar(article.get("article_id"))
        if article_id in split_index:
            candidates.add(split_index[article_id])
    if not candidates:
        raise ValueError(
            f"Cannot recover an existing split assignment for {source}; refusing to invent one"
        )
    if len(candidates) != 1:
        raise ValueError(f"Inconsistent split assignments for {source}: {sorted(candidates)}")
    return next(iter(candidates))


def parse_article(raw: Any, role: str, reasons: list[str]) -> Article | None:
    if not isinstance(raw, Mapping):
        reasons.append(f"missing_{role}_article")
        return None
    article_id = clean_scalar(raw.get("article_id"))
    if not article_id:
        reasons.append(f"missing_{role}_article_id")
    title_value = raw.get("title", "")
    title = title_value.strip() if isinstance(title_value, str) else ""
    body_value = raw.get("body")
    text = body_value.strip() if isinstance(body_value, str) else ""
    if not text:
        reasons.append(f"missing_{role}_body")
    if not article_id or not text:
        return None
    return Article(article_id, role, title, text)


def parse_event(raw: Any, split: str) -> tuple[Event | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(raw, Mapping):
        return None, ["event_not_object"]
    event_id = clean_scalar(raw.get("story_id"))
    if not event_id:
        reasons.append("missing_event_id")
    parsed = {role: parse_article(raw.get(role), role, reasons) for role in ROLES}
    if reasons:
        return None, sorted(set(reasons))
    assert event_id and all(parsed.values())
    return Event(event_id, split, parsed["left"], parsed["center"], parsed["right"]), []  # type: ignore[arg-type]


def load_events(
    files: Iterable[Path], split_index: Mapping[str, str]
) -> tuple[dict[str, list[Event]], Counter[str], dict[str, Counter[str]]]:
    events = {split: [] for split in SPLITS}
    source_counts: Counter[str] = Counter()
    rejection_reasons = {split: Counter() for split in SPLITS}
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read event record {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"Cannot recover an existing split assignment for non-object event {path}")
        split = recover_event_split(raw, split_index, path)
        source_counts[split] += 1
        event, reasons = parse_event(raw, split)
        if event is None:
            rejection_reasons[split].update(reasons)
        else:
            events[split].append(event)
    for split in SPLITS:
        events[split].sort(key=lambda event: stable_id(event.event_id))
    return events, source_counts, rejection_reasons


def baseline_record(event: Event, role: str) -> dict[str, str]:
    source = getattr(event, role)
    return {
        "example_id": f"{event.event_id}:{role}-to-center",
        "split": event.split,
        "event_id": event.event_id,
        "input_article_id": source.article_id,
        "input_ideology": role,
        "input_title": source.title,
        "input_text": source.text,
        "target_article_id": event.center.article_id,
        "target_title": event.center.title,
        "target_text": event.center.text,
    }


def looks_like_path_or_serialized_object(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            if isinstance(json.loads(stripped), (dict, list)):
                return True
        except json.JSONDecodeError:
            pass
    if "\n" not in stripped and not any(character.isspace() for character in stripped):
        lowered = stripped.lower()
        return lowered.startswith(("file://", "./", "../", "/")) or lowered.endswith(
            (".json", ".jsonl", ".txt", ".html", ".xml")
        )
    return False


def assert_triplet(
    record: Mapping[str, str], article_splits: Mapping[str, str], expected_target_article_id: str
) -> None:
    assert record["anchor_event_id"] == record["positive_event_id"]
    assert record["anchor_event_id"] != record["negative_event_id"]
    assert record["anchor_ideology"] == record["negative_ideology"]
    assert record["anchor_ideology"] != record["positive_ideology"]
    assert record["target_article_id"] == expected_target_article_id
    article_ids = (
        record["anchor_article_id"],
        record["positive_article_id"],
        record["negative_article_id"],
        record["target_article_id"],
    )
    assert all(article_splits[article_id] == record["split"] for article_id in article_ids)
    for field in ("anchor_text", "positive_text", "negative_text", "target_text"):
        assert not looks_like_path_or_serialized_object(record[field]), f"Suspicious {field} in {record['triplet_id']}"


def triplet_record(
    event: Event, negative_event: Event, role: str, article_splits: Mapping[str, str]
) -> dict[str, str]:
    opposite = "right" if role == "left" else "left"
    anchor = getattr(event, role)
    positive = getattr(event, opposite)
    negative = getattr(negative_event, role)
    record = {
        "triplet_id": f"{event.event_id}:{role}-anchor",
        "split": event.split,
        "anchor_event_id": event.event_id,
        "anchor_article_id": anchor.article_id,
        "anchor_ideology": role,
        "anchor_title": anchor.title,
        "anchor_text": anchor.text,
        "positive_event_id": event.event_id,
        "positive_article_id": positive.article_id,
        "positive_ideology": opposite,
        "positive_title": positive.title,
        "positive_text": positive.text,
        "negative_event_id": negative_event.event_id,
        "negative_article_id": negative.article_id,
        "negative_ideology": role,
        "negative_title": negative.title,
        "negative_text": negative.text,
        "target_article_id": event.center.article_id,
        "target_title": event.center.title,
        "target_text": event.center.text,
    }
    assert_triplet(record, article_splits, event.center.article_id)
    return record


def write_jsonl(handle: TextIO, record: Mapping[str, str]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def duplicate_count(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def empty_split_counts() -> dict[str, int]:
    return {split: 0 for split in SPLITS}


def prepare(input_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    files = find_event_files(input_dir)
    split_index = load_split_index(find_articles_file(input_dir))
    events_by_split, source_counts, event_rejections = load_events(files, split_index)
    article_splits = {
        article.article_id: event.split
        for split in SPLITS
        for event in events_by_split[split]
        for article in (event.left, event.center, event.right)
    }

    baseline_counts = empty_split_counts()
    triplet_counts = empty_split_counts()
    anchor_counts = {split: {"left": 0, "right": 0} for split in SPLITS}
    triplet_rejections = {split: Counter() for split in SPLITS}
    rng = random.Random(seed)

    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        baseline_path = output_dir / split / "baseline.jsonl"
        triplet_path = output_dir / split / "triplets.jsonl"
        events = events_by_split[split]
        with baseline_path.open("w", encoding="utf-8", newline="\n") as baseline_handle, triplet_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as triplet_handle:
            for event in events:
                for role in ("left", "right"):
                    baseline = baseline_record(event, role)
                    for field in ("input_text", "target_text"):
                        assert not looks_like_path_or_serialized_object(baseline[field]), (
                            f"Suspicious {field} in {baseline['example_id']}"
                        )
                    write_jsonl(baseline_handle, baseline)
                    baseline_counts[split] += 1

                candidates = [candidate for candidate in events if candidate.event_id != event.event_id]
                for role in ("left", "right"):
                    if not candidates:
                        triplet_rejections[split]["no_same_split_same_ideology_different_event_negative"] += 1
                        continue
                    negative_event = rng.choice(candidates)
                    triplet = triplet_record(event, negative_event, role, article_splits)
                    write_jsonl(triplet_handle, triplet)
                    triplet_counts[split] += 1
                    anchor_counts[split][role] += 1

    missing_titles: dict[str, dict[str, int]] = {}
    duplicate_ids = empty_split_counts()
    duplicate_texts = empty_split_counts()
    for split in SPLITS:
        articles = [
            article
            for event in events_by_split[split]
            for article in (event.left, event.center, event.right)
        ]
        role_counts = {
            role: sum(not getattr(event, role).title for event in events_by_split[split]) for role in ROLES
        }
        missing_titles[split] = {**role_counts, "total": sum(role_counts.values())}
        duplicate_ids[split] = duplicate_count(article.article_id for article in articles)
        duplicate_texts[split] = duplicate_count(article.text for article in articles)

    rejected_counts = {
        split: source_counts[split] - len(events_by_split[split]) for split in SPLITS
    }
    manifest: dict[str, Any] = {
        "seed": seed,
        "input_directory": str(input_dir.resolve()),
        "source_events_per_split": {split: source_counts[split] for split in SPLITS},
        "valid_events_per_split": {split: len(events_by_split[split]) for split in SPLITS},
        "rejected_events_per_split": rejected_counts,
        "event_rejection_reasons": {
            split: dict(sorted(event_rejections[split].items())) for split in SPLITS
        },
        "baseline_records_per_split": baseline_counts,
        "triplet_records_per_split": triplet_counts,
        "triplet_anchor_counts": anchor_counts,
        "rejected_triplets_per_split": {
            split: sum(triplet_rejections[split].values()) for split in SPLITS
        },
        "triplet_rejection_reasons": {
            split: dict(sorted(triplet_rejections[split].items())) for split in SPLITS
        },
        "missing_title_counts": missing_titles,
        "duplicate_article_id_counts": {
            "per_split": duplicate_ids,
            "total": sum(duplicate_ids.values()),
        },
        "duplicate_article_text_counts": {
            "per_split": duplicate_texts,
            "total": sum(duplicate_texts.values()),
        },
        "all_triplet_constraints_passed": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = prepare(args.input_dir, args.output_dir, args.seed)
    except (AssertionError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
