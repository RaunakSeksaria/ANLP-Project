#!/usr/bin/env python3
"""Prepare baseline pairs and ideology triplets from LLM-neutral articles."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT_PATH = PROJECT_ROOT / "dataset" / "llm_neutral" / "articles.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "llm_neutral_prepared"
SPLITS = ("train", "validation", "test")
ROLES = ("left", "center", "right")
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}
REQUIRED_ARTICLE_FIELDS = frozenset(
    {"article_id", "story_id", "stance", "split", "title", "body"}
)


@dataclass(frozen=True)
class Article:
    article_id: str
    story_id: str
    ideology: str
    split: str
    title: str
    text: str


@dataclass(frozen=True)
class Event:
    event_id: str
    split: str
    left: Article
    center: Article
    right: Article


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Directory containing articles.jsonl, or the articles.jsonl path itself "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Prepared-data output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def stable_id(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def articles_path(input_path: Path) -> Path:
    path = input_path / "articles.jsonl" if input_path.is_dir() else input_path
    if not path.is_file():
        raise FileNotFoundError(f"Article JSONL file not found: {path}")
    return path


def required_string(
    record: Mapping[str, Any], field: str, path: Path, line_number: int
) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line_number}: {field} must be a string")
    return value.strip()


def parse_article(record: Any, path: Path, line_number: int) -> Article:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be a JSON object")
    missing = sorted(REQUIRED_ARTICLE_FIELDS - record.keys())
    if missing:
        raise ValueError(
            f"{path}:{line_number}: missing required fields: {', '.join(missing)}"
        )
    article_id = required_string(record, "article_id", path, line_number)
    story_id = required_string(record, "story_id", path, line_number)
    ideology = required_string(record, "stance", path, line_number).lower()
    raw_split = required_string(record, "split", path, line_number).lower()
    title = required_string(record, "title", path, line_number)
    text = required_string(record, "body", path, line_number)
    if not article_id:
        raise ValueError(f"{path}:{line_number}: article_id is empty")
    if not story_id:
        raise ValueError(f"{path}:{line_number}: story_id is empty")
    if ideology not in ROLES:
        raise ValueError(f"{path}:{line_number}: unsupported stance {ideology!r}")
    if raw_split not in SPLIT_ALIASES:
        raise ValueError(f"{path}:{line_number}: unsupported split {raw_split!r}")
    if not text:
        raise ValueError(f"{path}:{line_number}: body is empty")
    return Article(
        article_id=article_id,
        story_id=story_id,
        ideology=ideology,
        split=SPLIT_ALIASES[raw_split],
        title=title,
        text=text,
    )


def load_events(path: Path) -> dict[str, list[Event]]:
    grouped: dict[str, dict[str, Article]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            article = parse_article(value, path, line_number)
            roles = grouped[article.story_id]
            if article.ideology in roles:
                raise ValueError(
                    f"{path}:{line_number}: story {article.story_id!r} has multiple "
                    f"{article.ideology} articles"
                )
            roles[article.ideology] = article

    events = {split: [] for split in SPLITS}
    for story_id in sorted(grouped, key=stable_id):
        roles = grouped[story_id]
        if set(roles) != set(ROLES):
            raise ValueError(
                f"Story {story_id!r} must contain exactly left, center, and right; "
                f"found {sorted(roles)}"
            )
        article_splits = {article.split for article in roles.values()}
        if len(article_splits) != 1:
            raise ValueError(
                f"Story {story_id!r} has inconsistent splits: {sorted(article_splits)}"
            )
        split = next(iter(article_splits))
        events[split].append(
            Event(story_id, split, roles["left"], roles["center"], roles["right"])
        )
    return events


def baseline_record(event: Event, ideology: str) -> dict[str, str]:
    source = getattr(event, ideology)
    return {
        "example_id": f"{event.event_id}:{ideology}-to-center",
        "split": event.split,
        "event_id": event.event_id,
        "input_article_id": source.article_id,
        "input_ideology": ideology,
        "input_title": source.title,
        "input_text": source.text,
        "target_article_id": event.center.article_id,
        "target_title": event.center.title,
        "target_text": event.center.text,
    }


def triplet_record(event: Event, negative_event: Event, ideology: str) -> dict[str, str]:
    positive_ideology = "right" if ideology == "left" else "left"
    anchor = getattr(event, ideology)
    positive = getattr(event, positive_ideology)
    negative = getattr(negative_event, ideology)
    record = {
        "triplet_id": f"{event.event_id}:{ideology}-anchor",
        "split": event.split,
        "anchor_event_id": event.event_id,
        "anchor_article_id": anchor.article_id,
        "anchor_ideology": ideology,
        "anchor_title": anchor.title,
        "anchor_text": anchor.text,
        "positive_event_id": event.event_id,
        "positive_article_id": positive.article_id,
        "positive_ideology": positive_ideology,
        "positive_title": positive.title,
        "positive_text": positive.text,
        "negative_event_id": negative_event.event_id,
        "negative_article_id": negative.article_id,
        "negative_ideology": ideology,
        "negative_title": negative.title,
        "negative_text": negative.text,
        "target_article_id": event.center.article_id,
        "target_title": event.center.title,
        "target_text": event.center.text,
    }
    assert record["anchor_event_id"] == record["positive_event_id"]
    assert record["anchor_event_id"] != record["negative_event_id"]
    assert record["anchor_ideology"] == record["negative_ideology"]
    assert record["anchor_ideology"] != record["positive_ideology"]
    assert negative_event.split == event.split
    assert record["target_article_id"] == event.center.article_id
    return record


def write_jsonl(handle: TextIO, record: Mapping[str, str]) -> None:
    handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def duplicate_extras(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def prepare(input_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    source_path = articles_path(input_path)
    events_by_split = load_events(source_path)
    rng = random.Random(seed)
    baseline_counts = {split: 0 for split in SPLITS}
    triplet_counts = {split: 0 for split in SPLITS}
    anchor_counts = {split: {"left": 0, "right": 0} for split in SPLITS}
    triplet_rejections = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        events = events_by_split[split]
        baseline_path = split_dir / "baseline.jsonl"
        triplet_path = split_dir / "triplets.jsonl"
        with baseline_path.open("w", encoding="utf-8", newline="\n") as baseline_handle, triplet_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as triplet_handle:
            for event in events:
                for ideology in ("left", "right"):
                    write_jsonl(baseline_handle, baseline_record(event, ideology))
                    baseline_counts[split] += 1
                candidates = [candidate for candidate in events if candidate.event_id != event.event_id]
                for ideology in ("left", "right"):
                    if not candidates:
                        triplet_rejections[split][
                            "no_same_split_same_ideology_different_event_negative"
                        ] += 1
                        continue
                    negative_event = rng.choice(candidates)
                    write_jsonl(
                        triplet_handle,
                        triplet_record(event, negative_event, ideology),
                    )
                    triplet_counts[split] += 1
                    anchor_counts[split][ideology] += 1

    missing_titles: dict[str, dict[str, int]] = {}
    duplicate_ids: dict[str, int] = {}
    duplicate_texts: dict[str, int] = {}
    for split in SPLITS:
        events = events_by_split[split]
        articles = [
            article
            for event in events
            for article in (event.left, event.center, event.right)
        ]
        role_counts = {
            role: sum(not getattr(event, role).title for event in events) for role in ROLES
        }
        missing_titles[split] = {**role_counts, "total": sum(role_counts.values())}
        duplicate_ids[split] = duplicate_extras(article.article_id for article in articles)
        duplicate_texts[split] = duplicate_extras(article.text for article in articles)

    source_counts = {split: len(events_by_split[split]) for split in SPLITS}
    manifest: dict[str, Any] = {
        "seed": seed,
        "input_directory": str(input_path.resolve()),
        "input_articles_file": str(source_path.resolve()),
        "source_events_per_split": source_counts,
        "valid_events_per_split": source_counts,
        "rejected_events_per_split": {split: 0 for split in SPLITS},
        "event_rejection_reasons": {split: {} for split in SPLITS},
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
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare(args.input_dir, args.output_dir, args.seed)
    except (AssertionError, OSError, ValueError) as exc:
        print(f"PREPARATION ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
