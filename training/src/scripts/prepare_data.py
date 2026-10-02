#!/usr/bin/env python3
"""Prepare baseline pairs and ideology triplets from LLM-neutral stories."""

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


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT.parent / "dataset" / "raw" / "neutral_articles_10k.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "dataset" / "neutral_articles_10k_prepared"
SPLITS = ("train", "validation", "test")
ROLES = ("left", "center", "right")
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}
FLAT_ARTICLE_FIELDS = frozenset(
    {"article_id", "story_id", "stance", "split", "title", "body"}
)
NESTED_STORY_FIELDS = frozenset(
    {"story_id", "split", "left", "right", "center", "neutral", "source", "generator"}
)
NESTED_ARTICLE_FIELDS = frozenset({"article_id", "publisher", "title", "body"})


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
            "Directory containing neutral_articles_10k.jsonl/articles.jsonl, or a "
            "JSONL path itself "
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
    if input_path.is_dir():
        neutral_path = input_path / "neutral_articles_10k.jsonl"
        flat_path = input_path / "articles.jsonl"
        path = neutral_path if neutral_path.is_file() else flat_path
    else:
        path = input_path
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
    missing = sorted(FLAT_ARTICLE_FIELDS - record.keys())
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


def load_flat_events(path: Path) -> dict[str, list[Event]]:
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


def nested_article(
    value: Any,
    *,
    role: str,
    story_id: str,
    split: str,
    path: Path,
    line_number: int,
) -> Article:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: {role} must be a JSON object")
    missing = sorted(NESTED_ARTICLE_FIELDS - value.keys())
    if missing:
        raise ValueError(
            f"{path}:{line_number}: {role} is missing required fields: "
            f"{', '.join(missing)}"
        )
    article_id = required_string(value, "article_id", path, line_number)
    title = required_string(value, "title", path, line_number)
    text = required_string(value, "body", path, line_number)
    required_string(value, "publisher", path, line_number)
    if not article_id:
        raise ValueError(f"{path}:{line_number}: {role}.article_id is empty")
    if not text:
        raise ValueError(f"{path}:{line_number}: {role}.body is empty")
    return Article(article_id, story_id, role, split, title, text)


def parse_neutral_story(record: Any, path: Path, line_number: int) -> Event:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be a JSON object")
    missing = sorted(NESTED_STORY_FIELDS - record.keys())
    if missing:
        raise ValueError(
            f"{path}:{line_number}: missing required fields: {', '.join(missing)}"
        )
    story_id = required_string(record, "story_id", path, line_number)
    raw_split = required_string(record, "split", path, line_number).lower()
    neutral_text = required_string(record, "neutral", path, line_number)
    required_string(record, "source", path, line_number)
    if not story_id:
        raise ValueError(f"{path}:{line_number}: story_id is empty")
    if raw_split not in SPLIT_ALIASES:
        raise ValueError(f"{path}:{line_number}: unsupported split {raw_split!r}")
    if not neutral_text:
        raise ValueError(f"{path}:{line_number}: neutral is empty")
    if not isinstance(record["generator"], dict):
        raise ValueError(f"{path}:{line_number}: generator must be a JSON object")

    split = SPLIT_ALIASES[raw_split]
    left = nested_article(
        record["left"],
        role="left",
        story_id=story_id,
        split=split,
        path=path,
        line_number=line_number,
    )
    right = nested_article(
        record["right"],
        role="right",
        story_id=story_id,
        split=split,
        path=path,
        line_number=line_number,
    )
    # Validate the supplied original center article, but use the generated
    # neutral text—not that article—as the supervised generation target.
    original_center = nested_article(
        record["center"],
        role="center",
        story_id=story_id,
        split=split,
        path=path,
        line_number=line_number,
    )
    neutral_id = f"{story_id}:neutral"
    article_ids = {left.article_id, right.article_id, original_center.article_id}
    if len(article_ids) != 3:
        raise ValueError(f"{path}:{line_number}: story contains duplicate article IDs")
    if neutral_id in article_ids:
        raise ValueError(
            f"{path}:{line_number}: synthetic neutral article ID {neutral_id!r} "
            "collides with a source article ID"
        )
    neutral = Article(neutral_id, story_id, "center", split, "", neutral_text)
    return Event(story_id, split, left, neutral, right)


def detect_schema(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    if not first_line:
        raise ValueError(f"{path}: input file is empty")
    if not first_line.strip():
        raise ValueError(f"{path}:1: blank lines are not allowed")
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:1: malformed JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path}:1: record must be a JSON object")
    if NESTED_STORY_FIELDS <= record.keys():
        return "nested_neutral_story"
    if FLAT_ARTICLE_FIELDS <= record.keys():
        return "flat_article"
    raise ValueError(
        f"{path}:1: unsupported schema; expected flat article fields or nested "
        "neutral-story fields"
    )


def load_neutral_events(path: Path) -> dict[str, list[Event]]:
    events = {split: [] for split in SPLITS}
    seen_story_ids: set[str] = set()
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
            event = parse_neutral_story(value, path, line_number)
            if event.event_id in seen_story_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate story_id {event.event_id!r}"
                )
            seen_story_ids.add(event.event_id)
            events[event.split].append(event)
    for split in SPLITS:
        events[split].sort(key=lambda event: stable_id(event.event_id))
    return events


def load_events(path: Path) -> dict[str, list[Event]]:
    schema = detect_schema(path)
    if schema == "nested_neutral_story":
        return load_neutral_events(path)
    return load_flat_events(path)


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
    source_schema = detect_schema(source_path)
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
        "source_schema": source_schema,
        "generation_target": (
            "neutral" if source_schema == "nested_neutral_story" else "center"
        ),
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
