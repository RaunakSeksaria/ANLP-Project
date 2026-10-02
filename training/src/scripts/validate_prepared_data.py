#!/usr/bin/env python3
"""Validate prepared AllSides baseline and triplet data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "validation", "test")
SPLIT_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
BASELINE_FIELDS = {
    "example_id",
    "split",
    "event_id",
    "input_article_id",
    "input_ideology",
    "input_title",
    "input_text",
    "target_article_id",
    "target_title",
    "target_text",
}
TRIPLET_FIELDS = {
    "triplet_id",
    "split",
    "anchor_event_id",
    "anchor_article_id",
    "anchor_ideology",
    "anchor_title",
    "anchor_text",
    "positive_event_id",
    "positive_article_id",
    "positive_ideology",
    "positive_title",
    "positive_text",
    "negative_event_id",
    "negative_article_id",
    "negative_ideology",
    "negative_title",
    "negative_text",
    "target_article_id",
    "target_title",
    "target_text",
}


@dataclass(frozen=True)
class LocatedRecord:
    data: dict[str, Any]
    path: Path
    line_number: int

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line_number}"


@dataclass
class ValidationResult:
    data_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    files: dict[str, dict[str, int]] = field(default_factory=dict)
    split_summaries: dict[str, dict[str, int]] = field(default_factory=dict)
    length_statistics: dict[str, dict[str, dict[str, float | int]]] = field(default_factory=dict)
    leakage: dict[str, dict[str, int]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args(argv)


def add_error(result: ValidationResult, message: str) -> None:
    result.errors.append(message)


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def validate_trimmed_string(
    record: LocatedRecord, field_name: str, result: ValidationResult, *, nonempty: bool = False
) -> None:
    value = record.data.get(field_name)
    if not isinstance(value, str):
        add_error(result, f"{record.location}: {field_name} must be a string")
    elif nonempty and not value:
        add_error(result, f"{record.location}: {field_name} must be non-empty")
    elif value != value.strip():
        add_error(result, f"{record.location}: {field_name} has surrounding whitespace")


def load_jsonl(path: Path, result: ValidationResult) -> list[LocatedRecord]:
    records: list[LocatedRecord] = []
    if not path.is_file():
        add_error(result, f"Missing required file: {path}")
        result.files[str(path)] = {"size_bytes": 0, "record_count": 0}
        return records
    try:
        size = path.stat().st_size
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    add_error(result, f"{path}:{line_number}: invalid JSON: {exc.msg}")
                    continue
                if not isinstance(value, dict):
                    add_error(result, f"{path}:{line_number}: record must be a JSON object")
                    continue
                records.append(LocatedRecord(value, path, line_number))
    except (OSError, UnicodeError) as exc:
        add_error(result, f"Cannot parse required file {path}: {exc}")
        size = path.stat().st_size if path.exists() else 0
    result.files[str(path)] = {"size_bytes": size, "record_count": len(records)}
    return records


def load_manifest(path: Path, result: ValidationResult) -> dict[str, Any] | None:
    if not path.is_file():
        add_error(result, f"Missing required file: {path}")
        result.files[str(path)] = {"size_bytes": 0, "record_count": 0}
        return None
    size = path.stat().st_size
    result.files[str(path)] = {"size_bytes": size, "record_count": 1}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        add_error(result, f"Cannot parse required manifest {path}: {exc}")
        return None
    if not isinstance(value, dict):
        add_error(result, f"{path}: manifest must be a JSON object")
        return None
    return value


def report_duplicate_ids(
    records: Iterable[LocatedRecord], id_field: str, result: ValidationResult
) -> None:
    locations: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = record.data.get(id_field)
        if isinstance(value, str):
            locations[value].append(record.location)
    for value, found_at in sorted(locations.items()):
        if len(found_at) > 1:
            add_error(
                result,
                f"Duplicate {id_field} {value!r} ({len(found_at)} occurrences): {', '.join(found_at)}",
            )


def report_duplicate_records(
    records: Iterable[LocatedRecord], label: str, result: ValidationResult
) -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        canonical = json.dumps(record.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        groups[canonical].append(record.location)
    for locations in groups.values():
        if len(locations) > 1:
            add_error(result, f"Duplicate {label} record ({len(locations)} occurrences): {', '.join(locations)}")


def validate_baselines(
    split: str, records: list[LocatedRecord], result: ValidationResult
) -> dict[str, dict[str, LocatedRecord]]:
    by_event: dict[str, dict[str, LocatedRecord]] = defaultdict(dict)
    report_duplicate_ids(records, "example_id", result)
    report_duplicate_records(records, "baseline", result)
    for record in records:
        missing = sorted(BASELINE_FIELDS - record.data.keys())
        if missing:
            add_error(result, f"{record.location}: missing baseline fields: {', '.join(missing)}")
        if record.data.get("split") != split:
            add_error(result, f"{record.location}: split does not match directory {split!r}")
        ideology = record.data.get("input_ideology")
        if ideology not in {"left", "right"}:
            add_error(result, f"{record.location}: input_ideology must be 'left' or 'right'")
        for name in ("event_id", "input_article_id", "target_article_id", "input_text", "target_text"):
            validate_trimmed_string(record, name, result, nonempty=True)
        for name in ("input_title", "target_title"):
            validate_trimmed_string(record, name, result)
        if (
            nonempty_string(record.data.get("input_article_id"))
            and record.data.get("input_article_id") == record.data.get("target_article_id")
        ):
            add_error(result, f"{record.location}: input and target article IDs are identical")
        event_id = record.data.get("event_id")
        if nonempty_string(event_id) and ideology in {"left", "right"}:
            if ideology in by_event[event_id]:
                add_error(result, f"{record.location}: event {event_id!r} has multiple {ideology} baselines")
            else:
                by_event[event_id][ideology] = record

    for event_id, roles in sorted(by_event.items()):
        if set(roles) != {"left", "right"}:
            add_error(result, f"{split}: event {event_id!r} does not have exactly one left and one right baseline")
            continue
        left, right = roles["left"].data, roles["right"].data
        if left.get("target_article_id") != right.get("target_article_id"):
            add_error(result, f"{split}: event {event_id!r} baseline target article IDs disagree")
        if left.get("target_text") != right.get("target_text"):
            add_error(result, f"{split}: event {event_id!r} baseline target texts disagree")
        if left.get("target_title") != right.get("target_title"):
            add_error(result, f"{split}: event {event_id!r} baseline target titles disagree")
    return by_event


def compare_article(
    triplet: LocatedRecord,
    prefix: str,
    baseline: LocatedRecord,
    expected_ideology: str,
    result: ValidationResult,
) -> None:
    pairs = (
        (f"{prefix}_article_id", "input_article_id"),
        (f"{prefix}_title", "input_title"),
        (f"{prefix}_text", "input_text"),
    )
    for triplet_field, baseline_field in pairs:
        if triplet.data.get(triplet_field) != baseline.data.get(baseline_field):
            add_error(
                result,
                f"{triplet.location}: {triplet_field} does not match baseline {baseline_field}",
            )
    if triplet.data.get(f"{prefix}_ideology") != expected_ideology:
        add_error(result, f"{triplet.location}: {prefix}_ideology does not match baseline ideology")


def validate_triplets(
    split: str,
    records: list[LocatedRecord],
    baselines: dict[str, dict[str, LocatedRecord]],
    result: ValidationResult,
) -> dict[str, dict[str, LocatedRecord]]:
    by_event: dict[str, dict[str, LocatedRecord]] = defaultdict(dict)
    baseline_by_article = {
        record.data.get("input_article_id"): record
        for roles in baselines.values()
        for record in roles.values()
        if nonempty_string(record.data.get("input_article_id"))
    }
    center_ids = {
        record.data.get("target_article_id")
        for roles in baselines.values()
        for record in roles.values()
        if nonempty_string(record.data.get("target_article_id"))
    }
    report_duplicate_ids(records, "triplet_id", result)
    report_duplicate_records(records, "triplet", result)

    for record in records:
        missing = sorted(TRIPLET_FIELDS - record.data.keys())
        if missing:
            add_error(result, f"{record.location}: missing triplet fields: {', '.join(missing)}")
        if record.data.get("split") != split:
            add_error(result, f"{record.location}: split does not match directory {split!r}")
        for name in (
            "anchor_event_id",
            "anchor_article_id",
            "positive_event_id",
            "positive_article_id",
            "negative_event_id",
            "negative_article_id",
            "target_article_id",
            "anchor_text",
            "positive_text",
            "negative_text",
            "target_text",
        ):
            validate_trimmed_string(record, name, result, nonempty=True)
        for name in ("anchor_title", "positive_title", "negative_title", "target_title"):
            validate_trimmed_string(record, name, result)

        anchor_event = record.data.get("anchor_event_id")
        positive_event = record.data.get("positive_event_id")
        negative_event = record.data.get("negative_event_id")
        anchor_ideology = record.data.get("anchor_ideology")
        positive_ideology = record.data.get("positive_ideology")
        negative_ideology = record.data.get("negative_ideology")
        for name, ideology in (
            ("anchor", anchor_ideology),
            ("positive", positive_ideology),
            ("negative", negative_ideology),
        ):
            if ideology not in {"left", "right"}:
                add_error(result, f"{record.location}: {name}_ideology must be 'left' or 'right'")
        if anchor_event != positive_event:
            add_error(result, f"{record.location}: anchor and positive event IDs differ")
        if anchor_event == negative_event:
            add_error(result, f"{record.location}: anchor and negative use the same event")
        if anchor_ideology != negative_ideology:
            add_error(result, f"{record.location}: anchor and negative ideologies differ")
        if {anchor_ideology, positive_ideology} != {"left", "right"}:
            add_error(result, f"{record.location}: anchor and positive ideologies are not opposite")
        role_article_ids = {
            record.data.get("anchor_article_id"),
            record.data.get("positive_article_id"),
            record.data.get("negative_article_id"),
        }
        used_centers = sorted(str(value) for value in role_article_ids & center_ids)
        if used_centers:
            add_error(result, f"{record.location}: center target used as triplet article: {used_centers}")

        if nonempty_string(anchor_event) and anchor_ideology in {"left", "right"}:
            if anchor_ideology in by_event[anchor_event]:
                add_error(result, f"{record.location}: event {anchor_event!r} has multiple {anchor_ideology}-anchor triplets")
            else:
                by_event[anchor_event][anchor_ideology] = record

        anchor_baseline = baselines.get(anchor_event, {}).get(anchor_ideology)
        if anchor_baseline is None:
            add_error(result, f"{record.location}: anchor cannot be resolved to a baseline input")
        else:
            compare_article(record, "anchor", anchor_baseline, str(anchor_ideology), result)
            for name in ("target_article_id", "target_title", "target_text"):
                if record.data.get(name) != anchor_baseline.data.get(name):
                    add_error(result, f"{record.location}: {name} does not match the anchor event baseline target")

        positive_baseline = baselines.get(anchor_event, {}).get(positive_ideology)
        if positive_baseline is None:
            add_error(result, f"{record.location}: positive cannot be resolved to the opposite baseline input")
        else:
            compare_article(record, "positive", positive_baseline, str(positive_ideology), result)
            if positive_event != positive_baseline.data.get("event_id"):
                add_error(result, f"{record.location}: positive event ID does not match baseline event")

        negative_baseline = baseline_by_article.get(record.data.get("negative_article_id"))
        if negative_baseline is None:
            add_error(result, f"{record.location}: negative article cannot be resolved in the same split")
        else:
            compare_article(record, "negative", negative_baseline, str(negative_ideology), result)
            if negative_event != negative_baseline.data.get("event_id"):
                add_error(result, f"{record.location}: negative event ID does not match baseline event")

    for event_id in sorted(set(baselines) | set(by_event)):
        roles = by_event.get(event_id, {})
        if event_id not in baselines:
            add_error(result, f"{split}: triplets refer to event {event_id!r} absent from baselines")
        if set(roles) != {"left", "right"}:
            add_error(result, f"{split}: event {event_id!r} does not have exactly one left-anchor and one right-anchor triplet")
    return by_event


def overlap_counts(sets: Mapping[str, set[str]]) -> dict[str, int]:
    return {f"{left}__{right}": len(sets[left] & sets[right]) for left, right in SPLIT_PAIRS}


def validate_leakage(
    baselines_by_split: Mapping[str, list[LocatedRecord]],
    triplets_by_split: Mapping[str, list[LocatedRecord]],
    result: ValidationResult,
) -> None:
    categories = {
        "event_ids": {
            split: {
                str(record.data["event_id"])
                for record in baselines_by_split[split]
                if nonempty_string(record.data.get("event_id"))
            }
            for split in SPLITS
        },
        "baseline_input_article_ids": {
            split: {
                str(record.data["input_article_id"])
                for record in baselines_by_split[split]
                if nonempty_string(record.data.get("input_article_id"))
            }
            for split in SPLITS
        },
    }
    for prefix in ("anchor", "positive", "negative"):
        field_name = f"{prefix}_article_id"
        categories[f"triplet_{prefix}_article_ids"] = {
            split: {
                str(record.data[field_name])
                for record in triplets_by_split[split]
                if nonempty_string(record.data.get(field_name))
            }
            for split in SPLITS
        }
    for category, values in categories.items():
        counts = overlap_counts(values)
        result.leakage[category] = counts
        for pair, count in counts.items():
            if count:
                add_error(result, f"Split leakage: {category} has {count} overlapping values in {pair}")


def manifest_split_count(
    manifest: Mapping[str, Any], section: str, split: str
) -> Any:
    section_value = manifest.get(section)
    return section_value.get(split) if isinstance(section_value, Mapping) else None


def validate_manifest(
    manifest: Mapping[str, Any] | None,
    baselines_by_split: Mapping[str, list[LocatedRecord]],
    triplets_by_split: Mapping[str, list[LocatedRecord]],
    result: ValidationResult,
) -> None:
    if manifest is None:
        return
    for split in SPLITS:
        actual_baselines = len(baselines_by_split[split])
        actual_triplets = len(triplets_by_split[split])
        stated_baselines = manifest_split_count(manifest, "baseline_records_per_split", split)
        stated_triplets = manifest_split_count(manifest, "triplet_records_per_split", split)
        if stated_baselines != actual_baselines:
            add_error(result, f"Manifest baseline count mismatch for {split}: {stated_baselines!r} != {actual_baselines}")
        if stated_triplets != actual_triplets:
            add_error(result, f"Manifest triplet count mismatch for {split}: {stated_triplets!r} != {actual_triplets}")
        anchor_section = manifest.get("triplet_anchor_counts")
        stated_anchors = anchor_section.get(split) if isinstance(anchor_section, Mapping) else None
        for role in ("left", "right"):
            actual = sum(record.data.get("anchor_ideology") == role for record in triplets_by_split[split])
            stated = stated_anchors.get(role) if isinstance(stated_anchors, Mapping) else None
            if stated != actual:
                add_error(result, f"Manifest {role}-anchor count mismatch for {split}: {stated!r} != {actual}")


def percentile_95(values: Sequence[int]) -> float | int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def length_stats(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "median": 0, "mean": 0, "p95": 0, "maximum": 0}
    median = statistics.median(values)
    return {
        "minimum": min(values),
        "median": int(median) if float(median).is_integer() else median,
        "mean": round(statistics.fmean(values), 2),
        "p95": percentile_95(values),
        "maximum": max(values),
    }


def duplicate_text_groups(entries: Iterable[tuple[str, Any]]) -> list[list[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for article_id, text in entries:
        if isinstance(text, str):
            groups[text].add(article_id)
    return sorted((sorted(ids) for ids in groups.values() if len(ids) > 1), key=lambda ids: tuple(ids))


def collect_diagnostics(
    baselines_by_split: Mapping[str, list[LocatedRecord]], result: ValidationResult
) -> None:
    empty_titles: dict[str, dict[str, int]] = {}
    duplicate_inputs: dict[str, list[list[str]]] = {}
    duplicate_targets: dict[str, list[list[str]]] = {}
    identical_sides: dict[str, list[str]] = {}
    extremes: dict[str, dict[str, list[dict[str, int | str]]]] = {}
    unique_counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        records = baselines_by_split[split]
        usable = [
            record
            for record in records
            if record.data.get("input_ideology") in {"left", "right"}
            and nonempty_string(record.data.get("event_id"))
        ]
        by_event: dict[str, dict[str, LocatedRecord]] = defaultdict(dict)
        for record in usable:
            by_event[str(record.data["event_id"])][str(record.data["input_ideology"])] = record
        empty_titles[split] = {
            "left": sum(record.data.get("input_ideology") == "left" and record.data.get("input_title") == "" for record in usable),
            "right": sum(record.data.get("input_ideology") == "right" and record.data.get("input_title") == "" for record in usable),
            "center": sum(
                roles and next(iter(roles.values())).data.get("target_title") == "" for roles in by_event.values()
            ),
        }
        duplicate_inputs[split] = duplicate_text_groups(
            (str(record.data.get("input_article_id", "")), record.data.get("input_text")) for record in usable
        )
        duplicate_targets[split] = duplicate_text_groups(
            (
                str(next(iter(roles.values())).data.get("target_article_id", "")),
                next(iter(roles.values())).data.get("target_text"),
            )
            for roles in by_event.values()
            if roles
        )
        identical_sides[split] = sorted(
            event_id
            for event_id, roles in by_event.items()
            if {"left", "right"} <= roles.keys()
            and roles["left"].data.get("input_text") == roles["right"].data.get("input_text")
        )
        input_lengths = [len(str(record.data.get("input_text", "")).split()) for record in usable]
        target_lengths = [
            len(str(next(iter(roles.values())).data.get("target_text", "")).split())
            for roles in by_event.values()
            if roles
        ]
        result.length_statistics[split] = {
            "input_words": length_stats(input_lengths),
            "target_words": length_stats(target_lengths),
        }
        length_items = sorted(
            (
                len(str(record.data.get("input_text", "")).split()),
                str(record.data.get("input_article_id", "")),
            )
            for record in usable
        )
        extremes[split] = {
            "shortest_inputs": [{"article_id": article_id, "word_count": count} for count, article_id in length_items[:5]],
            "longest_inputs": [
                {"article_id": article_id, "word_count": count}
                for count, article_id in sorted(length_items, key=lambda item: (-item[0], item[1]))[:5]
            ],
        }
        article_ids = {
            str(record.data[field_name])
            for record in records
            for field_name in ("input_article_id", "target_article_id")
            if nonempty_string(record.data.get(field_name))
        }
        unique_counts[split] = {"events": len(by_event), "article_ids": len(article_ids)}

        if sum(empty_titles[split].values()):
            result.warnings["empty_titles"].append(f"{split}: {empty_titles[split]}")
        if duplicate_inputs[split]:
            result.warnings["duplicate_input_texts"].append(f"{split}: {duplicate_inputs[split]}")
        if duplicate_targets[split]:
            result.warnings["duplicate_target_texts"].append(f"{split}: {duplicate_targets[split]}")
        if identical_sides[split]:
            result.warnings["identical_left_right_texts"].append(f"{split}: {identical_sides[split]}")

    result.diagnostics = {
        "empty_title_counts": empty_titles,
        "duplicate_input_text_groups": duplicate_inputs,
        "duplicate_target_text_groups": duplicate_targets,
        "identical_left_right_events": identical_sides,
        "input_length_extremes": extremes,
        "unique_counts": unique_counts,
    }


def validate_data(data_dir: Path) -> ValidationResult:
    data_dir = data_dir.resolve()
    result = ValidationResult(data_dir)
    manifest = load_manifest(data_dir / "manifest.json", result)
    if manifest and manifest.get("schema") == "anchor_specific_neutral_v1":
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.scripts.prepare_side_neutral import validate_prepared
        try:
            result.split_summaries = validate_prepared(data_dir)
            if result.split_summaries != manifest.get("counts"):
                result.errors.append("Manifest counts differ from prepared records")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result.errors.append(str(exc))
        return result
    baselines_by_split: dict[str, list[LocatedRecord]] = {}
    triplets_by_split: dict[str, list[LocatedRecord]] = {}
    baseline_indexes: dict[str, dict[str, dict[str, LocatedRecord]]] = {}

    for split in SPLITS:
        baselines = load_jsonl(data_dir / split / "baseline.jsonl", result)
        triplets = load_jsonl(data_dir / split / "triplets.jsonl", result)
        baselines_by_split[split] = baselines
        triplets_by_split[split] = triplets
        baseline_indexes[split] = validate_baselines(split, baselines, result)
        validate_triplets(split, triplets, baseline_indexes[split], result)
        result.split_summaries[split] = {
            "events": len(baseline_indexes[split]),
            "baseline_records": len(baselines),
            "triplet_records": len(triplets),
            "unique_article_ids": len(
                {
                    str(record.data[field_name])
                    for record in baselines
                    for field_name in ("input_article_id", "target_article_id")
                    if nonempty_string(record.data.get(field_name))
                }
            ),
        }

    validate_leakage(baselines_by_split, triplets_by_split, result)
    validate_manifest(manifest, baselines_by_split, triplets_by_split, result)
    collect_diagnostics(baselines_by_split, result)
    return result


def print_result(result: ValidationResult) -> None:
    print("FILES")
    for path, details in sorted(result.files.items()):
        print(f"  {path}: {details['size_bytes']} bytes, {details['record_count']} records")
    print("\nSUMMARY")
    for split in SPLITS:
        print(f"  {split}: {result.split_summaries.get(split, {})}")
    print("\nLENGTH STATISTICS")
    for split in SPLITS:
        print(f"  {split}: {result.length_statistics.get(split, {})}")
    print("\nLEAKAGE COUNTS")
    for category, counts in sorted(result.leakage.items()):
        print(f"  {category}: {counts}")
    print("\nCONTENT DIAGNOSTICS")
    print(json.dumps(result.diagnostics, indent=2, sort_keys=True))
    print("\nWARNINGS")
    if result.warnings:
        for category, messages in sorted(result.warnings.items()):
            print(f"  {category}:")
            for message in messages:
                print(f"    - {message}")
    else:
        print("  none")
    print("\nERRORS")
    if result.errors:
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("  none")
    print(f"\n{'VALIDATION PASSED' if result.passed else 'VALIDATION FAILED'}")
    print(f"Total error count: {len(result.errors)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_data(args.data_dir)
    print_result(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
