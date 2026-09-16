from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from src.scripts.validate_prepared_data import validate_data


SPLITS = ("train", "validation", "test")


def baseline(split: str, event: str, role: str) -> dict[str, str]:
    return {
        "example_id": f"{event}:{role}-to-center",
        "split": split,
        "event_id": event,
        "input_article_id": f"{event}:{role}",
        "input_ideology": role,
        "input_title": f"{event} {role} title",
        "input_text": f"Distinct body for {event} {role}",
        "target_article_id": f"{event}:center",
        "target_title": f"{event} center title",
        "target_text": f"Neutral target body for {event}",
    }


def triplet(split: str, event: str, role: str, negative_event: str) -> dict[str, str]:
    opposite = "right" if role == "left" else "left"
    return {
        "triplet_id": f"{event}:{role}-anchor",
        "split": split,
        "anchor_event_id": event,
        "anchor_article_id": f"{event}:{role}",
        "anchor_ideology": role,
        "anchor_title": f"{event} {role} title",
        "anchor_text": f"Distinct body for {event} {role}",
        "positive_event_id": event,
        "positive_article_id": f"{event}:{opposite}",
        "positive_ideology": opposite,
        "positive_title": f"{event} {opposite} title",
        "positive_text": f"Distinct body for {event} {opposite}",
        "negative_event_id": negative_event,
        "negative_article_id": f"{negative_event}:{role}",
        "negative_ideology": role,
        "negative_title": f"{negative_event} {role} title",
        "negative_text": f"Distinct body for {negative_event} {role}",
        "target_article_id": f"{event}:center",
        "target_title": f"{event} center title",
        "target_text": f"Neutral target body for {event}",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def build_valid_fixture(root: Path) -> None:
    anchor_counts: dict[str, dict[str, int]] = {}
    baseline_counts: dict[str, int] = {}
    triplet_counts: dict[str, int] = {}
    for split in SPLITS:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        events = (f"{split}-a", f"{split}-b")
        baselines = [baseline(split, event, role) for event in events for role in ("left", "right")]
        triplets = [
            triplet(split, event, role, events[1] if event == events[0] else events[0])
            for event in events
            for role in ("left", "right")
        ]
        write_jsonl(split_dir / "baseline.jsonl", baselines)
        write_jsonl(split_dir / "triplets.jsonl", triplets)
        baseline_counts[split] = len(baselines)
        triplet_counts[split] = len(triplets)
        anchor_counts[split] = {"left": 2, "right": 2}
    manifest = {
        "baseline_records_per_split": baseline_counts,
        "triplet_records_per_split": triplet_counts,
        "triplet_anchor_counts": anchor_counts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def mutate_first(path: Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    mutator(records[0])
    write_jsonl(path, records)


class ValidatePreparedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        build_valid_fixture(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def messages(self) -> str:
        return "\n".join(validate_data(self.root).errors)

    def test_fully_valid_dataset(self) -> None:
        result = validate_data(self.root)
        self.assertTrue(result.passed, "\n".join(result.errors))

    def test_cross_split_event_leakage(self) -> None:
        def leak_event(record: dict[str, Any]) -> None:
            record["event_id"] = "train-a"

        mutate_first(self.root / "test" / "baseline.jsonl", leak_event)
        self.assertIn("Split leakage: event_ids", self.messages())

    def test_invalid_negative_ideology(self) -> None:
        mutate_first(
            self.root / "train" / "triplets.jsonl",
            lambda record: record.__setitem__("negative_ideology", "right"),
        )
        self.assertIn("anchor and negative ideologies differ", self.messages())

    def test_same_event_negative(self) -> None:
        def use_anchor_as_negative(record: dict[str, Any]) -> None:
            record["negative_event_id"] = record["anchor_event_id"]
            record["negative_article_id"] = record["anchor_article_id"]
            record["negative_title"] = record["anchor_title"]
            record["negative_text"] = record["anchor_text"]

        mutate_first(self.root / "train" / "triplets.jsonl", use_anchor_as_negative)
        self.assertIn("anchor and negative use the same event", self.messages())

    def test_missing_required_field(self) -> None:
        mutate_first(
            self.root / "train" / "triplets.jsonl", lambda record: record.pop("anchor_text")
        )
        self.assertIn("missing triplet fields: anchor_text", self.messages())

    def test_manifest_count_mismatch(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["baseline_records_per_split"]["train"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIn("Manifest baseline count mismatch for train", self.messages())

    def test_unresolved_negative_article(self) -> None:
        mutate_first(
            self.root / "train" / "triplets.jsonl",
            lambda record: record.__setitem__("negative_article_id", "unknown:right"),
        )
        self.assertIn("negative article cannot be resolved in the same split", self.messages())

    def test_inconsistent_target_between_baseline_and_triplet(self) -> None:
        mutate_first(
            self.root / "train" / "triplets.jsonl",
            lambda record: record.__setitem__("target_text", "Different target text"),
        )
        self.assertIn("target_text does not match the anchor event baseline target", self.messages())


if __name__ == "__main__":
    unittest.main()
