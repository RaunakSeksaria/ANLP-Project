from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from src.data.baseline_dataset import (
    LEDBaselineCollator,
    LEDBaselineDataset,
    create_baseline_dataloaders,
)


class FakeTokenizer:
    sep_token = "<sep>"
    pad_token_id = 99
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool, int | None, bool]] = []

    @staticmethod
    def token_id(token: str) -> int:
        return 100 + sum((index + 1) * ord(character) for index, character in enumerate(token))

    def __call__(
        self,
        text: str | None = None,
        *,
        text_target: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        is_target = text_target is not None
        selected = text_target if is_target else text
        if selected is None:
            raise TypeError("text or text_target is required")
        self.calls.append(("target" if is_target else "source", selected, truncation, max_length, padding))
        content = [self.token_id(token) for token in selected.split()]
        if truncation and max_length is not None:
            content = content[: max(0, max_length - 2)]
        ids = [self.bos_token_id, *content, self.eos_token_id]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            if ids:
                ids[-1] = self.eos_token_id
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def make_record(index: int, split: str = "train", *, title: str = "Headline") -> dict[str, str]:
    return {
        "example_id": f"example-{split}-{index}",
        "split": split,
        "event_id": f"event-{split}-{index}",
        "input_article_id": f"input-{split}-{index}",
        "input_ideology": "left" if index % 2 == 0 else "right",
        "input_title": title,
        "input_text": f"body token {index}",
        "target_article_id": f"target-{split}-{index}",
        "target_title": f"target title {index}",
        "target_text": f"target body {index}",
    }


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class BaselineDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tokenizer = FakeTokenizer()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def dataset(
        self,
        records: list[dict[str, str]],
        *,
        split: str = "train",
        max_input_length: int = 256,
        max_target_length: int = 256,
    ) -> LEDBaselineDataset:
        path = self.root / split / "baseline.jsonl"
        write_records(path, records)
        return LEDBaselineDataset(
            path,
            self.tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )

    def test_source_formatting_and_empty_title(self) -> None:
        titled = make_record(0)
        titled["input_title"] = "  A title  "
        titled["input_text"] = "  article body  "
        untitled = make_record(1, title="   ")
        untitled["input_text"] = "  body only  "
        dataset = self.dataset([titled, untitled])

        dataset[0]
        dataset[1]
        source_calls = [call[1] for call in self.tokenizer.calls if call[0] == "source"]
        self.assertEqual(source_calls, ["A title <sep> article body", "body only"])

    def test_target_uses_target_text_only(self) -> None:
        record = make_record(0)
        record["target_title"] = "FORBIDDEN TARGET TITLE"
        record["target_text"] = "  decoder body only  "
        dataset = self.dataset([record])
        dataset[0]
        target_calls = [call[1] for call in self.tokenizer.calls if call[0] == "target"]
        self.assertEqual(target_calls, ["decoder body only"])
        self.assertNotIn("FORBIDDEN", target_calls[0])

    def test_input_and_target_truncation(self) -> None:
        record = make_record(0, title="")
        record["input_text"] = "one two three four five six"
        record["target_text"] = "alpha beta gamma delta epsilon"
        dataset = self.dataset(
            [record], max_input_length=5, max_target_length=4
        )
        example = dataset[0]
        self.assertEqual(len(example["input_ids"]), 5)
        self.assertEqual(len(example["labels"]), 4)
        self.assertEqual(example["input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(example["labels"][-1], self.tokenizer.eos_token_id)

    def test_global_attention_before_padding(self) -> None:
        example = self.dataset([make_record(0)])[0]
        self.assertEqual(len(example["global_attention_mask"]), len(example["input_ids"]))
        self.assertEqual(example["global_attention_mask"][0], 1)
        self.assertTrue(all(value == 0 for value in example["global_attention_mask"][1:]))

    def test_dynamic_padding_masks_and_label_ignore_index(self) -> None:
        short = make_record(0, title="")
        short["input_text"] = "short"
        short["target_text"] = "brief"
        long = make_record(1, title="")
        long["input_text"] = "a somewhat longer input"
        long["target_text"] = "a longer target"
        dataset = self.dataset([short, long])
        examples = [dataset[0], dataset[1]]
        batch = LEDBaselineCollator(self.tokenizer)(examples)

        expected_input_length = max(len(example["input_ids"]) for example in examples)
        expected_target_length = max(len(example["labels"]) for example in examples)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, expected_input_length))
        self.assertEqual(tuple(batch["labels"].shape), (2, expected_target_length))
        self.assertLess(expected_input_length, 256)
        short_input_length = len(examples[0]["input_ids"])
        short_target_length = len(examples[0]["labels"])
        self.assertTrue(torch.all(batch["input_ids"][0, short_input_length:] == self.tokenizer.pad_token_id))
        self.assertTrue(torch.all(batch["attention_mask"][0, :short_input_length] == 1))
        self.assertTrue(torch.all(batch["attention_mask"][0, short_input_length:] == 0))
        self.assertTrue(torch.all(batch["global_attention_mask"][0, short_input_length:] == 0))
        self.assertTrue(torch.equal(batch["labels"][0, :short_target_length], torch.tensor(examples[0]["labels"])))
        self.assertTrue(torch.all(batch["labels"][0, short_target_length:] == -100))
        self.assertFalse(torch.any(batch["labels"][1] == -100))

    def test_malformed_json_has_path_and_line(self) -> None:
        path = self.root / "train" / "baseline.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(make_record(0)) + "\n{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, rf"{path}:2: malformed JSON"):
            LEDBaselineDataset(path, self.tokenizer)

    def test_missing_required_field_is_rejected(self) -> None:
        record = make_record(0)
        del record["target_text"]
        with self.assertRaisesRegex(ValueError, r":1: missing required fields: target_text"):
            self.dataset([record])

    def test_empty_input_or_target_is_rejected(self) -> None:
        for field_name in ("input_text", "target_text"):
            with self.subTest(field_name=field_name):
                record = make_record(0)
                record[field_name] = "  "
                with self.assertRaisesRegex(ValueError, rf":1: {field_name} must be non-empty"):
                    self.dataset([record])

    def test_split_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"split 'validation'.*parent directory 'train'"):
            self.dataset([make_record(0, split="validation")], split="train")

    def test_file_order_is_preserved(self) -> None:
        records = [make_record(index, title="") for index in range(4)]
        dataset = self.dataset(records)
        observed = [dataset[index]["input_ids"][1] for index in range(len(dataset))]
        expected = [self.tokenizer.token_id("body")] * 4
        self.assertEqual(observed, expected)
        observed_unique = [dataset[index]["input_ids"][3] for index in range(len(dataset))]
        self.assertEqual(observed_unique, [self.tokenizer.token_id(str(index)) for index in range(4)])

    def _factory_data(self) -> None:
        for split in ("train", "validation", "test"):
            write_records(
                self.root / split / "baseline.jsonl",
                [make_record(index, split=split, title="") for index in range(6)],
            )

    @staticmethod
    def loader_order(loader: Any) -> list[int]:
        order: list[int] = []
        for batch in loader:
            order.extend(batch["input_ids"][:, 3].tolist())
        return order

    def test_training_shuffle_is_reproducible(self) -> None:
        self._factory_data()
        first = create_baseline_dataloaders(self.root, self.tokenizer, 2, 2, seed=73)
        second = create_baseline_dataloaders(self.root, self.tokenizer, 2, 2, seed=73)
        first_order = self.loader_order(first.dataloaders["train"])
        second_order = self.loader_order(second.dataloaders["train"])
        self.assertEqual(first_order, second_order)
        sequential = [self.tokenizer.token_id(str(index)) for index in range(6)]
        self.assertNotEqual(first_order, sequential)

    def test_validation_and_test_are_stable_and_unshuffled(self) -> None:
        self._factory_data()
        bundle = create_baseline_dataloaders(self.root, self.tokenizer, 2, 2, seed=42)
        expected = [self.tokenizer.token_id(str(index)) for index in range(6)]
        for split in ("validation", "test"):
            with self.subTest(split=split):
                self.assertEqual(self.loader_order(bundle.dataloaders[split]), expected)


if __name__ == "__main__":
    unittest.main()
