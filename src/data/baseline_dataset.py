"""Lazy baseline tokenization and dynamic padding for LED."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_FIELDS = frozenset(
    {
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
)
SPLITS = ("train", "validation", "test")


class TokenizerLike(Protocol):
    """Minimal tokenizer interface used by the dataset and collator."""

    sep_token: str | None
    pad_token_id: int | None

    def __call__(self, text: str | None = None, **kwargs: Any) -> Mapping[str, Any]: ...


class UnpaddedExample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    global_attention_mask: list[int]
    labels: list[int]


class PaddedBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    global_attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class BaselineRecord:
    example_id: str
    split: str
    event_id: str
    input_article_id: str
    input_ideology: str
    input_title: str
    input_text: str
    target_article_id: str
    target_title: str
    target_text: str
    event_ids: tuple[str, ...] = ()


class LEDBaselineDataset(Dataset[UnpaddedExample]):
    """Validated baseline records with lazy source and target tokenization."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: TokenizerLike,
        max_input_length: int = 256,
        max_target_length: int = 256,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        if max_input_length <= 0 or max_target_length <= 0:
            raise ValueError("Maximum input and target lengths must be positive")
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Baseline JSONL file not found: {self.jsonl_path}")
        if not tokenizer.sep_token:
            raise ValueError("Tokenizer must define sep_token")
        self._records = self._load_records()

    def _error(self, line_number: int, message: str) -> ValueError:
        return ValueError(f"{self.jsonl_path}:{line_number}: {message}")

    def _require_string(
        self, record: Mapping[str, Any], field_name: str, line_number: int
    ) -> str:
        value = record[field_name]
        if not isinstance(value, str):
            raise self._error(line_number, f"field {field_name!r} must be a string")
        return value

    def _parse_record(self, value: Any, line_number: int) -> BaselineRecord:
        if not isinstance(value, dict):
            raise self._error(line_number, "record must be a JSON object")
        missing = sorted(REQUIRED_FIELDS - value.keys())
        if missing:
            raise self._error(line_number, f"missing required fields: {', '.join(missing)}")

        strings = {
            field_name: self._require_string(value, field_name, line_number)
            for field_name in REQUIRED_FIELDS
        }
        input_title = strings["input_title"].strip()
        input_text = strings["input_text"].strip()
        target_title = strings["target_title"].strip()
        target_text = strings["target_text"].strip()
        if not input_text:
            raise self._error(line_number, "input_text must be non-empty after stripping")
        if not target_text:
            raise self._error(line_number, "target_text must be non-empty after stripping")
        expected_split = self.jsonl_path.parent.name
        if strings["split"] != expected_split:
            raise self._error(
                line_number,
                f"record split {strings['split']!r} does not match parent directory {expected_split!r}",
            )
        event_ids = value.get("event_ids", [])
        if not isinstance(event_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in event_ids
        ):
            raise self._error(line_number, "event_ids must be a list of non-empty strings")
        return BaselineRecord(
            example_id=strings["example_id"],
            split=strings["split"],
            event_id=strings["event_id"],
            input_article_id=strings["input_article_id"],
            input_ideology=strings["input_ideology"],
            input_title=input_title,
            input_text=input_text,
            target_article_id=strings["target_article_id"],
            target_title=target_title,
            target_text=target_text,
            event_ids=tuple(event_ids),
        )

    def _load_records(self) -> list[BaselineRecord]:
        records: list[BaselineRecord] = []
        try:
            with self.jsonl_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise self._error(line_number, "blank lines are not allowed")
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise self._error(line_number, f"malformed JSON: {exc.msg}") from exc
                    records.append(self._parse_record(value, line_number))
        except UnicodeError as exc:
            raise ValueError(f"{self.jsonl_path}: invalid UTF-8: {exc}") from exc
        return records

    def __len__(self) -> int:
        return len(self._records)

    def source_text(self, index: int) -> str:
        """Return the exact formatted encoder source for diagnostics."""
        record = self._records[index]
        if not record.input_title:
            return record.input_text
        return f"{record.input_title} {self.tokenizer.sep_token} {record.input_text}"

    def target_text(self, index: int) -> str:
        """Return the stripped decoder target for diagnostics."""
        return self._records[index].target_text

    def record(self, index: int) -> BaselineRecord:
        """Return validated metadata without duplicating JSONL loading logic."""
        return self._records[index]

    def _tokenize_target(self, text: str) -> Mapping[str, Any]:
        options = {
            "truncation": True,
            "max_length": self.max_target_length,
            "padding": False,
        }
        try:
            return self.tokenizer(text_target=text, **options)
        except TypeError as exc:
            if "text_target" not in str(exc):
                raise
            return self.tokenizer(text, **options)

    @staticmethod
    def _integer_list(value: Any, field_name: str) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Tokenizer output {field_name!r} must be a sequence")
        result = [int(item) for item in value]
        if not result:
            raise ValueError(f"Tokenizer returned an empty {field_name}")
        return result

    def __getitem__(self, index: int) -> UnpaddedExample:
        source_encoding = self.tokenizer(
            self.source_text(index),
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        )
        target_encoding = self._tokenize_target(self.target_text(index))
        input_ids = self._integer_list(source_encoding.get("input_ids"), "input_ids")
        attention_mask = self._integer_list(
            source_encoding.get("attention_mask"), "attention_mask"
        )
        labels = self._integer_list(target_encoding.get("input_ids"), "target input_ids")
        if len(input_ids) != len(attention_mask):
            raise ValueError("Tokenizer returned input_ids and attention_mask of different lengths")
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "labels": labels,
        }


class LEDBaselineCollator:
    """Dynamically pad LED baseline examples to the longest batch members."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, examples: Sequence[UnpaddedExample]) -> PaddedBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_input_length = max(len(example["input_ids"]) for example in examples)
        max_target_length = max(len(example["labels"]) for example in examples)
        if max_input_length == 0 or max_target_length == 0:
            raise ValueError("Cannot collate empty token sequences")

        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        global_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        input_lengths: list[int] = []
        target_lengths: list[int] = []
        for example in examples:
            input_length = len(example["input_ids"])
            target_length = len(example["labels"])
            if len(example["attention_mask"]) != input_length:
                raise ValueError("input_ids and attention_mask lengths differ")
            if len(example["global_attention_mask"]) != input_length:
                raise ValueError("input_ids and global_attention_mask lengths differ")
            input_padding = max_input_length - input_length
            target_padding = max_target_length - target_length
            input_rows.append(example["input_ids"] + [self.pad_token_id] * input_padding)
            attention_rows.append(example["attention_mask"] + [0] * input_padding)
            global_rows.append(example["global_attention_mask"] + [0] * input_padding)
            label_rows.append(example["labels"] + [-100] * target_padding)
            input_lengths.append(input_length)
            target_lengths.append(target_length)

        batch: PaddedBatch = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "global_attention_mask": torch.tensor(global_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }
        batch_size = len(examples)
        encoder_shape = (batch_size, max_input_length)
        assert tuple(batch["input_ids"].shape) == encoder_shape
        assert tuple(batch["attention_mask"].shape) == encoder_shape
        assert tuple(batch["global_attention_mask"].shape) == encoder_shape
        assert tuple(batch["labels"].shape) == (batch_size, max_target_length)
        for row, input_length in enumerate(input_lengths):
            assert torch.all(batch["attention_mask"][row, input_length:] == 0)
            assert torch.all(batch["global_attention_mask"][row, input_length:] == 0)
        assert torch.all(batch["global_attention_mask"].sum(dim=1) == 1)
        assert torch.all(batch["global_attention_mask"][:, 0] == 1)
        return batch


@dataclass(frozen=True)
class BaselineDataBundle:
    """Datasets and corresponding DataLoaders keyed by split name."""

    datasets: dict[str, LEDBaselineDataset]
    dataloaders: dict[str, DataLoader[PaddedBatch]]


def create_baseline_dataloaders(
    data_dir: str | Path,
    tokenizer: TokenizerLike,
    train_batch_size: int,
    eval_batch_size: int,
    max_input_length: int = 256,
    max_target_length: int = 256,
    seed: int = 42,
    num_workers: int = 0,
) -> BaselineDataBundle:
    """Build baseline datasets and reproducible split DataLoaders."""
    data_path = Path(data_dir)
    if train_batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    datasets = {
        split: LEDBaselineDataset(
            data_path / split / "baseline.jsonl",
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        for split in SPLITS
    }
    collator = LEDBaselineCollator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloaders: dict[str, DataLoader[PaddedBatch]] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=generator,
            num_workers=num_workers,
            drop_last=False,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            drop_last=False,
        ),
    }
    return BaselineDataBundle(datasets=datasets, dataloaders=dataloaders)
