#!/usr/bin/env python3
"""Smoke-test the baseline LED datasets and dynamic-padding DataLoaders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.baseline_dataset import (  # noqa: E402
    SPLITS,
    BaselineDataBundle,
    LEDBaselineDataset,
    PaddedBatch,
    create_baseline_dataloaders,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def target_encoding_without_truncation(tokenizer: Any, text: str) -> Mapping[str, Any]:
    options = {"truncation": False, "padding": False}
    try:
        return tokenizer(text_target=text, **options)
    except TypeError as exc:
        if "text_target" not in str(exc):
            raise
        return tokenizer(text, **options)


def complete_split_diagnostics(
    dataset: LEDBaselineDataset, tokenizer: Any
) -> dict[str, int | bool]:
    maximum_source_length = 0
    maximum_target_length = 0
    input_truncated = False
    target_truncated = False
    for index in range(len(dataset)):
        actual = dataset[index]
        full_source = tokenizer(
            dataset.source_text(index), truncation=False, padding=False
        )["input_ids"]
        full_target = target_encoding_without_truncation(
            tokenizer, dataset.target_text(index)
        )["input_ids"]
        maximum_source_length = max(maximum_source_length, len(actual["input_ids"]))
        maximum_target_length = max(maximum_target_length, len(actual["labels"]))
        input_truncated = input_truncated or len(full_source) > len(actual["input_ids"])
        target_truncated = target_truncated or len(full_target) > len(actual["labels"])
    return {
        "max_source_length": maximum_source_length,
        "max_target_length": maximum_target_length,
        "input_truncated": input_truncated,
        "target_truncated": target_truncated,
    }


def verify_batch(batch: PaddedBatch) -> None:
    expected_keys = {"input_ids", "attention_mask", "global_attention_mask", "labels"}
    assert set(batch) == expected_keys, f"Unexpected batch keys: {set(batch)}"
    assert all(tensor.dtype == torch.long for tensor in batch.values())
    encoder_shape = batch["input_ids"].shape
    assert batch["attention_mask"].shape == encoder_shape
    assert batch["global_attention_mask"].shape == encoder_shape
    assert batch["labels"].ndim == 2
    assert batch["labels"].shape[0] == encoder_shape[0]
    assert torch.all(batch["attention_mask"] >= 0)
    assert torch.all(batch["attention_mask"] <= 1)
    assert torch.all(batch["global_attention_mask"].sum(dim=1) == 1)
    assert torch.all(batch["global_attention_mask"][:, 0] == 1)
    assert torch.all(
        batch["global_attention_mask"][batch["attention_mask"] == 0] == 0
    )
    for row in range(batch["labels"].shape[0]):
        ignored = batch["labels"][row] == -100
        if torch.any(ignored):
            first_ignored = int(torch.nonzero(ignored, as_tuple=False)[0].item())
            assert torch.all(ignored[first_ignored:]), "Label padding must be a contiguous suffix"


def preview(text: str, limit: int = 300) -> str:
    safe = " ".join(text.split())
    return safe if len(safe) <= limit else safe[: limit - 1] + "…"


def print_batch(split: str, batch: PaddedBatch) -> None:
    print(f"{split} batch:")
    for name in ("input_ids", "attention_mask", "global_attention_mask", "labels"):
        tensor = batch[name]
        print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
    print(f"  non-padding input tokens: {batch['attention_mask'].sum(dim=1).tolist()}")
    print(f"  non-ignored target tokens: {(batch['labels'] != -100).sum(dim=1).tolist()}")
    positions = [
        torch.nonzero(row, as_tuple=False).flatten().tolist()
        for row in batch["global_attention_mask"]
    ]
    print(f"  global-attention positions: {positions}")


def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bundle: BaselineDataBundle = create_baseline_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    batches: dict[str, PaddedBatch] = {}
    diagnostics: dict[str, dict[str, int | bool]] = {}
    for split in SPLITS:
        print(f"{split} records: {len(bundle.datasets[split])}")
        try:
            batch = next(iter(bundle.dataloaders[split]))
        except StopIteration as exc:
            raise AssertionError(f"{split} DataLoader is empty") from exc
        verify_batch(batch)
        batches[split] = batch
        print_batch(split, batch)
        diagnostics[split] = complete_split_diagnostics(bundle.datasets[split], tokenizer)
        print(
            f"  complete split: max source={diagnostics[split]['max_source_length']}, "
            f"max target={diagnostics[split]['max_target_length']}, "
            f"input truncated={diagnostics[split]['input_truncated']}, "
            f"target truncated={diagnostics[split]['target_truncated']}"
        )

    train_batch = batches["train"]
    source_ids = train_batch["input_ids"][0][train_batch["attention_mask"][0].bool()]
    labels_for_decode = train_batch["labels"][0].clone()
    labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
    print("training source preview:")
    print(f"  {preview(tokenizer.decode(source_ids, skip_special_tokens=False))}")
    print("training target preview:")
    print(f"  {preview(tokenizer.decode(labels_for_decode, skip_special_tokens=False))}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
