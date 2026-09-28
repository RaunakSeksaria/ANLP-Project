#!/usr/bin/env python3
"""Create a side-by-side JSON comparison of two aligned validation decodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAMBDA01 = (
    PROJECT_ROOT
    / "outputs/joint_mean_lambda01/validation_generation_subset_predictions_epoch_002.jsonl"
)
DEFAULT_LAMBDA0 = (
    PROJECT_ROOT
    / "outputs/joint_control_lambda0/validation_generation_subset_predictions_epoch_002.jsonl"
)
DEFAULT_LAMBDA035 = (
    PROJECT_ROOT
    / "outputs/joint_mean_lambda035/validation_generation_subset_predictions_epoch_002.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/validation_neutrality_comparison_10.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda01-predictions", type=Path, default=DEFAULT_LAMBDA01)
    parser.add_argument("--lambda0-predictions", type=Path, default=DEFAULT_LAMBDA0)
    parser.add_argument("--lambda035-predictions", type=Path, default=DEFAULT_LAMBDA035)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--generated-text-only",
        action="store_true",
        help="Write only lambda 0, 0.1, and 0.35 generated text for each example",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def build_comparison(
    lambda01_rows: list[dict[str, Any]],
    lambda0_rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(lambda01_rows) != len(lambda0_rows):
        raise ValueError("prediction files contain different numbers of examples")
    if count > len(lambda01_rows):
        raise ValueError(f"requested {count} examples, but only {len(lambda01_rows)} exist")

    comparisons: list[dict[str, Any]] = []
    for lambda01, lambda0 in zip(lambda01_rows[:count], lambda0_rows[:count]):
        identity_fields = (
            "example_id",
            "event_id",
            "input_article_id",
            "input_ideology",
            "target_article_id",
            "source_text",
            "reference_text",
        )
        for field in identity_fields:
            if lambda01.get(field) != lambda0.get(field):
                raise ValueError(
                    f"unaligned prediction field {field!r} for "
                    f"{lambda01.get('example_id')!r} and {lambda0.get('example_id')!r}"
                )
        comparisons.append(
            {
                **{field: lambda01[field] for field in identity_fields},
                "lambda_0_1": {
                    "generated_text": lambda01["generated_text"],
                    "token_overlap_f1": lambda01["token_overlap_f1"],
                    "word_length_ratio": lambda01["word_length_ratio"],
                    "repeated_trigram_fraction": lambda01[
                        "repeated_trigram_fraction"
                    ],
                    "terminated_with_eos": lambda01["terminated_with_eos"],
                },
                "lambda_0": {
                    "generated_text": lambda0["generated_text"],
                    "token_overlap_f1": lambda0["token_overlap_f1"],
                    "word_length_ratio": lambda0["word_length_ratio"],
                    "repeated_trigram_fraction": lambda0[
                        "repeated_trigram_fraction"
                    ],
                    "terminated_with_eos": lambda0["terminated_with_eos"],
                },
            }
        )
    return comparisons


def build_text_only_comparison(
    lambda01_rows: list[dict[str, Any]],
    lambda0_rows: list[dict[str, Any]],
    lambda035_rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, str]]:
    if count <= 0:
        raise ValueError("count must be positive")
    lengths = {len(lambda01_rows), len(lambda0_rows), len(lambda035_rows)}
    if len(lengths) != 1:
        raise ValueError("prediction files contain different numbers of examples")
    if count > len(lambda01_rows):
        raise ValueError(f"requested {count} examples, but only {len(lambda01_rows)} exist")

    comparisons: list[dict[str, str]] = []
    for lambda01, lambda0, lambda035 in zip(
        lambda01_rows[:count], lambda0_rows[:count], lambda035_rows[:count]
    ):
        example_ids = {
            lambda01.get("example_id"),
            lambda0.get("example_id"),
            lambda035.get("example_id"),
        }
        source_texts = {
            lambda01.get("source_text"),
            lambda0.get("source_text"),
            lambda035.get("source_text"),
        }
        if len(example_ids) != 1 or len(source_texts) != 1:
            raise ValueError("prediction files are not aligned to identical inputs")
        comparisons.append(
            {
                "lambda_0": lambda0["generated_text"],
                "lambda_0_1": lambda01["generated_text"],
                "lambda_0_35": lambda035["generated_text"],
            }
        )
    return comparisons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lambda01_rows = read_jsonl(args.lambda01_predictions)
    lambda0_rows = read_jsonl(args.lambda0_predictions)
    if args.generated_text_only:
        document = build_text_only_comparison(
            lambda01_rows,
            lambda0_rows,
            read_jsonl(args.lambda035_predictions),
            args.count,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote {len(document)} aligned text-only examples to {args.output}")
        return 0

    comparisons = build_comparison(
        lambda01_rows,
        lambda0_rows,
        args.count,
    )
    document = {
        "description": (
            "Aligned validation generations from each run's selected best checkpoint "
            "(epoch 2); outputs are presented for qualitative neutrality comparison."
        ),
        "example_count": len(comparisons),
        "lambda_0_1_predictions": str(args.lambda01_predictions.resolve()),
        "lambda_0_predictions": str(args.lambda0_predictions.resolve()),
        "examples": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(comparisons)} aligned examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
