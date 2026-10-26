#!/usr/bin/env python3
"""Explicitly evaluate one selected joint/control checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_joint import evaluate_joint_checkpoint  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--decoding-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metrics = evaluate_joint_checkpoint(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            eval_batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            generation_max_new_tokens=args.generation_max_new_tokens,
            decoding_strategy=args.decoding_strategy,
            attention_window=args.attention_window,
            device_name=args.device,
            overwrite_output_dir=args.overwrite_output_dir,
        )
    except Exception as exc:
        print(f"JOINT EVALUATION ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
