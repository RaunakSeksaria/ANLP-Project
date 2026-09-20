#!/usr/bin/env python3
"""Run the pretrained LED baseline eight-example overfitting check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.overfit_baseline import OverfitConfig, run_overfit_test  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overfit_baseline"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = OverfitConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    try:
        results = run_overfit_test(config)
    except Exception as exc:
        print(f"OVERFIT TEST ERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
