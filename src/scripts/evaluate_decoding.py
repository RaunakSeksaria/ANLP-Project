#!/usr/bin/env python3
"""Evaluate an existing baseline or triplet checkpoint with greedy or beam decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.baseline_dataset import LEDBaselineCollator, LEDBaselineDataset  # noqa: E402
from src.models.led_baseline import load_led_baseline  # noqa: E402
from src.training.decoding import decoding_settings  # noqa: E402
from src.training.overfit_baseline import prepare_output_directory, seed_everything  # noqa: E402
from src.training.train_baseline import (  # noqa: E402
    GENERATION_DEFAULTS, atomic_write_json, evaluate_split,
    write_prediction_jsonl,
)
from src.training.train_joint import (  # noqa: E402
    GENERATION_SETTINGS, evaluate_joint_checkpoint, resolve_device,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-type", choices=("baseline", "triplet"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--decoding-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-entity", default="manasagrawal206-iiit-hyderabad")
    parser.add_argument("--wandb-project", default="anlp-project")
    parser.add_argument("--wandb-run-name")
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--disable-wandb", action="store_true")
    wandb_group.add_argument("--require-wandb", action="store_true")
    return parser.parse_args(argv)


def evaluate_baseline_checkpoint(args: argparse.Namespace) -> dict:
    if args.attention_window != 512:
        raise ValueError("baseline checkpoints use attention_window=512")
    prepare_output_directory(args.output_dir, overwrite=False)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset = LEDBaselineDataset(
        args.data_dir / args.split / "baseline.jsonl",
        tokenizer, args.max_input_length, args.max_target_length,
    )
    loader = DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=LEDBaselineCollator(tokenizer), drop_last=False,
    )
    model = load_led_baseline(args.checkpoint).to(device)
    settings = decoding_settings(
        {**GENERATION_DEFAULTS, "max_new_tokens": args.generation_max_new_tokens},
        args.decoding_strategy,
    )
    result = evaluate_split(
        model, dataset, loader, tokenizer, device, "fp32", settings,
        progress_description=f"{args.split} · {args.decoding_strategy}",
    )
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "teacher_forced_loss": result.teacher_forced_loss,
        "token_accuracy": result.token_accuracy,
        "token_overlap_f1": result.token_overlap_f1,
        "exact_match_count": result.exact_match_count,
        "exact_match_rate": result.exact_match_rate,
        "examples": result.examples,
        "generation_settings": settings,
    }
    write_prediction_jsonl(args.output_dir / f"{args.split}_predictions.jsonl", result.predictions)
    atomic_write_json(args.output_dir / f"{args.split}_metrics.json", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eval_batch_size <= 0 or min(
        args.max_input_length, args.max_target_length, args.generation_max_new_tokens
    ) <= 0:
        raise ValueError("batch size and sequence limits must be positive")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    seed_everything(args.seed)
    if args.checkpoint_type == "triplet":
        metrics = evaluate_joint_checkpoint(
            data_dir=args.data_dir, checkpoint=args.checkpoint,
            output_dir=args.output_dir, split=args.split,
            eval_batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            generation_max_new_tokens=args.generation_max_new_tokens,
            decoding_strategy=args.decoding_strategy,
            attention_window=args.attention_window,
            device_name=args.device,
            overwrite_output_dir=False,
        )
    else:
        metrics = evaluate_baseline_checkpoint(args)
    metrics.update({
        "checkpoint_type": args.checkpoint_type,
        "decoding_strategy": args.decoding_strategy,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "attention_window": args.attention_window,
        "seed": args.seed,
    })
    atomic_write_json(args.output_dir / f"{args.split}_metrics.json", metrics)
    if args.require_wandb:
        import wandb
        run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            name=args.wandb_run_name,
            job_type="decoding-evaluation",
            config={key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "disable_wandb"},
            dir=str(args.output_dir), resume="never",
        )
        try:
            run.log({f"{args.split}/{key}": value for key, value in metrics.items()
                     if isinstance(value, (int, float)) and not isinstance(value, bool)})
            run.summary.update(metrics)
        finally:
            run.finish()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
