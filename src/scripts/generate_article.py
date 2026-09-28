#!/usr/bin/env python3
"""Generate neutralized text for one article with one or more joint checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.led_triplet import load_led_triplet  # noqa: E402
from src.training.overfit_baseline import generate_with_cache  # noqa: E402
from src.training.train_joint import resolve_device  # noqa: E402


DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "outputs/joint_mean_lambda01/best",
    PROJECT_ROOT / "outputs/joint_control_lambda0/best",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    article_group = parser.add_mutually_exclusive_group(required=True)
    article_group.add_argument("--article", help="Article body supplied directly")
    article_group.add_argument("--article-file", type=Path, help="UTF-8 article text file")
    parser.add_argument("--title", default="", help="Optional article title")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        help="Checkpoint directory; repeat to compare models (defaults to lambda 0.1 and 0)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    state_path = checkpoint / "joint_state.json"
    if not state_path.is_file():
        return {}
    with state_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{state_path} must contain a JSON object")
    return value


def generate_one(
    checkpoint: Path,
    article: str,
    title: str,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    attention_window: int,
) -> dict[str, Any]:
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"checkpoint model not found: {checkpoint}")
    metadata = checkpoint_metadata(checkpoint)
    saved_window = metadata.get("attention_window")
    if saved_window is not None and saved_window != attention_window:
        raise ValueError(
            f"checkpoint attention_window={saved_window}, requested {attention_window}"
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    source_text = article if not title else f"{title.strip()} {tokenizer.sep_token} {article}"
    encoded = tokenizer(
        source_text,
        truncation=True,
        max_length=max_input_length,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
    }

    model = load_led_triplet(checkpoint, attention_window=attention_window).to(device)
    model.eval()
    settings: dict[str, int | bool] = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "early_stopping": False,
    }
    with torch.inference_mode():
        generated = generate_with_cache(model.led, batch, device, "fp32", settings)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "triplet_lambda": metadata.get("triplet_lambda"),
        "checkpoint_epoch": metadata.get("epoch"),
        "input_tokens": int(input_ids.shape[1]),
        "generated_text": generated_text,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    article = (
        args.article_file.read_text(encoding="utf-8")
        if args.article_file is not None
        else args.article
    )
    article = article.strip()
    if not article:
        raise ValueError("article text must not be empty")
    if args.max_input_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")

    checkpoints = args.checkpoint or list(DEFAULT_CHECKPOINTS)
    device = resolve_device(args.device)
    document = {
        "title": args.title.strip(),
        "article": article,
        "device": str(device),
        "generation_settings": {
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "attention_window": args.attention_window,
            "num_beams": 1,
            "do_sample": False,
        },
        "outputs": [
            generate_one(
                checkpoint,
                article,
                args.title,
                device,
                args.max_input_length,
                args.max_new_tokens,
                args.attention_window,
            )
            for checkpoint in checkpoints
        ],
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote generations to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
