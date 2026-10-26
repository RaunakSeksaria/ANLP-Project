#!/usr/bin/env python3
"""Evaluate held-out cross-ideology event retrieval for a first-token checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.baseline_dataset import LEDBaselineCollator, LEDBaselineDataset  # noqa: E402
from src.models.led_triplet import load_led_triplet  # noqa: E402
from src.training.train_baseline import atomic_write_json  # noqa: E402


def retrieval_scores(
    embeddings: np.ndarray, event_ids: list[str], ideologies: list[str]
) -> dict:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(event_ids) or len(event_ids) != len(ideologies):
        raise ValueError("embeddings and article metadata must align")
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("article embeddings must be finite and L2-normalized")
    if set(ideologies) != {"left", "right"}:
        raise ValueError("retrieval requires both left and right articles")

    def direction(query_side: str, candidate_side: str) -> dict:
        queries = np.flatnonzero(np.asarray(ideologies) == query_side)
        candidates = np.flatnonzero(np.asarray(ideologies) == candidate_side)
        scores = embeddings[queries] @ embeddings[candidates].T
        candidate_events = np.asarray(event_ids)[candidates]
        ranks = []
        for index, query in enumerate(queries):
            relevant = candidate_events == event_ids[query]
            if not relevant.any():
                raise ValueError(f"query event {event_ids[query]} has no opposite-ideology match")
            order = np.argsort(-scores[index], kind="stable")
            ranks.append(int(np.flatnonzero(relevant[order])[0]) + 1)
        ranks_array = np.asarray(ranks)
        return {
            "queries": len(queries), "candidates": len(candidates),
            "recall_at_1": float(np.mean(ranks_array <= 1)),
            "recall_at_5": float(np.mean(ranks_array <= 5)),
            "recall_at_10": float(np.mean(ranks_array <= 10)),
            "mrr": float(np.mean(1.0 / ranks_array)),
        }

    return {
        "left_to_right": direction("left", "right"),
        "right_to_left": direction("right", "left"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wandb-entity", default="manasagrawal206-iiit-hyderabad")
    parser.add_argument("--wandb-project", default="anlp-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.output.exists():
        raise ValueError("batch size must be positive and retrieval output must be new")
    state = json.loads((args.checkpoint / "joint_state.json").read_text())
    if state.get("pooling") != "first_token":
        raise ValueError("retrieval checkpoint must use first_token pooling")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset = LEDBaselineDataset(
        args.data_dir / "test" / "baseline.jsonl", tokenizer,
        int(state["max_input_length"]), int(state["max_target_length"]),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=LEDBaselineCollator(tokenizer), drop_last=False,
    )
    device = torch.device(args.device)
    model = load_led_triplet(
        args.checkpoint, attention_window=int(state["attention_window"])
    ).to(device).eval()
    if model.pooling != "first_token":
        raise ValueError("reloaded checkpoint did not preserve first_token pooling")
    vectors = []
    with torch.inference_mode():
        for batch in loader:
            vector, _ = model._encode(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["global_attention_mask"].to(device),
            )
            vectors.append(vector.cpu().numpy())
    embeddings = np.concatenate(vectors).astype(np.float32, copy=False)
    event_ids = [dataset.record(index).event_id for index in range(len(dataset))]
    ideologies = [dataset.record(index).input_ideology for index in range(len(dataset))]
    scores = retrieval_scores(embeddings, event_ids, ideologies)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str((args.data_dir / "test" / "baseline.jsonl").resolve()),
        "split": "test", "pooling": "first_token",
        "similarity": "cosine of L2-normalized embeddings",
        "match": "same prepared event_id across opposite ideologies",
        "triplet_margin": state.get("triplet_margin"),
        "triplet_lambda": state.get("triplet_lambda"),
        "articles": len(dataset), **scores,
    }
    atomic_write_json(args.output, result)
    if args.require_wandb:
        import wandb
        run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            name=args.wandb_run_name, job_type="retrieval-evaluation",
            config={key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()},
            dir=str(args.output.parent), resume="never",
        )
        try:
            run.log({f"retrieval/{direction}/{metric}": value
                     for direction, values in scores.items()
                     for metric, value in values.items()})
            run.summary.update(result)
        finally:
            run.finish()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
