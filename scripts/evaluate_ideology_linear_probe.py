#!/usr/bin/env python3
"""Linear ideology probes on frozen original-article encoder embeddings.

Runs paired stratified and publisher-held-out folds on every available lambda
matrix. Reads only sample_id, ideology, and publisher metadata columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_ideology_clustering import load_and_verify, sha256


METRICS = ("accuracy", "macro_f1", "roc_auc")


def label_for_key(key: str) -> str:
    return f"λ={float(key.removeprefix('lambda_').replace('_', '.')):g}"


def load_publisher(data_dir: Path, sample_ids: np.ndarray):
    path = data_dir / "metadata.parquet"
    table = pq.read_table(path, columns=["sample_id", "publisher"])
    ids = table["sample_id"].to_pylist()
    if ids != sample_ids.tolist():
        raise ValueError("Publisher metadata row order differs from the embedding sample IDs")
    raw = table["publisher"].to_pylist()
    if any(value is None or not str(value).strip() for value in raw):
        raise ValueError("Missing/empty publisher label; publisher-held-out probe is invalid")
    groups = np.asarray([str(value).strip() for value in raw], dtype=str)
    return groups


def verify_folds(folds, labels: np.ndarray, publishers: np.ndarray | None, n: int):
    if len(folds) < 2:
        return False, "fewer than two folds"
    test_indices = []
    for number, (train, test) in enumerate(folds):
        if len(train) == 0 or len(test) == 0:
            return False, f"empty train/test set in fold {number}"
        if len(set(labels[train])) != 2 or len(set(labels[test])) != 2:
            return False, f"fold {number} lacks an ideology class in train or test"
        if publishers is not None and set(publishers[train]).intersection(publishers[test]):
            return False, f"publisher leakage in fold {number}"
        test_indices.extend(test.tolist())
    if sorted(test_indices) != list(range(n)):
        return False, "test folds do not partition all samples exactly once"
    return True, "valid"


def construct_folds(labels: np.ndarray, publishers: np.ndarray, requested: int, seed: int):
    n = len(labels)
    dummy = np.zeros((n, 1))
    random_folds = list(StratifiedKFold(n_splits=requested, shuffle=True, random_state=seed).split(dummy, labels))
    valid, reason = verify_folds(random_folds, labels, None, n)
    if not valid:
        raise ValueError(f"Stratified folds invalid: {reason}")
    grouped = None
    rejected = {}
    max_splits = min(requested, len(set(publishers)))
    for count in range(max_splits, 1, -1):
        proposed = list(StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed).split(dummy, labels, publishers))
        valid, reason = verify_folds(proposed, labels, publishers, n)
        if valid:
            grouped = proposed
            break
        rejected[str(count)] = reason
    if grouped is None:
        raise ValueError(f"No publisher-disjoint grouped CV with both classes in all folds: {rejected}")
    return {"stratified": random_folds, "publisher_held_out": grouped}, rejected


def make_probe(seed: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                           solver="liblinear", random_state=seed)),
    ])


def fit_and_score(matrix: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int):
    probe = make_probe(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(matrix[train], labels[train])
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("Logistic regression failed to converge within max_iter=5000")
    predicted = probe.predict(matrix[test])
    probability_right = probe.predict_proba(matrix[test])[:, list(probe.classes_).index(1)]
    return {"accuracy": float(accuracy_score(labels[test], predicted)),
            "macro_f1": float(f1_score(labels[test], predicted, average="macro")),
            "roc_auc": float(roc_auc_score(labels[test], probability_right)),
            "iterations": int(probe.named_steps["classifier"].n_iter_[0])}


def fold_descriptor(kind: str, number: int, train: np.ndarray, test: np.ndarray,
                    sample_ids: np.ndarray, labels: np.ndarray, publishers: np.ndarray):
    train_count = Counter(labels[train])
    test_count = Counter(labels[test])
    train_publishers = sorted(set(publishers[train]))
    test_publishers = sorted(set(publishers[test]))
    if kind == "publisher_held_out" and set(train_publishers).intersection(test_publishers):
        raise ValueError(f"Publisher leakage in {kind} fold {number}")
    majority = int(train_count[1] > train_count[0])
    return {
        "split": kind, "fold": number,
        "train_sample_ids": sample_ids[train].tolist(), "test_sample_ids": sample_ids[test].tolist(),
        "train_count": len(train), "test_count": len(test),
        "train_ideology_counts": {"left": int(train_count[0]), "right": int(train_count[1])},
        "test_ideology_counts": {"left": int(test_count[0]), "right": int(test_count[1])},
        "train_publishers": train_publishers, "test_publishers": test_publishers,
        "majority_train_class_test_accuracy": float(np.mean(labels[test] == majority)),
    }


def summarize_folds(rows: list[dict], model_keys: list[str], fold_kinds: list[str]):
    summary = []
    for kind in fold_kinds:
        for key in model_keys:
            selected = [row for row in rows if row["split"] == kind and row["lambda_key"] == key]
            record = {"split": kind, "lambda_key": key, "lambda": label_for_key(key), "folds": len(selected)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(record)
    return summary


def paired_changes(rows: list[dict], model_keys: list[str], fold_kinds: list[str]):
    if "lambda_0" not in model_keys:
        return []
    comparisons = []
    for kind in fold_kinds:
        baseline = {row["fold"]: row for row in rows if row["split"] == kind and row["lambda_key"] == "lambda_0"}
        for key in model_keys:
            if key == "lambda_0":
                continue
            model = {row["fold"]: row for row in rows if row["split"] == kind and row["lambda_key"] == key}
            if model.keys() != baseline.keys():
                raise ValueError(f"Fold IDs differ between {key} and lambda_0")
            record = {"split": kind, "comparison": f"{key} - lambda_0", "folds": len(baseline)}
            for metric in METRICS:
                changes = [model[index][metric] - baseline[index][metric] for index in sorted(baseline)]
                record[f"{metric}_mean_change"] = float(np.mean(changes))
                record[f"{metric}_fold_changes"] = changes
            comparisons.append(record)
    return comparisons


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary: list[dict], keys: list[str], directory: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    x = np.arange(len(keys))
    offsets = {"stratified": -0.12, "publisher_held_out": 0.12}
    colors = {"stratified": "#2864a0", "publisher_held_out": "#c26043"}
    for kind in ("stratified", "publisher_held_out"):
        records = {row["lambda_key"]: row for row in summary if row["split"] == kind}
        means = [records[key]["macro_f1_mean"] for key in keys]
        stds = [records[key]["macro_f1_std"] for key in keys]
        count = records[keys[0]]["folds"]
        ax.errorbar(x + offsets[kind], means, yerr=stds, fmt="o", markersize=8,
                    color=colors[kind], capsize=4, linewidth=1.5,
                    label=f"{'Stratified' if kind == 'stratified' else 'Publisher held out'} ({count} folds)")
    ax.set_xticks(x, [label_for_key(key) for key in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (mean ± fold SD)")
    ax.set_xlabel("Triplet-loss weight")
    ax.set_title("Frozen-embedding ideology linear probe")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"ideology_linear_probe_f1.{extension}"
        fig.savefig(path, dpi=240)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("embedding_data_backup"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Requested stratified and maximum grouped folds")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("At least two folds are required")

    sample_ids, ideology, matrices, file_info, source_manifest, provenance = load_and_verify(args.data_dir)
    publishers = load_publisher(args.data_dir, sample_ids)
    y = np.asarray(ideology == "right", dtype=np.int8)
    counts = Counter(publishers)
    if len(counts) < 2:
        raise ValueError("Publisher-held-out evaluation requires at least two publishers")
    folds, rejected = construct_folds(y, publishers, args.folds, args.seed)
    assignments = [fold_descriptor(kind, number, train, test, sample_ids, y, publishers)
                   for kind, split_list in folds.items() for number, (train, test) in enumerate(split_list)]
    assignment_lookup = {(record["split"], record["fold"]): record for record in assignments}
    rows = []
    for key, matrix in matrices.items():
        for kind, split_list in folds.items():
            for number, (train, test) in enumerate(split_list):
                scores = fit_and_score(matrix, y, train, test, args.seed)
                assignment = assignment_lookup[(kind, number)]
                rows.append({"split": kind, "lambda_key": key, "lambda": label_for_key(key),
                             "fold": number, "train_count": len(train), "test_count": len(test),
                             "train_publishers": len(assignment["train_publishers"]),
                             "test_publishers": len(assignment["test_publishers"]),
                             "test_left": assignment["test_ideology_counts"]["left"],
                             "test_right": assignment["test_ideology_counts"]["right"],
                             "majority_train_class_test_accuracy": assignment["majority_train_class_test_accuracy"],
                             **scores})
                print(f"{kind} {label_for_key(key)} fold {number + 1}/{len(split_list)}: "
                      f"acc={scores['accuracy']:.4f}, F1={scores['macro_f1']:.4f}, "
                      f"AUC={scores['roc_auc']:.4f}, iterations={scores['iterations']}", flush=True)

    keys = list(matrices)
    summary = summarize_folds(rows, keys, list(folds))
    comparisons = paired_changes(rows, keys, list(folds))
    # One deterministic label-permutation check, using the standard 5 folds
    # and the lambda=0 representation. The embeddings and fold indices stay fixed.
    reference_key = "lambda_0" if "lambda_0" in matrices else keys[0]
    permuted = np.random.default_rng(args.seed + 1).permutation(y)
    permutation_scores = []
    for number, (train, test) in enumerate(folds["stratified"]):
        if len(set(permuted[train])) != 2 or len(set(permuted[test])) != 2:
            raise ValueError("Permuted labels lost a class in a fold")
        permutation_scores.append({"fold": number, **fit_and_score(matrices[reference_key], permuted, train, test, args.seed)})
    permutation = {"reference_model": reference_key, "seed": args.seed + 1,
                   "method": "globally permute ideology labels once; reuse stratified fold indices; fit pipeline anew",
                   "fold_scores": permutation_scores,
                   **{f"{metric}_mean": float(np.mean([row[metric] for row in permutation_scores])) for metric in METRICS}}

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.results_dir / "ideology_linear_probe.csv"
    summary_json = args.results_dir / "ideology_linear_probe.json"
    fold_csv = args.results_dir / "ideology_linear_probe_folds.csv"
    write_csv(summary_csv, summary)
    write_csv(fold_csv, rows)
    summary_json.write_text(json.dumps({"summary": summary, "paired_changes": comparisons,
                                        "permutation": permutation}, indent=2) + "\n")
    plot_paths = save_plot(summary, keys, args.plots_dir)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    publisher_ideology = {publisher: {"left": int(np.count_nonzero((publishers == publisher) & (y == 0))),
                                       "right": int(np.count_nonzero((publishers == publisher) & (y == 1)))}
                          for publisher in sorted(counts)}
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "data_dir": str(args.data_dir), "seed": args.seed, "requested_folds": args.folds,
        "stratified_folds": len(folds["stratified"]),
        "publisher_held_out_folds": len(folds["publisher_held_out"]),
        "grouped_fold_rejections": rejected,
        "grouped_fold_reason": "All 8 publishers are ideology-exclusive; only 3 right publishers, so 5 publisher-held-out test folds cannot all contain both classes.",
        "source_embeddings": file_info, "metadata": provenance,
        "sample_id_column": "sample_id", "ideology_column": "ideology", "publisher_column": "publisher",
        "sample_count": len(sample_ids),
        "ideology_counts": {"left": int(np.count_nonzero(y == 0)), "right": int(np.count_nonzero(y == 1))},
        "publisher_count": len(counts), "publisher_counts": dict(counts),
        "publisher_ideology_counts": publisher_ideology,
        "majority_class_accuracy": max(np.count_nonzero(y == 0), np.count_nonzero(y == 1)) / len(y),
        "balanced_chance_accuracy": 0.5,
        "preprocessing": "Stored original-article embeddings used as is; StandardScaler fit inside each training-fold Pipeline only",
        "source_pooling": source_manifest.get("pooling"),
        "classifier": {"class": "sklearn.linear_model.LogisticRegression", "solver": "liblinear",
                       "max_iter": 5000, "class_weight": "balanced", "random_state": args.seed},
        "scikit_learn_version": sklearn.__version__,
        "fold_assignments": assignments,
        "paired_comparisons": comparisons,
        "permutation": permutation,
        "plot_files": plot_paths,
        "ignored_metadata": ["event_ids", "neutral rewrites", "article text"],
        "row_order_limitation": "The .npy matrices contain no embedded IDs; their row alignment follows the extraction README/manifest and the metadata's documented sample ID sequence.",
    }
    manifest_path = args.results_dir / "ideology_linear_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sizes = sorted(counts.values())
    print(f"Samples: {len(y)} ({np.count_nonzero(y == 0)} left, {np.count_nonzero(y == 1)} right); "
          f"publishers: {len(counts)}, min/median/mean/max = {sizes[0]}/{np.median(sizes):.1f}/{np.mean(sizes):.1f}/{sizes[-1]}")
    print(f"Publisher-held-out folds: {len(folds['publisher_held_out'])}; "
          f"permuted {label_for_key(reference_key)} accuracy/F1/AUC: "
          f"{permutation['accuracy_mean']:.4f}/{permutation['macro_f1_mean']:.4f}/{permutation['roc_auc_mean']:.4f}")
    print(f"Results: {summary_csv}, {summary_json}, {fold_csv}, {manifest_path}; plots: {', '.join(plot_paths)}")


if __name__ == "__main__":
    main()
