#!/usr/bin/env python3
"""Compare ideology clustering and pairwise geometry of original-article embeddings.

Uses only *_original.npy files from the matched 2k dataset. Neutral embeddings,
neutral text, event IDs, and model checkpoints are never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from evaluate_ideology_clustering import evaluate as clustering_metrics


MODEL_PATTERN = re.compile(r"^(base|lambda_\d+(?:_\d+)?)_original\.npy$")
PAIR_NAMES = {"left_left": "Left–Left", "right_right": "Right–Right", "left_right": "Left–Right"}
PAIR_COLORS = {"left_left": "#285a9d", "right_right": "#c4573e", "left_right": "#6851a2"}
IDEOLOGY_COLORS = {"left": "#285a9d", "right": "#c4573e"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def model_sort(key: str):
    return -1.0 if key == "base" else float(key.removeprefix("lambda_").replace("_", "."))


def model_name(key: str):
    return "Pretrained LED" if key == "base" else f"λ={model_sort(key):g}"


def read_matched_dataset(data_dir: Path):
    metadata_path = data_dir / "metadata.parquet"
    manifest_path = data_dir / "manifest.json"
    readme_path = data_dir / "README.md"
    for path in (metadata_path, manifest_path, readme_path):
        if not path.is_file():
            raise ValueError(f"Missing required data/row-order record: {path}")
    manifest = json.loads(manifest_path.read_text())
    if "Row `i` of all six matrices is row `i` of `metadata.parquet`" not in readme_path.read_text():
        raise ValueError("Dataset README does not establish shared matrix row order")
    if not manifest.get("original_fields") or not manifest.get("checkpoints"):
        raise ValueError("Manifest lacks original-article fields or checkpoint mapping")
    if manifest.get("base_description") != "pretrained LED before ANLP task training; not randomly initialized":
        raise ValueError("The meaning of the base model is not established")

    columns = ["sample_id", "ideology", "original_article", "original_article_sha256"]
    rows = pq.read_table(metadata_path, columns=columns).to_pydict()
    ids = rows["sample_id"]
    labels = np.asarray(rows["ideology"], dtype=str)
    if len(ids) != int(manifest["samples"]) or len(ids) != len(set(ids)):
        raise ValueError("Metadata row count or sample ID uniqueness failed")
    if set(labels) != {"left", "right"}:
        raise ValueError(f"Unexpected ideology labels: {set(labels)}")
    if (int(np.count_nonzero(labels == "left")), int(np.count_nonzero(labels == "right"))) != (int(manifest["left_samples"]), int(manifest["right_samples"])):
        raise ValueError("Class balance differs from the extraction manifest")
    if ids != sorted(ids, key=lambda item: (not item.startswith("left_"), int(item.split("_")[-1]))):
        raise ValueError("Sample IDs are not in documented source order")
    for sample_id, label, article, expected_hash in zip(ids, labels, rows["original_article"], rows["original_article_sha256"]):
        if not sample_id.startswith(label + "_") or not article or hashlib.sha256(article.encode()).hexdigest() != expected_hash:
            raise ValueError(f"Original-article metadata/hash mismatch for {sample_id}")

    paths = list(data_dir.glob("*_original.npy"))
    if not paths:
        raise ValueError("No original-article embedding files found")
    arrays = {}
    file_info = {}
    for path in paths:
        match = MODEL_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognized original embedding file: {path.name}")
        key = match.group(1)
        if key not in manifest["checkpoints"] or path.stem not in manifest.get("shapes", {}):
            raise ValueError(f"Missing checkpoint or shape mapping for {path.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != len(ids) or array.shape[1] != int(manifest["hidden_dim"]):
            raise ValueError(f"Invalid shape for {path}: {array.shape}")
        if list(array.shape) != manifest["shapes"][path.stem] or str(array.dtype) != manifest["dtype"]:
            raise ValueError(f"Manifest shape/dtype mismatch for {path}")
        if not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) <= 1e-12):
            raise ValueError(f"Non-finite or zero embeddings in {path}")
        arrays[key] = array
        file_info[key] = {"file": path.name, "checkpoint": manifest["checkpoints"][key],
                          "shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest(path)}
    if not {"base", "lambda_0", "lambda_0_35"}.issubset(arrays):
        raise ValueError("Need base, lambda_0, and lambda_0_35 original matrices for this comparison")
    arrays = dict(sorted(arrays.items(), key=lambda item: model_sort(item[0])))
    return ids, labels, rows["original_article_sha256"], arrays, file_info, manifest, {
        "metadata_file": str(metadata_path), "metadata_sha256": digest(metadata_path),
        "extraction_manifest_sha256": digest(manifest_path),
        "row_order_evidence": "README maps row i of every matrix to metadata row i; metadata IDs are unique and source ordered; original article hashes were verified.",
    }


def verify_against_full(data_dir: Path, ids: list[str], labels: np.ndarray, hashes: list[str], arrays: dict):
    """Optional read-only cross-check against the larger paired original dataset."""
    metadata_path = data_dir / "metadata.parquet"
    if not metadata_path.is_file():
        return {"status": "skipped", "reason": f"{metadata_path} is unavailable"}
    full = pq.read_table(metadata_path, columns=["sample_id", "ideology", "original_article_sha256"]).to_pydict()
    lookup = {sample_id: i for i, sample_id in enumerate(full["sample_id"])}
    if len(lookup) != len(full["sample_id"]):
        raise ValueError("Duplicate sample IDs in full paired dataset")
    try:
        full_indices = np.array([lookup[sample_id] for sample_id in ids])
    except KeyError as exc:
        raise ValueError(f"2k sample missing from full paired dataset: {exc}") from exc
    if any(full["ideology"][i] != label or full["original_article_sha256"][i] != article_hash
           for i, label, article_hash in zip(full_indices, labels, hashes)):
        raise ValueError("Ideology or original-article hash differs from full paired dataset")
    differences = {}
    for key, subset in arrays.items():
        path = data_dir / f"{key}_original.npy"
        if not path.is_file():
            continue  # The pretrained base is only available in the 2k dataset.
        full_array = np.load(path, mmap_mode="r", allow_pickle=False)
        if full_array.ndim != 2 or len(full_array) != len(lookup) or full_array.shape[1] != subset.shape[1]:
            raise ValueError(f"Shape mismatch in full paired array: {path}")
        difference = np.abs(np.asarray(subset) - np.asarray(full_array[full_indices]))
        maximum = float(difference.max())
        if maximum > 1e-5:
            raise ValueError(f"2k and full original embeddings disagree for {key}: max difference {maximum}")
        differences[key] = maximum
    return {"status": "passed", "matched_sample_ids": len(ids), "matched_original_hashes": len(ids),
            "maximum_absolute_embedding_difference": differences, "full_metadata_sha256": digest(metadata_path)}


def normalized(array: np.ndarray):
    z = np.asarray(array, dtype=np.float64).copy()
    norms = np.linalg.norm(z, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Invalid norm")
    z /= norms[:, None]
    if not np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-6):
        raise ValueError("Normalization failed")
    return z


def cosine_pairs(a: np.ndarray, b: np.ndarray, within: bool, block: int = 128):
    parts = []
    columns = np.arange(len(b))
    for start in range(0, len(a), block):
        end = min(start + block, len(a))
        similarity = np.clip(a[start:end] @ b.T, -1, 1)
        if within:
            parts.append(1 - similarity[columns[None, :] > np.arange(start, end)[:, None]])
        else:
            parts.append((1 - similarity).ravel())
    return np.concatenate(parts)


def bootstrap_article_counts(n_left: int, n_right: int, repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    return (rng.multinomial(n_left, np.full(n_left, 1 / n_left), size=repetitions).astype(np.float64),
            rng.multinomial(n_right, np.full(n_right, 1 / n_right), size=repetitions).astype(np.float64))


def boot_means(left: np.ndarray, right: np.ndarray, left_counts: np.ndarray, right_counts: np.ndarray):
    """Exact pair means for article-resampled positions, using vector-sum identities."""
    nl, nr = len(left), len(right)
    sl, sr = left_counts @ left, right_counts @ right
    return {
        "left_left": 1 - (np.einsum("ij,ij->i", sl, sl) - nl) / (nl * (nl - 1)),
        "right_right": 1 - (np.einsum("ij,ij->i", sr, sr) - nr) / (nr * (nr - 1)),
        "left_right": 1 - np.einsum("ij,ij->i", sl, sr) / (nl * nr),
    }


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def pairwise_results(z: np.ndarray, labels: np.ndarray, counts_left: np.ndarray, counts_right: np.ndarray):
    left, right = z[labels == "left"], z[labels == "right"]
    pair_values = {"left_left": cosine_pairs(left, left, True),
                   "right_right": cosine_pairs(right, right, True),
                   "left_right": cosine_pairs(left, right, False)}
    expected = {"left_left": len(left) * (len(left) - 1) // 2,
                "right_right": len(right) * (len(right) - 1) // 2,
                "left_right": len(left) * len(right)}
    boot = boot_means(left, right, counts_left, counts_right)
    summaries = {}
    for pair, values in pair_values.items():
        if len(values) != expected[pair] or not np.isfinite(values).all():
            raise ValueError(f"Invalid pairwise distances for {pair}")
        mean, median, std = map(float, (values.mean(), np.median(values), values.std()))
        ci = interval(boot[pair])
        summaries[pair] = {"pair_count": expected[pair],
                           "cosine": {"mean": mean, "median": median, "std": std, "mean_ci_95": ci},
                           "squared_l2": {"mean": 2 * mean, "median": 2 * median, "std": 2 * std,
                                          "mean_ci_95": [2 * x for x in ci]}}
    within = (summaries["left_left"]["cosine"]["mean"] + summaries["right_right"]["cosine"]["mean"]) / 2
    cross = summaries["left_right"]["cosine"]["mean"]
    boot_within = (boot["left_left"] + boot["right_right"]) / 2
    boot_gap = boot["left_right"] - boot_within
    relative = {"within_mean_cosine": within, "cross_minus_within_cosine": cross - within,
                "cross_over_within": cross / within,
                "cross_minus_within_cosine_ci_95": interval(boot_gap),
                "within_mean_squared_l2": 2 * within,
                "cross_minus_within_squared_l2": 2 * (cross - within)}
    return summaries, relative


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_pca_plot(coordinates: dict, labels: np.ndarray, clustering: list[dict], plot_dir: Path):
    models = list(coordinates)
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 4.7), sharex=True, sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    everything = np.concatenate(list(coordinates.values()))
    xlo, ylo = everything.min(axis=0)
    xhi, yhi = everything.max(axis=0)
    xpad, ypad = max((xhi - xlo) * 0.04, 0.01), max((yhi - ylo) * 0.04, 0.01)
    metrics = {row["model_key"]: row for row in clustering}
    for ax, key in zip(axes, models):
        x = coordinates[key]
        for ideology in ("left", "right"):
            keep = labels == ideology
            ax.scatter(x[keep, 0], x[keep, 1], s=8, alpha=0.5, linewidths=0,
                       color=IDEOLOGY_COLORS[ideology], label=ideology.title())
        ax.set_title(model_name(key))
        ax.set_xlabel(f"PC1 ({100 * metrics[key]['pca_pc1_explained_variance']:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * metrics[key]['pca_pc2_explained_variance']:.1f}%)")
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.grid(alpha=0.12)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Original article embeddings · ideology labels · PCA", fontsize=13)
    paths = []
    for extension in ("png", "pdf"):
        path = plot_dir / f"original_base_ideology_pca.{extension}"
        fig.savefig(path, dpi=240)
        paths.append(str(path))
    plt.close(fig)
    return paths


def save_distance_plot(pairwise: list[dict], plot_dir: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5), constrained_layout=True)
    x = np.arange(len(pairwise))
    width = 0.24
    for offset, pair in enumerate(PAIR_NAMES):
        means = [row["pairs"][pair]["cosine"]["mean"] for row in pairwise]
        intervals = [row["pairs"][pair]["cosine"]["mean_ci_95"] for row in pairwise]
        errors = np.array([[mean - lo for mean, (lo, hi) in zip(means, intervals)],
                           [hi - mean for mean, (lo, hi) in zip(means, intervals)]])
        ax.bar(x + (offset - 1) * width, means, width, label=PAIR_NAMES[pair],
               color=PAIR_COLORS[pair], yerr=errors, capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, [model_name(row["model_key"]) for row in pairwise])
    ax.set_ylabel("Mean cosine distance (95% article-bootstrap CI)")
    ax.set_title("Original article embedding distances by ideology pair")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    paths = []
    for extension in ("png", "pdf"):
        path = plot_dir / f"original_base_ideology_pairwise_distance.{extension}"
        fig.savefig(path, dpi=240)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("neutral_pair_embedding_data_2k"))
    parser.add_argument("--full-data-dir", type=Path, default=Path("neutral_pair_embedding_data"),
                        help="Optional read-only cross-check with the 20k original arrays")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_replicates < 20:
        raise ValueError("Use at least 20 article-bootstrap replicates")

    ids, labels, hashes, arrays, file_info, source_manifest, input_record = read_matched_dataset(args.data_dir)
    full_check = verify_against_full(args.full_data_dir, ids, labels, hashes, arrays)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.results_dir / "original_base_ideology_sample_ids.csv"
    write_csv(sample_path, [{"row_index": i, "sample_id": sample_id, "ideology": str(labels[i])}
                            for i, sample_id in enumerate(ids)])

    nl, nr = int(np.count_nonzero(labels == "left")), int(np.count_nonzero(labels == "right"))
    left_counts, right_counts = bootstrap_article_counts(nl, nr, args.bootstrap_replicates, args.seed + 1)
    clustering, pairwise, coordinates = [], [], {}
    for key, matrix in arrays.items():
        z = normalized(matrix)
        cluster_result, coords = clustering_metrics(z, labels, args.seed)
        cluster_result = {"model": model_name(key), "model_key": key, **cluster_result}
        pair_summaries, relative = pairwise_results(z, labels, left_counts, right_counts)
        pair_result = {"model": model_name(key), "model_key": key,
                       "pairs": pair_summaries, "relative": relative}
        clustering.append(cluster_result)
        pairwise.append(pair_result)
        coordinates[key] = coords
        print(f"{model_name(key)}: ARI={cluster_result['ari']:.5f}, NMI={cluster_result['nmi']:.5f}, "
              f"silhouette={cluster_result['ideology_silhouette']:.5f}; "
              f"LL={pair_summaries['left_left']['cosine']['mean']:.6f}, "
              f"RR={pair_summaries['right_right']['cosine']['mean']:.6f}, "
              f"LR={pair_summaries['left_right']['cosine']['mean']:.6f}, "
              f"LR−within={relative['cross_minus_within_cosine']:.6f}")

    clustering_csv = args.results_dir / "original_base_ideology_clustering.csv"
    cluster_rows = [{**row, "kmeans_cluster_sizes": json.dumps(row["kmeans_cluster_sizes"])} for row in clustering]
    write_csv(clustering_csv, cluster_rows)
    clustering_json = args.results_dir / "original_base_ideology_clustering.json"
    clustering_json.write_text(json.dumps(clustering, indent=2) + "\n")
    distance_rows = []
    for model in pairwise:
        for pair, values in model["pairs"].items():
            row = {"model": model["model"], "model_key": model["model_key"], "pair_type": pair,
                   "pair_count": values["pair_count"]}
            for metric in ("cosine", "squared_l2"):
                stats = values[metric]
                row.update({f"{metric}_mean": stats["mean"], f"{metric}_median": stats["median"],
                            f"{metric}_std": stats["std"], f"{metric}_ci_low": stats["mean_ci_95"][0],
                            f"{metric}_ci_high": stats["mean_ci_95"][1]})
            row.update(model["relative"])
            distance_rows.append(row)
    distance_csv = args.results_dir / "original_base_ideology_pairwise_distances.csv"
    write_csv(distance_csv, distance_rows)
    distance_json = args.results_dir / "original_base_ideology_pairwise_distances.json"
    distance_json.write_text(json.dumps(pairwise, indent=2) + "\n")
    pca_plots = save_pca_plot(coordinates, labels, clustering, args.plots_dir)
    distance_plots = save_distance_plot(pairwise, args.plots_dir)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    evaluation_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit,
        "data_dir": str(args.data_dir), "metadata": input_record,
        "source_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "source_selection": source_manifest.get("selection"),
        "source_selection_seed": source_manifest.get("selection_seed"),
        "models": file_info, "base_meaning": source_manifest["base_description"],
        "original_fields": source_manifest["original_fields"],
        "rows": len(ids), "left_rows": nl, "right_rows": nr,
        "sample_ids": ids, "sample_ids_csv": str(sample_path),
        "full_data_cross_check": full_check,
        "normalization": "L2 normalize a float64 in-memory copy; saved embeddings unchanged",
        "clustering": {"k": 2, "random_state": args.seed, "n_init": 20,
                       "labels_used_for_fit": False, "comparison": ["ARI", "NMI"],
                       "true_label_silhouette_metric": "cosine", "pca_components": 2},
        "pairwise": {"pair_counts": {"left_left": nl * (nl - 1) // 2,
                                     "right_right": nr * (nr - 1) // 2, "left_right": nl * nr},
                     "distances": ["cosine", "squared L2 = 2 * cosine on unit vectors"],
                     "bootstrap_unit": "article, independently resampled within ideology; same draws across models",
                     "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.seed + 1},
        "plots": {"pca": pca_plots, "distance": distance_plots},
        "ignored_fields": "neutral text/embeddings, event IDs, publishers",
    }
    manifest_path = args.results_dir / "original_base_ideology_manifest.json"
    manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n")
    print(f"Compared {len(ids)} identical original articles ({nl} left, {nr} right); "
          f"full-data cross-check: {full_check['status']}")
    print(f"Results: {clustering_csv}, {distance_csv}, {manifest_path}; plots: {', '.join(pca_plots + distance_plots)}")


if __name__ == "__main__":
    main()
