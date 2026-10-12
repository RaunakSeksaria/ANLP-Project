#!/usr/bin/env python3
"""Evaluate ideology geometry in existing, row-aligned article embeddings.

Only sample_id and ideology are read from metadata. No model, article text,
neutral rewrite, event identifier, or publisher is used by this evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


EMBEDDING_PATTERN = re.compile(r"^(lambda_\d+(?:_\d+)?)_embeddings\.npy$")
COLORS = {"left": "#285a9d", "right": "#c4573e"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_value(key: str) -> float:
    return float(key.removeprefix("lambda_").replace("_", "."))


def plot_suffix(key: str) -> str:
    # lambda_0_35 -> lambda_035; lambda_0_1 -> lambda_01.
    return "lambda_" + key.removeprefix("lambda_").replace("_", "")


def load_and_verify(data_dir: Path):
    metadata_path = data_dir / "metadata.parquet"
    manifest_path = data_dir / "manifest.json"
    readme_path = data_dir / "README.md"
    for path in (metadata_path, manifest_path, readme_path):
        if not path.is_file():
            raise ValueError(f"Missing row-alignment evidence or metadata: {path}")

    manifest = json.loads(manifest_path.read_text())
    readme = readme_path.read_text()
    if "Row `i` of every `*_embeddings.npy` is row `i` of `metadata.parquet`" not in readme:
        raise ValueError("README does not explicitly guarantee embedding/metadata row order")
    if "original_article_fields" not in manifest or "ignored_rewrite_fields" not in manifest:
        raise ValueError("Manifest does not establish original article extraction")

    table = pq.read_table(metadata_path, columns=["sample_id", "ideology"])
    ids = np.asarray(table["sample_id"].to_pylist(), dtype=str)
    labels = np.asarray(table["ideology"].to_pylist(), dtype=str)
    if len(ids) == 0 or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate sample IDs")
    if set(labels) != {"left", "right"}:
        raise ValueError(f"Ideology labels must be exactly left/right: {set(labels)}")
    if int(manifest["samples"]) != len(ids):
        raise ValueError("Manifest and metadata row counts differ")
    if int(manifest["left_samples"]) != int(np.count_nonzero(labels == "left")):
        raise ValueError("Manifest left count differs from metadata")
    if int(manifest["right_samples"]) != int(np.count_nonzero(labels == "right")):
        raise ValueError("Manifest right count differs from metadata")
    expected_ids = [f"left_{i:06d}" for i in range(int(manifest["left_samples"]))]
    expected_ids += [f"right_{i:06d}" for i in range(int(manifest["right_samples"]))]
    if ids.tolist() != expected_ids:
        raise ValueError("Metadata sample IDs violate the documented extraction row order")

    files = sorted(data_dir.glob("lambda_*_embeddings.npy"), key=lambda p: lambda_value(EMBEDDING_PATTERN.match(p.name).group(1)) if EMBEDDING_PATTERN.match(p.name) else float("inf"))
    if not files:
        raise ValueError("No lambda embedding files found")
    matrices = {}
    file_info = {}
    hidden_dim = None
    for path in files:
        match = EMBEDDING_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognized embedding filename: {path.name}")
        key = match.group(1)
        if key not in manifest.get("checkpoints", {}) or key not in manifest.get("shapes", {}):
            raise ValueError(f"No checkpoint/shape mapping for {key} in extraction manifest")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError(f"Invalid shape for {path}: {matrix.shape}")
        if list(matrix.shape) != manifest["shapes"][key]:
            raise ValueError(f"Manifest shape mismatch for {path}")
        if not np.issubdtype(matrix.dtype, np.floating) or str(matrix.dtype) != manifest["dtype"]:
            raise ValueError(f"Unexpected dtype for {path}: {matrix.dtype}")
        if hidden_dim is None:
            hidden_dim = matrix.shape[1]
        if matrix.shape[1] != hidden_dim or hidden_dim != int(manifest["hidden_dim"]):
            raise ValueError(f"Hidden dimension mismatch for {path}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"NaN or infinity in {path}")
        if np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
            raise ValueError(f"Zero/near-zero embedding row in {path}")
        matrices[key] = matrix
        file_info[key] = {
            "filename": path.name,
            "checkpoint": manifest["checkpoints"][key],
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": sha256(path),
        }
    return ids, labels, matrices, file_info, manifest, {
        "metadata_file": metadata_path.name,
        "metadata_sha256": sha256(metadata_path),
        "extraction_manifest_sha256": sha256(manifest_path),
        "row_alignment_evidence": "Extraction README explicitly maps row i of every embedding matrix to metadata row i; metadata IDs also match documented left-then-right source order.",
    }


def balanced_sample(labels: np.ndarray, requested: int, seed: int) -> np.ndarray:
    if requested < 2:
        raise ValueError("--sample-size must be at least 2")
    per_class = min(requested // 2, int(np.count_nonzero(labels == "left")), int(np.count_nonzero(labels == "right")))
    if per_class < 2:
        raise ValueError("Need at least two articles per ideology for silhouette")
    rng = np.random.default_rng(seed)
    left = rng.choice(np.flatnonzero(labels == "left"), size=per_class, replace=False)
    right = rng.choice(np.flatnonzero(labels == "right"), size=per_class, replace=False)
    return np.sort(np.concatenate([left, right]))


def normalize_subset(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # float64 avoids accumulated rounding error; source .npy remains untouched.
    x = np.asarray(matrix[indices], dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Invalid norm in evaluation subset")
    z = x / norms[:, None]
    if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-6):
        raise ValueError("Normalization verification failed")
    return z


def evaluate(z: np.ndarray, labels: np.ndarray, seed: int):
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=20)
    predicted = kmeans.fit_predict(z)
    ari = float(adjusted_rand_score(labels, predicted))
    nmi = float(normalized_mutual_info_score(labels, predicted))
    silhouette = float(silhouette_score(z, labels, metric="cosine"))
    left_centroid = z[labels == "left"].mean(axis=0)
    right_centroid = z[labels == "right"].mean(axis=0)
    centroid_distance = float(1 - np.dot(left_centroid, right_centroid) / (np.linalg.norm(left_centroid) * np.linalg.norm(right_centroid)))
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    coordinates = pca.fit_transform(z)
    return {
        "ari": ari,
        "nmi": nmi,
        "ideology_silhouette": silhouette,
        "centroid_cosine_distance": centroid_distance,
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_cluster_sizes": np.bincount(predicted, minlength=2).tolist(),
        "pca_pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
    }, coordinates


def save_plot(coordinates: np.ndarray, labels: np.ndarray, result: dict, key: str, limits, plot_dir: Path):
    fig, ax = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    for ideology in ("left", "right"):
        mask = labels == ideology
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=12, alpha=0.55,
                   color=COLORS[ideology], label=f"{ideology.title()} (n={mask.sum()})", linewidths=0)
    ax.set_xlabel(f"PC1 ({100 * result['pca_pc1_explained_variance']:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * result['pca_pc2_explained_variance']:.1f}% variance)")
    ax.set_title(f"Original article encoder embeddings · λ={lambda_value(key):g}")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.legend(frameon=False, markerscale=1.8)
    ax.grid(alpha=0.16)
    stem = f"ideology_pca_{plot_suffix(key)}"
    paths = []
    for extension in ("png", "pdf"):
        path = plot_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=240)
        paths.append(str(path))
    plt.close(fig)
    return paths


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log_wandb(results: list[dict], plots: dict, manifest: dict, project: str):
    try:
        import wandb
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        with wandb.init(project=project, mode=mode, job_type="ideology_clustering", config={
            "seed": manifest["seed"], "sample_size": manifest["sample_size"],
            "normalization": manifest["normalization"], "n_init": 20,
        }) as run:
            for row in results:
                key = row["lambda_key"]
                for metric in ("ari", "nmi", "ideology_silhouette", "centroid_cosine_distance"):
                    run.log({f"ideology_clustering/{metric}/{key}": row[metric]})
                run.log({f"ideology_clustering/pca/{key}": wandb.Image(plots[key][0])})
            columns = ["lambda", "ari", "nmi", "ideology_silhouette", "centroid_cosine_distance"]
            run.log({"ideology_clustering/results": wandb.Table(columns=columns, data=[
                [row[column] for column in columns] for row in results
            ])})
        return "succeeded" if mode == "online" else "saved offline (no WANDB_API_KEY)"
    except Exception as exc:
        print(f"W&B logging failed; local outputs are complete: {exc}")
        return f"failed: {type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("embedding_data_backup"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Opt in to Weights & Biases logging")
    parser.add_argument("--wandb-project", default="ANLP-embedding-evals")
    args = parser.parse_args()

    ids, labels, matrices, file_info, extraction, provenance = load_and_verify(args.data_dir)
    indices = balanced_sample(labels, args.sample_size, args.seed)
    sampled_ids = ids[indices]
    sampled_labels = labels[indices]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.results_dir / "ideology_sample_ids.csv"
    write_csv(sample_path, [
        {"row_index": int(index), "sample_id": str(sample_id), "ideology": str(label)}
        for index, sample_id, label in zip(indices, sampled_ids, sampled_labels)
    ], ["row_index", "sample_id", "ideology"])

    results = []
    coordinates = {}
    for key, matrix in matrices.items():
        z = normalize_subset(matrix, indices)
        result, coords = evaluate(z, sampled_labels, args.seed)
        results.append({"lambda": lambda_value(key), "lambda_key": key, **result})
        coordinates[key] = coords
        print(f"λ={lambda_value(key):g}: ARI={result['ari']:.5f}, NMI={result['nmi']:.5f}, "
              f"silhouette={result['ideology_silhouette']:.5f}, "
              f"centroid cosine distance={result['centroid_cosine_distance']:.5f}, "
              f"inertia={result['kmeans_inertia']:.4f}")
    # Shared plot limits make the panels directly readable, despite independent PCA fits.
    all_coords = np.concatenate(list(coordinates.values()))
    mins = all_coords.min(axis=0)
    maxes = all_coords.max(axis=0)
    padding = np.maximum((maxes - mins) * 0.04, 0.01)
    limits = (mins[0] - padding[0], maxes[0] + padding[0], mins[1] - padding[1], maxes[1] + padding[1])
    plots = {key: save_plot(coordinates[key], sampled_labels, row, key, limits, args.plots_dir)
             for row in results for key in [row["lambda_key"]]}

    csv_path = args.results_dir / "ideology_clustering.csv"
    fields = ["lambda", "lambda_key", "ari", "nmi", "ideology_silhouette", "centroid_cosine_distance",
              "kmeans_inertia", "kmeans_cluster_sizes", "pca_pc1_explained_variance", "pca_pc2_explained_variance"]
    write_csv(csv_path, [{**row, "kmeans_cluster_sizes": json.dumps(row["kmeans_cluster_sizes"])} for row in results], fields)
    json_path = args.results_dir / "ideology_clustering.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    eval_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "input_dir": str(args.data_dir),
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "sample_size": len(indices),
        "sample_counts": {ideology: int(np.count_nonzero(sampled_labels == ideology)) for ideology in COLORS},
        "sample_id_column": "sample_id",
        "ideology_column": "ideology",
        "sample_ids": sampled_ids.tolist(),
        "sample_indices": indices.tolist(),
        "sample_ids_csv": str(sample_path),
        "normalization": "L2 per row on an in-memory copy; input files unchanged",
        "kmeans": {"k": 2, "random_state": args.seed, "n_init": 20, "fit_uses_ideology_labels": False},
        "silhouette": {"labels": "true ideology", "metric": "cosine"},
        "centroid_distance": "1 - cosine similarity of left/right centroids of normalized embeddings",
        "pca": {"components": 2, "solver": "randomized", "random_state": args.seed, "use": "qualitative visualization only"},
        "stability": "single fixed seed; K-means uses 20 initializations and inertia is reported",
        "source_embeddings": file_info,
        "source_manifest": {"original_article_fields": extraction["original_article_fields"],
                            "ignored_rewrite_fields": extraction["ignored_rewrite_fields"],
                            "pooling": extraction.get("pooling")},
        "provenance": provenance,
        "plots": plots,
        "wandb": "disabled",
        "note": "Only sample_id and ideology metadata columns were read. No event, publisher, article, or rewrite content was used.",
    }
    if args.wandb:
        eval_manifest["wandb"] = log_wandb(results, plots, eval_manifest, args.wandb_project)
    manifest_path = args.results_dir / "ideology_clustering_manifest.json"
    manifest_path.write_text(json.dumps(eval_manifest, indent=2) + "\n")
    print(f"Sample: {len(indices)} articles ({eval_manifest['sample_counts']}); models: {', '.join(matrices)}")
    print(f"Results: {csv_path}, {json_path}, {manifest_path}; sample IDs: {sample_path}")


if __name__ == "__main__":
    main()
