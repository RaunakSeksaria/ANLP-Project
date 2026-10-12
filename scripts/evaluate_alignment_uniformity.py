#!/usr/bin/env python3
"""Evaluate original-article embeddings without loading models or text fields."""
import argparse
import csv
import hashlib
import json
import importlib.metadata
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import pdist


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_articles(labels, size, seed):
    require(2 <= size <= len(labels), 'sample-size must be between 2 and the population size')
    rng = np.random.default_rng(seed)
    left, right = np.flatnonzero(labels == 'left'), np.flatnonzero(labels == 'right')
    if len(left) >= size // 2 and len(right) >= size - size // 2:
        return np.sort(np.r_[rng.choice(left, size // 2, replace=False),
                             rng.choice(right, size - size // 2, replace=False)]), 'stratified'
    return np.sort(rng.choice(len(labels), size, replace=False)), 'simple_random'


def positive_pairs(rows, labels, field):
    groups = defaultdict(list)
    memberships = []
    for i, row in enumerate(rows):
        value = row.get(field) if field else None
        values = value if isinstance(value, list) else [value]
        events = {str(v).strip() for v in values if v is not None and str(v).strip()}
        memberships.append(events)
        for event in events:
            groups[event].append(i)
    pairs = set()
    for members in groups.values():
        for i in members:
            for j in members:
                if i < j and labels[i] != labels[j]:
                    pairs.add((i, j))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2), memberships


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if len(values) else [None, None]


def metrics(z, selected, pairs, labels, sampling, seed, repetitions):
    distances = pdist(z[selected], metric='sqeuclidean')
    kernel = np.exp(-2 * distances)
    score = float(np.log(kernel.mean()))
    positive_distances = np.sum((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2, axis=1)
    alignment = float(positive_distances.mean()) if len(pairs) else None
    # Resample articles, not the dependent pairwise distances, for uniformity.
    # Multiplicities include zero-distance pairs between bootstrap duplicates.
    i, j = np.triu_indices(len(selected), k=1)
    uniformity_bootstrap, alignment_bootstrap = [], []
    rng_u = np.random.default_rng(seed + 1)
    rng_a = np.random.default_rng(seed + 2)
    strata = ([np.flatnonzero(labels[selected] == side) for side in ('left', 'right')]
              if sampling == 'stratified' else [np.arange(len(selected))])
    for _ in range(repetitions):
        counts = np.zeros(len(selected), dtype=np.int64)
        for stratum in strata:
            counts[stratum] = rng_u.multinomial(len(stratum), np.full(len(stratum), 1 / len(stratum)))
        total = np.dot(counts[i] * counts[j], kernel) + np.sum(counts * (counts - 1)) / 2
        uniformity_bootstrap.append(np.log(total / len(distances)))
        if len(pairs):
            alignment_bootstrap.append(positive_distances[rng_a.integers(len(pairs), size=len(pairs))].mean())
    a_ci, u_ci = interval(alignment_bootstrap), interval(uniformity_bootstrap)
    return dict(alignment=alignment, alignment_ci_low=a_ci[0], alignment_ci_high=a_ci[1],
                alignment_pairs=len(pairs), uniformity=score,
                uniformity_ci_low=u_ci[0], uniformity_ci_high=u_ci[1],
                uniformity_articles=len(selected), uniformity_distances=len(distances))


def plot_results(results, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric, ylabel in [('alignment', 'Alignment: mean squared distance ↓'),
                           ('uniformity', 'Uniformity: log mean exp(−2d²) ↓')]:
        available = [r for r in results if r[metric] is not None]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.5), layout='constrained')
        x = np.arange(len(available))
        y = np.array([r[metric] for r in available])
        ax.scatter(x, y, color='#245a81', zorder=3)
        for idx, r in enumerate(available):
            lo, hi = r[metric + '_ci_low'], r[metric + '_ci_high']
            if lo is not None:
                ax.vlines(idx, lo, hi, color='#245a81')
                ax.hlines([lo, hi], idx - .06, idx + .06, color='#245a81')
        ax.set(xticks=x, xticklabels=[str(r['lambda']) for r in available],
               xlabel='Triplet-loss weight λ', ylabel=ylabel, xlim=(-.4, len(x) - .6))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        for suffix in ('png', 'pdf'):
            path = directory / f'{metric}_comparison.{suffix}'
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
    return paths


def log_wandb(enabled, results, manifest, paths, results_dir):
    if not enabled:
        return {'status': 'disabled', 'reason': 'Enable explicitly with --wandb; no login required.'}
    try:
        import wandb
        if not (os.environ.get('WANDB_API_KEY') or wandb.api.api_key):
            return {'status': 'skipped', 'reason': 'No configured W&B credentials.'}
        run = wandb.init(project='ANLP-embedding-evals', job_type='evaluation',
                         config={k: manifest[k] for k in ('seed', 'hidden_dimension', 'pooling',
                                  'uniformity_sample_count', 'alignment_pair_count')},
                         settings=wandb.Settings(init_timeout=30))
        payload = {}
        for row in results:
            for metric in ('alignment', 'uniformity'):
                if row[metric] is not None:
                    payload[f'{metric}/lambda_{row["lambda"]:g}'] = row[metric]
        columns = list(results[0])
        payload['results'] = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in results])
        for path in paths:
            if path.suffix == '.png':
                payload[path.stem] = wandb.Image(str(path))
        run.log(payload)
        manifest['wandb'] = {'status': 'logged', 'url': run.url}
        (results_dir / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        artifact = wandb.Artifact('alignment-uniformity', type='evaluation')
        for path in paths + list(results_dir.glob('*.json')) + list(results_dir.glob('*.csv')):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        url = run.url
        run.finish()
        return {'status': 'logged', 'url': url}
    except Exception as exc:
        warnings.warn(f'Optional W&B logging failed: {exc}')
        return {'status': 'failed', 'reason': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('embedding_data_backup'))
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--plots-dir', type=Path, default=Path('plots'))
    parser.add_argument('--sample-size', type=int, default=2000)
    parser.add_argument('--max-alignment-pairs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-repetitions', type=int, default=500)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    require(args.max_alignment_pairs > 0 and args.bootstrap_repetitions >= 0, 'Invalid pair/bootstrap count')
    source = json.loads((args.data_dir / 'manifest.json').read_text())
    metadata_path = args.data_dir / 'metadata.parquet'
    fields = pq.read_schema(metadata_path).names
    require('sample_id' in fields and 'ideology' in fields, 'Missing sample_id or ideology field')
    event_field = next((k for k in ('event_ids', 'event_id', 'story_id', 'cluster_id') if k in fields), None)
    columns = [k for k in fields if k in ('sample_id', 'ideology', 'source_file', 'original_row_index',
                                         'article_id', 'publisher', 'story_id', 'event_ids', 'event_id', 'cluster_id')]
    rows = pq.read_table(metadata_path, columns=columns).to_pylist()
    ids = [r['sample_id'] for r in rows]
    labels = np.array([r['ideology'] for r in rows])
    require(len(ids) == len(set(ids)) == source['samples'], 'Duplicate IDs or metadata count mismatch')
    require(set(labels) == {'left', 'right'}, 'Expected left and right ideology labels')
    # This dataset declares left rows followed by right rows and positional IDs.
    expected_ids = [f'{side}_{i:06d}' for side in ('left', 'right')
                    for i in range(int(np.sum(labels == side)))]
    require(ids == expected_ids, 'Metadata sample order differs from extraction contract')
    for r in rows:
        require(r['sample_id'] == f"{r['ideology']}_{r['original_row_index']:06d}", 'Source row/ID mismatch')
    selected, sampling = select_articles(labels, args.sample_size, args.seed)
    candidates, memberships = positive_pairs(rows, labels, event_field)
    total_pairs = len(candidates)
    if total_pairs > args.max_alignment_pairs:
        choice = np.sort(np.random.default_rng(args.seed).choice(total_pairs, args.max_alignment_pairs, replace=False))
        pairs = candidates[choice]
    else:
        pairs = candidates
    alignment_reason = None
    if not event_field:
        alignment_reason = ('The current embedding dataset does not contain a reliable positive-pair identifier, '
                            'so event-level alignment cannot be computed without fabricating positives.')
    elif not len(pairs):
        alignment_reason = 'No same-event cross-ideology pairs exist.'
    examples = []
    for i, j in pairs:
        require(labels[i] != labels[j] and bool(memberships[i] & memberships[j]), 'Invalid alignment positive pair')
    for i, j in pairs[:5]:
        examples.append({'sample_ids': [ids[i], ids[j]], 'shared_events': sorted(memberships[i] & memberships[j]),
                         'ideologies': [str(labels[i]), str(labels[j])]})
    print('Positive-pair examples:', json.dumps(examples, indent=2))
    results, files = [], {}
    dimensions = set()
    for key, checkpoint in source['checkpoints'].items():
        weight = float(key.removeprefix('lambda_').replace('_', '.'))
        path = args.data_dir / f'{key}_embeddings.npy'
        require(path.is_file(), f'Manifest checkpoint is missing its matrix: {path}')
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        require(raw.ndim == 2 and raw.shape[0] == len(rows), f'Row-count mismatch: {path}')
        require(list(raw.shape) == source['shapes'][key], f'Manifest shape mismatch: {path}')
        require(np.isfinite(raw).all(), f'NaN/Inf in {path}')
        dimensions.add(raw.shape[1])
        require(len(dimensions) == 1 and raw.shape[1] == source['hidden_dim'], 'Hidden dimension mismatch')
        z = np.array(raw, dtype=np.float64, copy=True)
        norms = np.linalg.norm(z, axis=1)
        require(np.all(norms > 0), f'Zero-norm embedding: {path}')
        z /= norms[:, None]
        require(np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-12), 'Normalization failed')
        print(f'Evaluating lambda={weight:g}: {raw.shape}, {raw.dtype}', flush=True)
        result = metrics(z, selected, pairs, labels, sampling, args.seed, args.bootstrap_repetitions)
        results.append({'lambda': weight, **result})
        files[key] = {'file': str(path), 'sha256': sha256(path), 'shape': list(raw.shape),
                      'dtype': str(raw.dtype), 'checkpoint': checkpoint}
    require(bool(results), 'No embeddings in manifest')
    results.sort(key=lambda r: r['lambda'])
    missing = sorted({0., .1, .35} - {r['lambda'] for r in results})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    manifest = dict(seed=args.seed, hidden_dimension=next(iter(dimensions)), pooling=source.get('pooling'),
        metadata_file=str(metadata_path), metadata_fields=fields, metadata_sha256=sha256(metadata_path),
        ideology_field='ideology', event_field=event_field, sample_id_field='sample_id',
        normalization='float64 copy: z = h / ||h||_2; originals untouched',
        formulas={'alignment': 'mean(||z_i-z_j||^2) for same-event cross-ideology pairs',
                  'uniformity': 'log(mean(exp(-2*||z_i-z_j||^2))) over unique unordered article pairs'},
        pair_definition='Different ideology and nonempty intersection of event memberships; each article pair counted once',
        uniformity_sampling=sampling, uniformity_sample_count=len(selected), population_count=len(rows),
        uniformity_sample_ids=[ids[i] for i in selected], uniformity_row_indices=selected.tolist(),
        alignment_pair_count=len(pairs), total_valid_alignment_pairs=total_pairs,
        alignment_pair_ids=[[ids[i], ids[j]] for i, j in pairs], alignment_row_indices=pairs.tolist(),
        alignment_unavailable_reason=alignment_reason, positive_pair_examples=examples,
        embedding_files=files, missing_lambdas=missing,
        row_order_verification='Declared shared extraction order, metadata IDs, source indices and shapes checked; arrays have no embedded IDs, so arbitrary internal permutations cannot be independently detected.',
        bootstrap={'repetitions': args.bootstrap_repetitions, 'confidence': .95, 'interval': 'percentile',
                   'uniformity': 'Article bootstrap, stratified when sampling is stratified; repeated article copies have zero distance',
                   'alignment': 'Positive-pair bootstrap; conditional on sampled pairs; shared articles/events cause dependence not captured by this simple interval',
                   'seeds': {'uniformity': args.seed+1, 'alignment': args.seed+2},
                   'paired_across_models': True},
        timestamp=datetime.now(timezone.utc).isoformat(), git_commit=git.stdout.strip(),
        package_versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pyarrow', 'matplotlib')},
        neutral_text_used=False)
    manifest_path = args.results_dir / 'evaluation_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (args.results_dir / 'alignment_uniformity.json').write_text(json.dumps(results, indent=2) + '\n')
    with (args.results_dir / 'alignment_uniformity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    paths = plot_results(results, args.plots_dir)
    manifest['wandb'] = log_wandb(args.wandb, results, manifest, paths, args.results_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    # Confirm read-only input semantics, including during optional logging.
    require(sha256(metadata_path) == manifest['metadata_sha256'], 'Input metadata changed during evaluation')
    for file in files.values():
        require(sha256(Path(file['file'])) == file['sha256'], 'Input embedding file changed during evaluation')
    print(json.dumps(results, indent=2))
    print('W&B:', manifest['wandb'])


if __name__ == '__main__':
    main()
