#!/usr/bin/env python3
"""Audit saved original-article embeddings, then evaluate fixed metadata triplets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import csv
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATTERN = re.compile(r'^lambda_(\d+(?:[._]\d+)?)_embeddings\.npy$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def describe(values):
    return {key: float(fn(values)) for key, fn in
            [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}


def git_source(commit, path):
    result = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT,
                            capture_output=True, check=True)
    return result.stdout


def memberships_from_rows(rows, field):
    result = []
    for row in rows:
        raw = row[field]
        entries = raw if isinstance(raw, list) else [raw]
        require(bool(entries) and all(x is not None and str(x).strip() for x in entries),
                f'Missing event identity for {row["sample_id"]}')
        result.append(frozenset(str(x).strip() for x in entries))
    return result


def build_triplet_pools(rows, events):
    """Unique directed article tuples; negative must share NO event with anchor."""
    by_event = defaultdict(set)
    by_side = defaultdict(set)
    for i, row in enumerate(rows):
        by_side[row['ideology']].add(i)
        for event in events[i]:
            by_event[event].add(i)
    positives, negatives = {}, {}
    for i, row in enumerate(rows):
        overlapping = set().union(*(by_event[e] for e in events[i]))
        positives[i] = np.array(sorted(overlapping - by_side[row['ideology']]), dtype=np.int64)
        negatives[i] = np.array(sorted(by_side[row['ideology']] - overlapping), dtype=np.int64)
    count = sum(len(positives[a]) * len(negatives[a]) for a in positives)
    stats = {
        'unique_events': len(by_event),
        'articles_per_event': {e: len(v) for e, v in sorted(by_event.items())},
        'articles_per_event_statistics': describe(np.array([len(v) for v in by_event.values()])),
        'events_with_both_ideologies': sum(len({rows[i]['ideology'] for i in v}) == 2 for v in by_event.values()),
        'unique_cross_ideology_positive_pairs': sum(len(p) for p in positives.values()) // 2,
        'potential_directed_triplets': count,
        'anchors_with_valid_triplets': sum(bool(len(positives[a]) and len(negatives[a])) for a in positives),
    }
    return positives, negatives, stats


def audit_inputs(data, audit):
    manifest_path, metadata_path, readme_path = data / 'manifest.json', data / 'metadata.parquet', data / 'README.md'
    require(manifest_path.is_file() and metadata_path.is_file(), 'Missing manifest.json or metadata.parquet')
    manifest = json.loads(manifest_path.read_text())
    fields = pq.read_schema(metadata_path).names
    event_field = next((f for f in ('event_ids', 'event_id', 'story_id', 'cluster_id') if f in fields), None)
    needed = ['sample_id', 'ideology', 'original_article', 'original_article_sha256', 'original_row_index', 'source_file']
    require(all(f in fields for f in needed) and event_field is not None,
            'Cannot establish article, sample, ideology, or event identity')
    # Deliberately exclude all neutral/summary/generated text columns.
    columns = [f for f in fields if f in needed + [event_field, 'story_id', 'title', 'article_id',
                                                   'publisher', 'split', 'effective_input_length']]
    rows = pq.read_table(metadata_path, columns=columns).to_pylist()
    ids = [r['sample_id'] for r in rows]
    require(len(ids) > 1 and len(ids) == len(set(ids)), 'Empty data or duplicate sample IDs')
    counts = Counter(r['ideology'] for r in rows)
    require(set(counts) == {'left', 'right'}, f'Invalid ideology labels: {dict(counts)}')
    require(len(rows) == manifest['samples'], 'Metadata count contradicts manifest')
    for side in counts:
        require(counts[side] == manifest[f'{side}_samples'], 'Ideology counts contradict manifest')
    expected_ids = [f'{side}_{i:06d}' for side in ('left', 'right') for i in range(counts[side])]
    require(ids == expected_ids, 'Metadata ordering contradicts extraction order')
    for row in rows:
        side = row['ideology']
        require(row['sample_id'] == f"{side}_{row['original_row_index']:06d}", 'Sample/source index mismatch')
        require(row['source_file'] == f'{side}_neutral_dedup.jsonl', 'Unexpected original source file')
        require(isinstance(row['original_article'], str) and row['original_article'].strip(), 'Empty original article')
        require(hashlib.sha256(row['original_article'].encode()).hexdigest() == row['original_article_sha256'],
                f'Original body hash mismatch at {row["sample_id"]}')
        require(manifest['original_article_fields'][side] == f'{side}_original.body', 'Wrong original field provenance')
        require(manifest['ignored_rewrite_fields'][side] != manifest['original_article_fields'][side], 'Rewrite used as original')
    readme = readme_path.read_text() if readme_path.exists() else ''
    guarantee = 'Row `i` of every `*_embeddings.npy` is row `i` of `metadata.parquet`.'
    documented_order = guarantee in readme or manifest.get('row_order') == 'metadata.parquet'
    events = memberships_from_rows(rows, event_field)
    if event_field == 'event_ids' and 'story_id' in columns:
        require(all(str(r['story_id']) in e for r, e in zip(rows, events)), 'Representative story absent from memberships')
    audit['dataset'] = {
        'directory': str(data), 'files': sorted(p.name for p in data.iterdir() if p.is_file()),
        'metadata_fields': fields, 'sample_id_field': 'sample_id', 'ideology_field': 'ideology',
        'event_field': event_field, 'article_text_field': 'original_article', 'publisher_field': 'publisher' if 'publisher' in fields else None,
        'rows': len(rows), 'ideology_counts': dict(counts), 'pooling': manifest.get('pooling'),
        'original_field_provenance': manifest['original_article_fields'],
        'original_body_hashes_validated': len(rows), 'original_article_confirmed': True,
        'original_confirmation_scope': 'Extraction manifest + README + stored original-body hashes; spot check separately verifies selected model rows.',
        'extraction_git_commit': manifest.get('git_commit'),
        'extraction_manifest_has_explicit_row_order': 'row_order' in manifest,
        'row_order_record': 'README.md: ' + guarantee if guarantee in readme else manifest.get('row_order'),
        'duplicate_original_bodies': len(rows) - len(set(r['original_article_sha256'] for r in rows)),
    }
    audit['input_sha256'] = {str(p): digest(p) for p in (manifest_path, metadata_path, readme_path) if p.is_file()}
    matrices, models = {}, {}
    dimensions = set()
    for path in sorted(data.glob('*.npy')):
        match = MATRIX_PATTERN.fullmatch(path.name)
        if not match:
            continue
        key = path.name.removesuffix('_embeddings.npy')
        weight = float(match.group(1).replace('_', '.'))
        require(weight not in [v['lambda'] for v in models.values()], 'Duplicate lambda mapping')
        require(key in manifest['checkpoints'], f'Checkpoint mapping missing for {path.name}; update extraction manifest')
        require(key in manifest['shapes'], f'Extraction shape record missing for {path.name}')
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        require(a.ndim == 2 and a.shape[0] == len(rows), f'Wrong dimensions/row count: {path.name}')
        require(list(a.shape) == manifest['shapes'][key], f'Manifest shape mismatch: {path.name}')
        require(a.dtype.kind == 'f' and str(a.dtype) == manifest['dtype'], f'Unexpected dtype: {path.name}')
        nan, posinf, neginf = int(np.isnan(a).sum()), int(np.isposinf(a).sum()), int(np.isneginf(a).sum())
        audit.setdefault('models', {})[key] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'NaNs': nan, '+Inf': posinf, '-Inf': neginf}
        require(nan + posinf + neginf == 0, f'Nonfinite embeddings: {path.name}')
        dimensions.add(a.shape[1])
        require(len(dimensions) == 1 and a.shape[1] == manifest['hidden_dim'], 'Hidden dimensions disagree')
        f64 = np.asarray(a, dtype=np.float64)
        norms = np.linalg.norm(f64, axis=1)
        require(np.all(norms > 1e-8), f'Near-zero vector: {path.name}')
        median = float(np.median(norms))
        extreme = int(((norms > 100 * median) | (norms < median / 100)).sum())
        require(extreme == 0, f'Extremely abnormal norms: {path.name}')
        if 'L2 normalization' in manifest.get('pooling', ''):
            require(np.allclose(norms, 1, atol=1e-5, rtol=1e-5), 'Norms contradict recorded extraction normalization')
        unique_count = len(np.unique(a, axis=0))
        require(unique_count > 1, f'All embedding rows identical: {path.name}')
        explicit_ids = path.with_name(key + '_sample_ids.json')
        if explicit_ids.exists():
            require(json.loads(explicit_ids.read_text()) == ids, f'Explicit sample IDs/order mismatch: {key}')
            order_evidence = str(explicit_ids)
        else:
            require(documented_order, f'No explicit IDs or extraction row-order guarantee for {key}')
            order_evidence = audit['dataset']['row_order_record']
        info = audit['models'][key]
        info.update(value_statistics=describe(f64), norm_statistics=describe(norms), duplicate_rows=len(a)-unique_count,
                    near_zero_rows=int((norms <= 1e-8).sum()), abnormal_norm_rows=extreme,
                    shape_valid=True, metadata_row_count_valid=True, sample_order='PASS: '+order_evidence,
                    original_article_confirmed=True, status='WARN' if unique_count != len(a) else 'PASS')
        if unique_count != len(a):
            audit['warnings'].append(f'{key}: {len(a)-unique_count} duplicate embedding rows')
        audit['input_sha256'][str(path)] = digest(path)
        matrices[key] = a
        models[key] = {'lambda': weight, 'file': str(path), 'checkpoint': manifest['checkpoints'][key]}
    require(bool(matrices), 'No valid lambda embedding files')
    audit['model_mapping'] = models
    audit['corresponding_row_sanity_checks'] = {}
    for first, second in itertools.combinations(matrices, 2):
        a, b = [np.asarray(matrices[k], dtype=np.float64) for k in (first, second)]
        near = bool(np.allclose(a, b, atol=1e-7, rtol=1e-6))
        comparison = {'exactly_identical': bool(np.array_equal(a, b)), 'nearly_identical': near,
                      'mean_cosine': float(np.mean(np.sum(a*b, axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))),
                      'mean_L2_distance': float(np.linalg.norm(a-b, axis=1).mean()),
                      'max_absolute_difference': float(np.abs(a-b).max())}
        audit['corresponding_row_sanity_checks'][f'{second}_vs_{first}'] = comparison
        require(not near, f'Suspicious identical/nearly identical matrices: {first}, {second}')
    positives, negatives, event_stats = build_triplet_pools(rows, events)
    audit['metadata_integrity'] = event_stats
    require(event_stats['potential_directed_triplets'] > 0, 'No valid metadata triplets')
    return manifest, rows, events, matrices, models, positives, negatives


def checkpoint_records(models, cache_dir, audit):
    from huggingface_hub import snapshot_download
    paths = {}
    audit['checkpoint_records'] = {}
    for key, model in models.items():
        spec = model['checkpoint']
        require('@' in spec, f'Unpinned checkpoint mapping: {spec}')
        location, revision = spec.rsplit('@', 1)
        parts = location.split('/')
        require(len(parts) >= 3 and re.fullmatch('[a-f0-9]{40}', revision), f'Unrecognized checkpoint record: {spec}')
        repo, folder = '/'.join(parts[:2]), '/'.join(parts[2:])
        root = Path(snapshot_download(repo, revision=revision, cache_dir=str(cache_dir), max_workers=2,
                    allow_patterns=[folder+'/config.json', folder+'/joint_state.json', folder+'/tokenizer*', folder+'/metrics/training_config.json']))
        path = root / folder
        state = json.loads((path/'joint_state.json').read_text())
        config = json.loads((path/'config.json').read_text())
        require(float(state['triplet_lambda']) == model['lambda'], f'Wrong checkpoint lambda: {key}')
        require(config['model_type'] == 'led', f'Unexpected model architecture: {key}')
        paths[key] = dict(path=path, repo=repo, revision=revision, folder=folder)
        audit['checkpoint_records'][key] = {'state': state, 'state_sha256': digest(path/'joint_state.json'),
            'state_source': f'{repo}/{folder}/joint_state.json@{revision}',
            'config_sha256': digest(path/'config.json'), 'hidden_dim': config['d_model']}
    return paths


def historical_helpers(commit, directory):
    files = ['src/data/__init__.py', 'src/data/baseline_dataset.py', 'src/data/triplet_dataset.py',
             'src/models/__init__.py', 'src/models/led_baseline.py', 'src/models/led_triplet.py', 'src/models/pooling.py']
    hashes = {}
    for relative in files:
        contents = git_source(commit, relative)
        path = Path(directory)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        hashes[relative] = hashlib.sha256(contents).hexdigest()
    (Path(directory)/'src/__init__.py').write_text('')
    sys.path.insert(0, directory)
    return hashes


def spot_check(manifest, rows, matrices, paths, args, audit):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    required = ('git_commit', 'tokenizer_sha256', 'max_input_length', 'attention_window', 'global_attention', 'pooling')
    if not all(k in manifest for k in required):
        audit['spot_check'] = {'status': 'SKIPPED', 'reason': 'Missing exact extraction settings'}
        return
    require(manifest['global_attention'] == 'first token only', 'Unknown global-attention behavior')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'Requested CUDA is unavailable')
    indices = np.sort(np.random.default_rng(args.seed).choice(len(rows), min(args.spot_check_size, len(rows)), replace=False))
    audit['spot_check'] = {'status': 'RUNNING', 'row_indices': indices.tolist(),
        'sample_ids': [rows[i]['sample_id'] for i in indices], 'seed': args.seed,
        'device': str(device), 'max_abs_tolerance': 2e-5, 'min_cosine_tolerance': .99999,
        'models': {}, 'scope': 'Selected original articles only; no full re-extraction',
        'runtime_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'huggingface-hub')}}
    with tempfile.TemporaryDirectory(prefix='anlp-training-source-') as directory:
        try:
            audit['historical_source_sha256'] = historical_helpers(manifest['git_commit'], directory)
        except subprocess.CalledProcessError:
            audit['spot_check'].update(status='SKIPPED', reason='Recorded training source commit unavailable')
            return
        from src.data.triplet_dataset import ArticleRef, format_article
        from src.models.led_triplet import load_led_triplet
        for key, details in paths.items():
            snapshot_download(details['repo'], revision=details['revision'], cache_dir=str(args.cache_dir),
                              allow_patterns=[details['folder']+'/model.safetensors'], max_workers=1)
            path = details['path']
            require(digest(path/'tokenizer.json') == manifest['tokenizer_sha256'], f'Tokenizer fingerprint mismatch: {key}')
            state = audit['checkpoint_records'][key]['state']
            for field in ('max_input_length', 'attention_window'):
                require(state[field] == manifest[field], f'Extraction setting mismatch: {key} {field}')
            require(state['pooling'] == 'float32_masked_mean_l2_normalized', 'Unknown training pooling')
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = load_led_triplet(path, attention_window=manifest['attention_window']).to(device).eval()
            require(model.config.d_model == matrices[key].shape[1], 'Checkpoint hidden dimension mismatch')
            require(manifest['max_input_length'] <= min(tokenizer.model_max_length, model.config.max_encoder_position_embeddings),
                    'Input limit exceeds tokenizer/model capacity')
            texts = [format_article(ArticleRef(event_id=str(rows[i]['story_id']), article_id=rows[i]['article_id'],
                ideology=rows[i]['ideology'], split=rows[i].get('split') or '', title=rows[i].get('title') or '',
                text=rows[i]['original_article']), tokenizer) for i in indices]
            regenerated = []
            with torch.inference_mode():
                for start in range(0, len(indices), 4):
                    encoded = tokenizer(texts[start:start+4], truncation=True, max_length=manifest['max_input_length'],
                                        padding=True, return_tensors='pt')
                    lengths = encoded['attention_mask'].sum(dim=1).tolist()
                    require(lengths == [rows[i]['effective_input_length'] for i in indices[start:start+4]], 'Token length mismatch')
                    batch = {k:v.to(device) for k,v in encoded.items() if k in ('input_ids', 'attention_mask')}
                    global_mask = torch.zeros_like(batch['attention_mask']); global_mask[:,0] = 1
                    pooled, _ = model._encode(batch['input_ids'], batch['attention_mask'], global_mask)
                    regenerated.append(pooled.cpu().numpy())
            actual = np.concatenate(regenerated).astype(np.float64)
            saved = np.asarray(matrices[key][indices], dtype=np.float64)
            error = np.abs(actual-saved)
            cosine = np.sum(actual*saved,axis=1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(saved,axis=1))
            result = {'mean_cosine_similarity': float(cosine.mean()), 'min_cosine_similarity': float(cosine.min()),
                      'max_absolute_difference': float(error.max()), 'mean_absolute_difference': float(error.mean()),
                      'per_row': [{'sample_id':rows[i]['sample_id'], 'cosine':float(cosine[k]), 'max_absolute_difference':float(error[k].max())}
                                  for k,i in enumerate(indices)]}
            audit['spot_check']['models'][key] = result
            require(np.isfinite(actual).all() and error.max() <= 2e-5 and cosine.min() >= .99999,
                    f'Re-extraction disagrees with saved rows: {key}')
            result['status'] = 'PASS'
            print(f'Spot check {key}: max_abs={error.max():.3g}, mean cosine={cosine.mean():.10f}', flush=True)
            del model, pooled
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        sys.path.remove(directory)
    audit['spot_check']['status'] = 'PASS'


def verify_triplet(triplet, rows, events):
    a, p, n = triplet
    require(len({a,p,n}) == 3, 'Repeated article within triplet')
    require(rows[a]['ideology'] != rows[p]['ideology'] and bool(events[a] & events[p]), 'Invalid positive')
    require(rows[a]['ideology'] == rows[n]['ideology'] and not events[a] & events[n], 'Invalid negative')


def sample_triplets(positives, negatives, count, seed):
    """Prefer globally unused articles; fall back to least-used articles if needed."""
    rng = np.random.default_rng(seed)
    anchors = np.array([a for a in positives if len(positives[a]) and len(negatives[a])], dtype=np.int64)
    total = sum(len(positives[a])*len(negatives[a]) for a in anchors)
    target = min(count, total)
    if total <= count:
        return np.array([(a,p,n) for a in anchors for p in positives[a] for n in negatives[a]], dtype=np.int64)
    usage = np.zeros(len(positives), dtype=np.int64)
    chosen, seen = [], set()
    def add(a,p,n):
        chosen.append((int(a),int(p),int(n))); seen.add((int(a),int(p),int(n)))
        usage[[a,p,n]] += 1
    # First pass constructs wholly disjoint triplets whenever possible.
    for a in rng.permutation(anchors):
        if usage[a]:
            continue
        ps, ns = positives[a][usage[positives[a]] == 0], negatives[a][usage[negatives[a]] == 0]
        if len(ps) and len(ns):
            add(a,rng.choice(ps),rng.choice(ns))
            if len(chosen) == target:
                return np.asarray(chosen)
    while len(chosen) < target:
        added = False
        order = rng.permutation(anchors)
        order = order[np.argsort(usage[order], kind='stable')]
        for a in order:
            ps = rng.permutation(positives[a]); ps = ps[np.argsort(usage[ps],kind='stable')]
            for p in ps:
                ns = np.array([n for n in negatives[a] if (int(a),int(p),int(n)) not in seen])
                if len(ns):
                    add(a,p,rng.choice(ns[usage[ns] == usage[ns].min()])); added=True
                    break
            if len(chosen) == target:
                break
        require(added, 'Triplet sampler exhausted before reaching requested count')
    return np.asarray(chosen)


def fixed_triplets(args, rows, events, positives, negatives, metadata_hash):
    path = args.results_dir/'sampled_triplets.csv'
    previous_manifest = args.results_dir/'triplet_eval_manifest.json'
    target = min(args.num_triplets, sum(len(positives[a])*len(negatives[a]) for a in positives))
    lookup = {r['sample_id']:i for i,r in enumerate(rows)}
    if path.exists() and not args.resample_triplets:
        require(previous_manifest.is_file(), 'Existing triplets have no sampling manifest; use --resample-triplets')
        old = json.loads(previous_manifest.read_text())
        require(old['metadata_sha256'] == metadata_hash and old['seed'] == args.seed and old['triplet_count'] == target,
                'Saved sampling settings/data differ; use --resample-triplets to deliberately replace')
        require(digest(path) == old['sampled_triplets_sha256'], 'Saved triplet CSV checksum mismatch')
        with path.open() as f:
            saved = list(csv.DictReader(f))
        triplets = np.array([[lookup[r[k]] for k in ('anchor_id','positive_id','negative_id')] for r in saved])
        require(len(triplets) == target, 'Saved triplet count mismatch')
        reused = True
    else:
        triplets = sample_triplets(positives, negatives, args.num_triplets, args.seed)
        reused = False
    require(len(set(map(tuple,triplets))) == len(triplets), 'Duplicate triplets')
    for t in triplets:
        verify_triplet(t, rows, events)
    if not reused:
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['triplet_id','anchor_id','positive_id','negative_id','shared_positive_events'])
            for k,(a,p,n) in enumerate(triplets):
                writer.writerow([f'triplet_{k:06d}', rows[a]['sample_id'], rows[p]['sample_id'], rows[n]['sample_id'],
                                 json.dumps(sorted(events[a]&events[p]))])
    return triplets, reused


def bootstrap_mean_ci(values, draws):
    return [float(x) for x in np.quantile(values[draws].mean(axis=1), [.025,.975])]


def triplet_values(z, triplets, margin):
    a,p,n = triplets.T
    pos_delta, neg_delta = z[a]-z[p], z[a]-z[n]
    dpos, dneg = np.sum(pos_delta**2,axis=1), np.sum(neg_delta**2,axis=1)
    gap = dneg-dpos
    values = {'d_pos': dpos, 'd_neg': dneg, 'gap': gap}
    if margin is not None:
        values['margin_gap'] = gap-margin
        # Historical TripletMarginLoss(p=2, eps=1e-6) uses unsquared norms.
        values['training_margin_gap'] = np.linalg.norm(neg_delta+1e-6,axis=1)-np.linalg.norm(pos_delta+1e-6,axis=1)-margin
    return values


def plot_triplets(results, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    x = np.arange(len(results)); ticks = [f"{r['lambda']:g}" for r in results]
    fig, ax = plt.subplots(figsize=(5.5,3.6),layout='constrained')
    for field, shift, label, color in [('d_pos',-.08,'Positive distance','#245a81'),('d_neg',.08,'Negative distance','#bf6b24')]:
        means = np.array([r['mean_'+field] for r in results])
        lo = np.array([r[field+'_ci_low'] for r in results]); hi = np.array([r[field+'_ci_high'] for r in results])
        ax.errorbar(x+shift,means,yerr=np.array([means-lo,hi-means]),fmt='o',capsize=4,label=label,color=color)
    ax.set(xticks=x,xticklabels=ticks,xlabel='Triplet-loss weight λ',ylabel='Mean squared L2 distance',xlim=(-.45,len(x)-.55))
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'): fig.savefig(directory/f'triplet_distances.{ext}',dpi=300)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(5.5,3.6),layout='constrained')
    means=np.array([r['mean_separation_gap'] for r in results])
    lo=np.array([r['gap_ci_low'] for r in results]); hi=np.array([r['gap_ci_high'] for r in results])
    ax.errorbar(x,means,yerr=[means-lo,hi-means],fmt='o',capsize=4,color='#245a81')
    ax.axhline(0,color='gray',lw=.8,ls='--')
    ax.set(xticks=x,xticklabels=ticks,xlabel='Triplet-loss weight λ',ylabel='Mean separation gap (d_neg − d_pos) ↑',xlim=(-.4,len(x)-.6))
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'): fig.savefig(directory/f'triplet_separation_gap.{ext}',dpi=300)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('embedding_data_backup'))
    parser.add_argument('--results-dir',type=Path,default=Path('results'))
    parser.add_argument('--plots-dir',type=Path,default=Path('plots'))
    parser.add_argument('--num-triplets',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    parser.add_argument('--spot-check-size',type=int,default=10)
    parser.add_argument('--skip-spot-check',action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--cache-dir',type=Path,default=Path('/tmp/anlp-triplet-hf-cache'))
    parser.add_argument('--resample-triplets',action='store_true')
    args=parser.parse_args(argv)
    require(args.num_triplets>0 and args.bootstrap_repetitions>0 and args.spot_check_size>0,'Counts must be positive')
    args.results_dir.mkdir(parents=True,exist_ok=True)
    audit={'status':'RUNNING','phase2_started':False,'warnings':[], 'timestamp':datetime.now(timezone.utc).isoformat()}
    audit_path=args.results_dir/'embedding_audit.json'
    try:
        manifest,rows,events,matrices,models,positives,negatives=audit_inputs(args.data_dir,audit)
        # Checkpoint metadata establishes the actual lambda/margin without inference.
        paths=checkpoint_records(models,args.cache_dir,audit)
        if args.skip_spot_check:
            audit['spot_check']={'status':'SKIPPED','reason':'Explicit --skip-spot-check'}
        else:
            spot_check(manifest,rows,matrices,paths,args,audit)
        audit['margin']={'values':{k:r['state'].get('triplet_margin') for k,r in audit['checkpoint_records'].items()},
                         'training_distance':'unsquared L2; TripletMarginLoss(p=2, eps=1e-6)',
                         'requested_distance':'squared L2',
                         'note':'Requested squared-margin satisfaction is not the training loss criterion; both are reported separately.'}
        # Establish distance semantics from the recorded repository code, never assume.
        try:
            source=git_source(manifest['git_commit'],'src/models/led_triplet.py').decode()
            semantics_known='nn.TripletMarginLoss(' in source and 'margin=margin, p=2, eps=1e-6' in source
        except subprocess.CalledProcessError:
            semantics_known=False
        audit['margin']['training_semantics_verified']=semantics_known
        for path,expected in audit['input_sha256'].items():
            require(digest(path)==expected,'Input changed during audit: '+path)
        audit['status']='PASS'
        write_json(audit_path,audit)
    except Exception as exc:
        audit.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        write_json(audit_path,audit)
        print('EMBEDDING AUDIT: FAIL — '+str(exc),file=sys.stderr)
        return 1
    print('EMBEDDING AUDIT: PASS',flush=True)
    print('| model | shape | NaNs | Inf | duplicate rows | mean norm | original article | row order | spot check |')
    print('|---|---|---:|---:|---:|---:|---|---|---|')
    for key,info in audit['models'].items():
        print(f"| {key} | {info['shape']} | {info['NaNs']} | {info['+Inf']+info['-Inf']} | "
              f"{info['duplicate_rows']} | {info['norm_statistics']['mean']:.9f} | PASS | PASS (documented) | "
              f"{audit['spot_check']['status']} |",flush=True)
    # Everything below is gated by PASS.
    metadata_hash=digest(args.data_dir/'metadata.parquet')
    triplets,reused=fixed_triplets(args,rows,events,positives,negatives,metadata_hash)
    draws=np.random.default_rng(args.seed+1).integers(len(triplets),size=(args.bootstrap_repetitions,len(triplets)))
    results,per_model=[],{}
    for key,model in sorted(models.items(),key=lambda item:item[1]['lambda']):
        z=np.array(matrices[key],dtype=np.float64,copy=True)
        z/=np.linalg.norm(z,axis=1,keepdims=True)
        require(np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-12),'Normalization failed')
        margin=audit['margin']['values'][key]
        values=triplet_values(z,triplets,margin)
        per_model[key]=values
        row={'lambda':model['lambda'],'num_triplets':len(triplets),
             'mean_d_pos':float(values['d_pos'].mean()),'mean_d_neg':float(values['d_neg'].mean()),
             'mean_separation_gap':float(values['gap'].mean()),'median_separation_gap':float(np.median(values['gap'])),
             'ordering_success_pct':float(100*np.mean(values['gap']>0)), 'margin':margin,
             'mean_requested_margin_gap':float(values['margin_gap'].mean()) if margin is not None else None,
             'requested_squared_margin_satisfaction_pct':float(100*np.mean(values['margin_gap']>0)) if margin is not None else None,
             'actual_training_margin_satisfaction_pct':float(100*np.mean(values['training_margin_gap']>0)) if margin is not None and semantics_known else None}
        for field in ('d_pos','d_neg','gap'):
            row[field+'_ci_low'],row[field+'_ci_high']=bootstrap_mean_ci(values[field],draws)
        results.append(row)
    comparisons=[]
    baseline=next((k for k,m in models.items() if m['lambda']==0),None)
    if baseline:
        for key in per_model:
            if key==baseline: continue
            delta=per_model[key]['gap']-per_model[baseline]['gap']
            comparisons.append({'comparison':f'{key} - {baseline}','mean_paired_gap_change':float(delta.mean()),
                                'bootstrap_95_ci':bootstrap_mean_ci(delta,draws),'num_triplets':len(triplets)})
    with (args.results_dir/'triplet_separation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    write_json(args.results_dir/'triplet_separation.json',{'models':results,'paired_comparisons':comparisons})
    # Per-triplet numbers allow independent paired analysis without recomputation.
    with (args.results_dir/'triplet_per_sample.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['triplet_id','lambda','d_pos','d_neg','separation_gap'])
        for key,values in per_model.items():
            for i in range(len(triplets)):
                writer.writerow([f'triplet_{i:06d}',models[key]['lambda'],values['d_pos'][i],values['d_neg'][i],values['gap'][i]])
    plot_triplets(results,args.plots_dir)
    for path,expected in audit['input_sha256'].items():
        require(digest(path)==expected,'Input changed during evaluation: '+path)
    git=subprocess.run(['git','rev-parse','--verify','HEAD'],capture_output=True,text=True,cwd=ROOT)
    usage=Counter(triplets.flatten().tolist())
    evaluation={'timestamp':datetime.now(timezone.utc).isoformat(),'git_commit':git.stdout.strip() if git.returncode==0 else None,
        'script_sha256':digest(Path(__file__)),'seed':args.seed,'triplet_count':len(triplets),
        'potential_directed_triplets':audit['metadata_integrity']['potential_directed_triplets'],
        'metadata_sha256':metadata_hash,'sampled_triplets_file':str(args.results_dir/'sampled_triplets.csv'),
        'sampled_triplets_sha256':digest(args.results_dir/'sampled_triplets.csv'),'reused_saved_triplets':reused,
        'sample_id_field':'sample_id','ideology_field':'ideology','event_field':audit['dataset']['event_field'],
        'sampling':'Seeded greedy preference for globally unused articles, then least-used alternatives; unique directed tuples; not claimed uniform over all possible triplets',
        'negative_rule':'Same ideology as anchor and disjoint event memberships from anchor',
        'positive_rule':'Different ideology from anchor and at least one shared event membership',
        'article_usage':{'unique_articles':len(usage),'max_uses':max(usage.values()),'usage_histogram':dict(Counter(usage.values()))},
        'normalization':'float64 L2-normalized copies, stored files untouched',
        'distances':'d_pos=||z_a-z_p||^2, d_neg=||z_a-z_n||^2; gap=d_neg-d_pos',
        'margin':audit['margin'],'models':models,'input_sha256':audit['input_sha256'],
        'audit_file':str(audit_path),'audit_status':'PASS',
        'bootstrap':{'seed':args.seed+1,'repetitions':args.bootstrap_repetitions,'unit':'triplet','interval':'95% percentile',
            'same_draws_across_models':True,'limitation':'Conditional on this sampled triplet set; shared events can cause dependence; excludes training-run uncertainty'},
        'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','pyarrow','matplotlib')},
        'neutral_text_used':False}
    write_json(args.results_dir/'triplet_eval_manifest.json',evaluation)
    audit['phase2_started']=True
    write_json(audit_path,audit)
    print(json.dumps({'results':results,'paired_comparisons':comparisons},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
