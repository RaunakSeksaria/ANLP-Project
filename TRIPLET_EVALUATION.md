# Embedding audit and triplet separation

Run from the repository root in `torch_env`:

```bash
HF_HUB_DISABLE_XET=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/verify_and_evaluate_triplets.py \
  --data-dir embedding_data_backup --num-triplets 2000 --seed 42
```

Core packages are in `requirements-evaluation.txt`. The default ten-article
re-extraction spot check additionally needs `requirements-triplet-spotcheck.txt`.
`--device auto` uses CUDA when available and otherwise CPU; `--device cuda`
requires a GPU. Models are loaded only for the ten selected original articles.
No full embedding extraction or other representation metrics are run.

The script discovers `lambda_*_embeddings.npy` files. Each must have a matching
checkpoint and shape entry in the source manifest. A new lambda matrix must follow
the documented common metadata order; an optional `lambda_*_sample_ids.json`
sidecar is compared directly if provided. No model weights are substituted.

## Audit gate

The script checks metadata uniqueness, source row order, original-body SHA256
hashes and original-field provenance; event memberships and ideology; array
shapes, dtype, finite values, duplicate vectors, near-zero or extreme norms; and
cross-model equality/corresponding-row similarity. It reads only the original
article column for text verification and re-extraction. It never reads neutral
rewrite or generated text columns.

The extraction README explicitly guarantees shared row order. The source JSON
manifest itself lacks an explicit row-order field, and the arrays carry no IDs.
The audit combines the README guarantee with manifest dimensions, deterministic
metadata IDs/source indices, and ten random re-extracted rows per model. This is
strong provenance and sampled identity verification, not proof against arbitrary
permutation of every unchecked row. The audit records that scope explicitly.

Pinned HF checkpoint states verify the lambda mapping and training settings.
The spot check imports the exact pooling, formatting, and LED loader helpers from
the training commit recorded in the extraction manifest, using a temporary source
directory. It uses `eval()`, `inference_mode()`, encoder-only inference, original
title + separator + body, the recorded input limit, and first-token global
attention. It checks effective token lengths and compares regenerated vectors
with stored rows (maximum absolute error ≤ 2e-5; minimum cosine ≥ 0.99999).

`--skip-spot-check` explicitly skips only re-extraction. Checkpoint state files
are still read to validate the checkpoint mapping and margin. If exact historical
source/settings are missing, the stronger check is recorded as skipped; known
mismatches fail the audit. Structural or identity failures write
`results/embedding_audit.json` with `FAIL` and exit before computing separation.
A passed audit is written before Phase 2 begins. Original input hashes are checked
again after evaluation; stored arrays remain untouched.

## Triplets and reuse

A positive has different ideology and shares at least one `event_ids` membership
with its anchor. A negative has the anchor's ideology and has **no event membership
in common with the anchor**. This prevents another membership of a deduplicated
article from creating a false negative. Distinct ordered article tuples are
counted once even when multiple events connect the same positive pair.

The sampler uses seed 42 and prefers wholly unused articles, then less-used
articles when needed. It is not uniform over all possible triplets. The exact
same triplets are used for every matrix. `results/sampled_triplets.csv` is reused
on later runs, including after lambda 0.1 is added, provided its checksum, metadata
hash, seed, and requested size match. Use `--resample-triplets` only to deliberately
replace that set.

Distances use float64 L2-normalized copies:

- `d_pos = ||z_anchor - z_positive||²`
- `d_neg = ||z_anchor - z_negative||²`
- `gap = d_neg - d_pos`
- Ordering success: `gap > 0`.

## Margin conventions

The checkpoint states record margin **0.2**. Historical training code uses
`TripletMarginLoss(margin=0.2, p=2, eps=1e-6)`, which operates on **unsquared** L2
norms. The requested squared-distance threshold and actual training criterion
are therefore reported separately:

- `requested_squared_margin_satisfaction_pct`: `d_neg > d_pos + 0.2`.
- `actual_training_margin_satisfaction_pct`: the unsquared negative norm exceeds
  the unsquared positive norm by 0.2, including the loss's epsilon convention.

The first is the requested metric; it must not be described as the exact training
loss satisfaction rate. The lambda-zero checkpoint records the same margin
configuration, but has no triplet-loss contribution during training.

## Uncertainty and outputs

The script uses 2,000 percentile bootstrap replicates over triplets, seed 43,
with identical bootstrap draws across models. Paired gap changes are bootstrapped
from per-triplet differences against lambda 0. Intervals are conditional on this
triplet sample and do not incorporate training-run uncertainty. Shared events can
still create dependence, even if no articles are reused.

Outputs are `embedding_audit.json`, `triplet_separation.csv`,
`triplet_separation.json`, `sampled_triplets.csv`, `triplet_eval_manifest.json`,
and `triplet_per_sample.csv` in `results/`. JSON contains paired comparisons;
per-sample CSV permits independent recomputation. Only two new figures are made,
each as PNG and PDF: `plots/triplet_distances` and `plots/triplet_separation_gap`.
Existing results from other experiments are not changed.

Analytical and failure-gate tests:

```bash
python -m unittest discover -s tests -p 'test_triplet_evaluation.py' -v
```
