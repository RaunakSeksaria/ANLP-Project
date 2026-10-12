<<<<<<< HEAD
# ANLP-Project
=======
# Original-article alignment and uniformity

This evaluation reads the existing `embedding_data_backup/` only. It does not load
models or article text, or modify stored embeddings. The supplied backup contains
lambda 0 and lambda 0.35; lambda 0.1 is not present and is not evaluated.

```bash
conda activate torch_env
python -m pip install -r requirements-evaluation.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_alignment_uniformity.py \
  --data-dir embedding_data_backup --sample-size 2000 \
  --max-alignment-pairs 2000 --seed 42 --bootstrap-repetitions 500
```

Add `--wandb` to request optional logging to `ANLP-embedding-evals`. Missing
credentials/package or logging errors do not prevent local results. W&B is not a
required dependency. Outputs go to `results/` and `plots/` by default.

## Definitions and samples

Embeddings are copied into float64 and L2-normalized independently per model.
Uniformity uses one seed-42 sample of 1,000 left and 1,000 right articles, with
all 1,999,000 unique unordered pairs. Its score is
`log(mean(exp(-2 * squared_distance)))`.

Alignment is the mean squared distance between normalized positive pairs.
The metadata's `event_ids` list retains multiple event memberships for deduplicated
articles; `story_id` is the representative story and is included in that list.
Articles with different ideologies and any shared event ID form a positive pair.
An article pair sharing multiple events is counted once. Up to 2,000 pairs are
sampled with seed 42 from the full population, independently of the uniformity
sample. Both models use identical samples and pairs, explicitly recorded in the
manifest. Lower scores correspond to closer positives (alignment) and more spread
out representations (uniformity). These metrics do not establish neutrality or
generation quality.

## Confidence intervals and provenance

The plots and tables use 500 deterministic percentile bootstrap replicates.
Uniformity resamples articles within ideology strata, recomputing the pair mean
using article multiplicities, including zero-distance duplicate copies. It does
not treat the approximately two million distances as independent observations.
Alignment resamples positive pairs. That simple interval is conditional on the
selected pair population and does not account for dependence from shared articles
or events; it should not be presented as an event-cluster-adjusted interval.
Both models use the same bootstrap draws. Intervals describe sampling variability,
not variation across training runs.

The source README declares one shared row order for all arrays. The evaluator
checks that contract against metadata sample IDs/source indices, manifest counts,
and matrix shapes, and records SHA256 input hashes. Since the `.npy` matrices do
not carry their own row IDs, an arbitrary internal matrix permutation cannot be
independently detected. This limitation is explicitly recorded in the evaluation
manifest. No claims of stronger per-matrix identity verification are made.

Only identity/ideology/event/publisher metadata columns are read; original article
bodies and any neutral or generated text are excluded. Inputs are checked for
finite values and nonzero norms, and normalized copies for unit norms. Each
selected positive pair is checked for shared event membership and different
ideology. Five examples are printed without article bodies.

Run the small analytical checks with:

```bash
python -m unittest discover -s tests -v
```
>>>>>>> embedding-evals
