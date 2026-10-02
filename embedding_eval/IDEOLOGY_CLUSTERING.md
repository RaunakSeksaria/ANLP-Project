# Ideology clustering on original-article embeddings

Run from the repository root (`uv sync` installs every requirement):

```bash
uv run python embedding_eval/scripts/evaluate_ideology_clustering.py \
  --data-dir embedding_data --sample-size 2000 --seed 42
```

The script discovers the available `lambda_*_embeddings.npy` matrices and requires a matching checkpoint and shape entry in the extraction manifest. It reads only `sample_id` and `ideology` from `metadata.parquet`; source article text, neutral rewrites, `event_ids`, and publisher fields are not used. The extraction README states that row `i` in every matrix corresponds to row `i` in metadata. The script also checks the metadata's documented left-then-right ID sequence, row counts, dimensions, dtypes, finite values, and nonzero norms. The `.npy` files have no embedded IDs, so their row order is established by the extraction record rather than independently encoded per array.

For a balanced sample, the script selects the same metadata indices for all models, saves them in `results/ideology_sample_ids.csv`, and L2-normalizes in memory. It fits K-means with `k=2`, `random_state=42`, and `n_init=20` without labels. ARI and NMI compare its arbitrary cluster IDs to the true ideology labels. Silhouette uses the true ideology labels and cosine distance. Centroid distance is `1 - cosine(left_centroid, right_centroid)` on normalized embeddings. Each model gets a separate two-component PCA fit for qualitative visualization only. PCA panels use common axis limits, although independently fitted components need not represent the same directions.

The fixed-seed K-means result and its inertia are reported; this run does not estimate sampling uncertainty. Results, the sampled IDs, input file hashes, settings, and PCA plots are saved under `embedding_eval/results/` and `embedding_eval/plots/`. Add `--wandb` to opt into W&B logging. Without `WANDB_API_KEY`, an installed W&B package saves an offline run for later syncing. The script remains usable without W&B or a login.
