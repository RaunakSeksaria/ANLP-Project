# training/

LED-base-16384 (`allenai/led-base-16384`) fine-tuned to rewrite a partisan
article neutrally, with an optional event triplet loss on the encoder:

    L = L_gen + λ · max(0, d(a,p) − d(a,n) + 0.2)

where d is the Euclidean distance between L2-normalised masked-mean encoder
states. The anchor is the input article, the positive is the opposite-side
article about the same event, and the negative is a different-event article from
the anchor's side (ideology-anchored) or the anchor's own publisher
(publisher-anchored). λ = 0 never uses the negative, so it is the control for
both variants.

Trained checkpoints are on Hugging Face: `RaunakSeksaria/anlp-neutral-led`,
one folder per `<experiment>/<run>`. Any script that takes `--checkpoint`
accepts a local directory or such a folder name (`src/hub.py` downloads it).

## Data

Build every split first (from the repo root): `splits/build.sh`. It writes

| `--data-dir` | experiment | train / val / test |
| --- | --- | --- |
| `dataset/neutral_articles_10k_prepared` | E1: article → the event's multi-source neutral article | 16,000 / 2,000 / 2,000 |
| `dataset/side_neutral_full/prepared` | E2: article → its own neutral rewrite, ideology negatives | 12,484 / 1,778 / 1,775 |
| `dataset/side_neutral_full_publisher/prepared` | E2 with publisher negatives | 12,484 / 1,778 / 1,775 |

and checks each file against `splits/manifest.json`.

The scripts behind it, in order: `make_pairable_subset.py` (keeps articles that
share an event with the other side), `prepare_side_neutral.py` (E2 pairs and
triplets), `make_negative_variants.py` (rewrites only the negatives, e.g.
`--policy publisher`), `prepare_data.py` (E1). `validate_prepared_data.py`
re-checks a prepared directory.

## Train and evaluate

Run from the repo root. All published runs used these settings; only λ and
`--data-dir` change.

```bash
uv run python training/src/scripts/train_joint.py \
    --data-dir dataset/side_neutral_full/prepared --output-dir runs/lambda035 \
    --model-name allenai/led-base-16384 \
    --train-batch-size 1 --eval-batch-size 4 --gradient-accumulation-steps 8 \
    --max-epochs 4 --early-stopping-patience 2 \
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10 --max-grad-norm 1.0 \
    --max-input-length 512 --max-target-length 512 --generation-max-new-tokens 512 \
    --triplet-lambda 0.35 --triplet-margin 0.2 --attention-window 512 \
    --seed 42 --device cuda --precision fp32 --disable-wandb

uv run python training/src/scripts/evaluate_joint.py \
    --data-dir dataset/side_neutral_full/prepared --checkpoint runs/lambda035/best \
    --output-dir runs/lambda035/test_eval --split test --eval-batch-size 4

uv run python training/src/scripts/generate_article.py \
    --checkpoint side-neutral-full/lambda035 --title "..." --article "..."
```

- `--dry-run` on `train_joint.py` checks the data and settings without loading
  the model.
- W&B logging: `uv sync --extra tracking`, set `WANDB_API_KEY`, and drop
  `--disable-wandb` (see `--wandb-entity`, `--wandb-project`, `--wandb-run-name`).
- A λ > 0 run takes about 9 h on a Kaggle T4 including evaluation, and about
  7 h of training on an RTX 2080 Ti (12,484 triplets, 4 epochs, fp32); λ = 0 is
  about twice as fast. Launchers for Ada and Kaggle are in `scripts/`.
- `train_baseline.py` is the generation-only trainer without the triplet
  machinery; `overfit_baseline.py` is a sanity check that the model can memorise
  a few examples.

Outputs of a run: `best/` (model, tokenizer, `joint_state.json`), `last/`,
`training_history.json` (per epoch), `step_metrics.jsonl` (per optimizer step),
`run_summary.json`, `validation_metrics_best.json` and prediction files.

## Code

| path | contents |
| --- | --- |
| `src/data/` | `baseline_dataset.py` (input/target pairs; input is `title </s> body`), `triplet_dataset.py` (resolves and validates anchor/positive/negative) |
| `src/models/` | `led_triplet.py` (generation + triplet loss), `pooling.py` (masked mean, L2 norm) |
| `src/training/` | `train_joint.py`, `train_baseline.py`, `overfit_baseline.py` (generation helpers) |
| `src/scripts/` | the CLIs above |
| `tests/` | `uv sync --extra dev`, then `cd training && uv run python -m pytest tests` |
