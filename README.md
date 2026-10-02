# Event-Aware Representation Learning for Neutral News

IIIT Hyderabad ANLP project (Thogata, Lalwani, Agrawal, Seksaria). An LED
encoder–decoder rewrites a partisan news article neutrally, and an event triplet
loss on its encoder pulls reports of the same event together and pushes apart
different events from the same outlet or ideology. This branch holds the code
behind the mid-evaluation.

| | |
| --- | --- |
| Datasets | https://huggingface.co/datasets/RaunakSeksaria/anlp-project-datasets (`reframing_left`, `reframing_right`, `neutralised_across_articles`) |
| Checkpoints | https://huggingface.co/RaunakSeksaria/anlp-neutral-led (one folder per `<experiment>/<run>`) |
| Training logs | https://forge.coreweave.com/wandb/manasagrawal206-iiit-hyderabad/anlp-project |

## Layout

| folder | stage |
| --- | --- |
| `data_generation/` | Building the datasets: BigNewsAlign events, splits, Nemotron-generated neutral targets |
| `splits/` | The exact train/validation/test splits: per-split example ids, SHA-256 of every prepared file, and `build.sh` to rebuild them from Hugging Face |
| `training/` | LED training with the triplet loss, test evaluation, generation |
| `embedding_eval/` | Embedding-level metrics: alignment and uniformity, triplet separation, ideology clustering |
| `generation_eval/` | Article-level metrics on generated text: political-bias classifiers, tonality, zero-shot labelling, QA faithfulness |
| `scripts/` | Launchers (Ada, Kaggle) and the Hugging Face upload job |

Each folder has its own README.

## Quick start

Python 3.14 with [uv](https://docs.astral.sh/uv/); torch comes from the CUDA 12.8 index.

```bash
uv sync                          # add --extra dev for tests, --extra tracking for W&B
cp .env.example .env             # HF_TOKEN is needed while the HF repos are private
splits/build.sh                  # dataset/: the E1 and E2 splits, checked against splits/manifest.json

uv run python training/src/scripts/evaluate_joint.py \
    --data-dir dataset/side_neutral_full/prepared \
    --checkpoint side-neutral-full/lambda035 \
    --output-dir runs/eval_lambda035 --split test --eval-batch-size 4
```

`dataset/`, `runs/` and the other data and output folders are gitignored.

## Experiments

| | input → target | splits (train / val / test) |
| --- | --- | --- |
| E1 | left or right article → the event's multi-source neutral article | 16,000 / 2,000 / 2,000 |
| E2 | one article → the same article rewritten without framing | 12,484 / 1,778 / 1,775 |

Both use LED-base-16384, AdamW at 3×10⁻⁵ with 10% warm-up, effective batch 8,
fp32, seed 42, greedy decoding up to 512 tokens, and the triplet loss with
margin 0.2, at λ ∈ {0, 0.1, 0.35} for E1 and λ ∈ {0, 0.1, 0.2, 0.35} for E2. E2 has two triplet variants that differ
only in the negative: **ideology-anchored** (a different event from the same
side) and **publisher-anchored** (a different event from the same outlet). The
positive is always the opposite-side article about the same event.
