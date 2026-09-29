# CLAUDE.md

Working notes for this repo. Project state and results live in `CONTEXT.md`; open work in `TODO.md`.

## What this is

Branch `harshit` of the team repo `RaunakSeksaria/ANLP-Project` (private; other branches hold the
team's summarisation/de-biasing work). This branch builds the **political-bias classifiers** listed
in `eval_metrics.md` and evaluates them on the team's `stories_10k` data: replicating ChunkyBERT
(Loiya et al. 2026) and Baly et al. 2020, then scoring the team's articles and neutral summaries.

## Layout

```
chunkybert/     ChunkyBERT from the authors' own code (model.py, train.py verbatim) + Kaggle kernel
replication/    Rebuilt ChunkyBERT baselines/ablations (variants.py = every model as one class)
                REPLICATION.md is the full log with measured-vs-paper tables
evaluation/     Notebooks scoring models on stories_10k; scores/ = per-article probabilities; vad/
baly/           Baly et al. 2020 replication (in progress)
jev/            Jev (TypeSafe): ideology classifier (jev_eval.ipynb) and AllSides bias types (bias_types.py/.ipynb);
                results/spend.json = API spend ledger
data/           gitignored — datasets (see below)
models/         gitignored — trained weights (see below)
external/       gitignored — clones of dloiya/chunkybert and ramybaly/Article-Bias-Prediction
```

## Data (`data/`, not in git)

| File | What | How to regenerate |
|---|---|---|
| `article_bias_prediction.csv` | Baly et al. corpus, 37,554 articles, `content`/`bias` (+ link, title, source, author) | clone `ramybaly/Article-Bias-Prediction` into `external/`, then `python chunkybert/prepare_data.py --jsons-dir external/Article-Bias-Prediction/data/jsons --output data/article_bias_prediction.csv` |
| `stories_10k.jsonl` | Team data: 10k stories, each with `left`/`center`/`right` source articles (labelled by slot) and one LLM `neutral` summary (unlabelled) | upstream name `neutral_articles_10k.jsonl` in the private team repo; `gh api .../contents/neutral_articles_10k.jsonl --jq .download_url`, then `curl` that URL. The download truncates or over-appends; keep exactly the first 10,000 valid JSON lines |

Baly's official train/valid/test splits (media-based and random) are in
`external/Article-Bias-Prediction/data/splits/`, keyed by article `ID`.

**Labels everywhere: 0 = Left, 1 = Centre, 2 = Right.** In `stories_10k`, only the `neutral` field
is neutral; the three source articles are ordinary partisan news. Do not call the whole file
"neutral".

## Models (`models/`, not in git)

| File | Model | Val acc | Kaggle output name |
|---|---|---|---|
| `chunkybert_5ep.pt` | ChunkyBERT, authors' code, 5 epochs | 85.7% (final epoch) | `chunkybert_final_epoch.pt` |
| `no_classifier_1ep.pt` | `VariantClassifier(3, simple_head=True)`, 1 epoch | 82.2% | `best_variant_no_classifier.pt` |

Load: `ChunkedBERTClassifier(3)` from `chunkybert/model.py`, or `VariantClassifier` from
`replication/variants.py`. `forward(input_ids, attention_mask, tokenizer)` takes the tokenizer.

## Compute rules

- **Local GPU**: RTX 3050, 4 GB, and the Windows host can hold ~1 GB more that `nvidia-smi` inside
  WSL does not show. BERT fine-tuning locally needs batch 1–2 at ≤512 tokens, so use it only for
  smoke tests. Small from-scratch models (GRU/TCN) train fine locally.
- **Any training run estimated over 1 hour locally → smoke-test locally, run for real on Kaggle.**
  Estimate by timing a small `--limit` run first.
- **Ask before pushing a Kaggle kernel or uploading a dataset.** It runs under the user's account and
  spends their weekly GPU quota (30 h).

## Kaggle mechanics (learned the hard way)

- CLI is authenticated (`kaggle kernels list --mine`); account `harshitlalwani`.
- Training data is the private dataset `harshitlalwani/media-bias-csv` (one file, `out.csv`). It mounts
  under `/kaggle/input/datasets/harshitlalwani/...`, **not** `/kaggle/input/media-bias-csv/` — find
  it with `os.walk` for `out.csv`. A freshly created dataset needs a few minutes before kernels can
  mount it.
- Kernels are `script` type, GPU = 2×T4; `DataParallel` kicks in automatically. Save weights with
  `model.module.state_dict()` when wrapped. Nothing is saved unless the script saves it.
- Logs and outputs are only downloadable **after** the run finishes (`kaggle kernels output <ref> -p dir -o`).
  To track progress, poll `kaggle kernels status`.
- When polling, match the enum exactly (`KernelWorkerStatus.COMPLETE|ERROR|CANCEL`). Transient CLI
  network errors contain the word "error" and will trip a loose grep.
- `kaggle datasets version` **replaces all files**. `chunkybert/kaggle/model_upload/` deliberately
  lacks the `.pt` and `model.py` (they were duplicates). Copy them back in before pushing a new version.
- Timing reference (2×T4): BERT at 512 tokens ≈ 12 min/epoch on ~30k articles; ChunkyBERT at 5,000
  tokens ≈ 30 min/epoch.

## How the user wants work done

- **Use the authors' code verbatim when it exists**; rebuild from the paper only when it does not,
  and say so.
- **Start with the cheapest reading and escalate only if results don't match.** Stop tuning once a
  result is clearly not reproducible; don't search hyperparameters until a paper number is hit.
- **Report measured next to paper numbers**, with the gap, for every model.
- **Keep a markdown progress log updated during long efforts** (pattern: `replication/REPLICATION.md`).
- Report the **full probability vector** per article, not only the argmax.
- Execute notebooks headlessly so outputs are embedded:
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 X.ipynb`.

## TypeSafe / Jev

- API key is `JEV_API` in `.env`, read by `jev/jev_classify.py`. **Never print or read `.env` in the
  transcript.**
- Price is $0.042 per million input tokens (output free), about 600 tokens per article with one
  question. `results/spend.json` is a persistent ledger; `--cap-usd` refuses requests past the cap.
  Spent: $0.95 on the ideology run (budget $1) + $0.19 on the 3,000-text AllSides bias-type run (user-approved) = $1.14 total. Ask before spending more.
- Score answer probabilities come back with **int** keys in the Python SDK (docs show strings).

## Gotchas

- `pkill -f <pattern>` matches the invoking shell's own command line and kills it (exit 144).
  Collect PIDs with `ps` and kill by PID.
- `replication/results/results_table8_lr1e-3.csv` has a ragged schema (the `tcn_levels` column was
  added midway). Read it positionally; `variant, tag, val_acc` are the first three columns.
- `replication/variants.py` masks padding in attention/mean pooling. The authors' released
  `chunkybert/model.py` does not. Keep that in mind when comparing across the two.
