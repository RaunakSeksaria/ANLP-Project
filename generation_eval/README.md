# generation_eval/

Article-level evaluation of the generated text: what an external reader or
classifier can still recover from a rewrite, and what it costs. Every scorer
reads the article **body** only and treats inputs, Nemotron targets and LED
outputs alike. See `eval_metrics.md` for the list of metrics.

| folder | what it does |
| --- | --- |
| `led_eval/` | `generate.py`: LED rewrites of the held-out test articles (or DiverseSumm) with the checkpoints from `RaunakSeksaria/anlp-neutral-led`; `build_texts.py`: one table, `data/led_eval/texts.jsonl`, of every text to score (originals, Nemotron targets, centre articles, multi-source articles, LED outputs) |
| `evaluation/` | `chunkybert_eval.ipynb`, `best_variant_eval.ipynb`: left/centre/right probabilities from two political-bias classifiers; `vad/vad_scores.py`: valence/arousal/dominance tonality (Warriner et al. 2013 lexicon, `vad/warriner_vad.csv`) |
| `chunkybert/` | ChunkyBERT (Loiya et al. 2026), the authors' code: `model.py`, `train.py`, `prepare_data.py`; Kaggle training kernel in `kaggle/train/` |
| `replication/` | `variants.py`: the No-Classifier variant (`VariantClassifier`) used as the second classifier; `kaggle/best_variant/best_variant_train.py` trains and saves it; `run_variant.py` runs one-epoch variant comparisons |
| `jev/` | Zero-shot ideology and AllSides bias-type labelling with the TypeSafe Jev API (paid; spend ledger and hard cap built in) |
| `qa_eval/` | QA-based faithfulness and coverage: DiverseSumm answer coverage (NLI) and QuestEval on the held-out test articles |

Labels everywhere: **0 = Left, 1 = Centre, 2 = Right**.

## Setup

From the repository root:

```bash
uv run python splits/download_hf.py       # dataset/raw/ (export HF_TOKEN; the dataset repo is private)
generation_eval/setup_data.sh             # links data/stories_10k.jsonl etc., extracts data/own_test.jsonl,
                                          # clones DiverseSumm into external/
```

`data/stories_10k.jsonl` is `neutral_articles_10k.jsonl` under its older name.
`data/own_test.jsonl` holds the 2,000 held-out test articles (left and right of
the 1,000 test events) with the event's centre article as `target_text`.

Classifier weights go in `models/` (gitignored):

| file | model | where to get it |
| --- | --- | --- |
| `models/chunkybert_5ep.pt` | ChunkyBERT, 5 epochs | Kaggle dataset `harshitlalwani/chunkybert-political-bias` (`chunkybert_final_epoch.pt`), or train with `chunkybert/` |
| `models/no_classifier_1ep.pt` | `VariantClassifier(3, simple_head=True)`, 1 epoch | output of the Kaggle kernel `chunkybert-best-variant` (`best_variant_no_classifier.pt`), or train with `replication/kaggle/best_variant/best_variant_train.py` |

Both were trained on Baly et al.'s Article-Bias-Prediction corpus: clone
`https://github.com/ramybaly/Article-Bias-Prediction` into `external/` and run
`chunkybert/prepare_data.py` (see `chunkybert/README.md`). The LED checkpoints
are downloaded from Hugging Face on first use.

## Running

From `generation_eval/`:

```bash
uv run python led_eval/generate.py --set test        # LED rewrites -> led_eval/out/
uv run python led_eval/build_texts.py                # -> data/led_eval/texts.jsonl
uv run python evaluation/vad/vad_scores.py           # -> evaluation/vad/vad_scores.csv
uv run python qa_eval/qa_eval.py --stage all --out results/qa   # --limit 5 for a smoke test
(cd evaluation && uv run jupyter nbconvert --to notebook --execute --inplace chunkybert_eval.ipynb)
```

- The notebooks resolve paths from their own folder, so run them from there.
- QuestEval needs `pip install questeval --no-deps`; `qa_eval.py` carries the
  `datasets.load_metric` shim it needs.
- Jev needs the `typesafe_sdk` package and `JEV_API` in `generation_eval/.env`;
  `jev/jev_classify.py --help` lists the modes. Spend is recorded in
  `jev/results/spend.json` and capped by `--cap-usd`.
- LED generation and the BERT classifiers want a GPU; ChunkyBERT reads up to
  5,000 tokens per article (about 30 min per epoch on 2× T4).
