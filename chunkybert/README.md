# ChunkyBERT

Model source (`model.py`, `train.py`) is taken verbatim from the paper's official
code release, https://github.com/dloiya/chunkybert (`chunkybert-final.ipynb`):

> Loiya, D., Kulal, S. S., Reddy, M. S. M. & Shetty, R. D. "ChunkyBERT: a novel
> technique for multiclass political bias detection in news media." *Scientific
> Reports* 16, 15323 (2026). https://doi.org/10.1038/s41598-026-46646-z

The only changes made here: the notebook's hard-coded Kaggle CSV path is
replaced with a `--data-csv` CLI flag, plotting is made headless (`--no-plots`
saves nothing, otherwise PNGs are written instead of `plt.show()`), and the
notebook's documented unterminated `print(...)` at the end of `main()` is
closed. The `ChunkedBERTClassifier` architecture and the `train`/`run` loops
are unmodified.

Dataset: the paper's Data Availability section points to
https://github.com/ramybaly/Article-Bias-Prediction (Baly et al.'s
Article-Bias-Prediction corpus, 37,554 articles labeled Left/Centre/Right —
matches Table 2/3 of the paper exactly). This is the same corpus used for the
Bailey baseline in [eval_metrics.md](../eval_metrics.md).

## Setup

```bash
# from repo root
git clone https://github.com/ramybaly/Article-Bias-Prediction.git external/Article-Bias-Prediction

cd chunkybert
pip install -r requirements.txt

python prepare_data.py \
  --jsons-dir ../external/Article-Bias-Prediction/data/jsons \
  --output ../data/article_bias_prediction.csv
```

## Train

Paper defaults (Table 4: batch 16, 5 epochs, lr 2e-5, bert-base-uncased,
max_chunk_length 512, max doc length 5000 tokens, 80/20 stratified split):

```bash
python train.py --data-csv ../data/article_bias_prediction.csv --output-dir runs/full   # runs/ is gitignored
```

Smoke test (small subset, short run, to verify the pipeline end-to-end):

```bash
python train.py --data-csv ../data/article_bias_prediction.csv --limit 40 --num-epochs 1 \
  --batch-size 2 --max-length 512 --no-plots --output-dir runs/smoke
```

Full training does not fit this machine's 4 GB GPU in reasonable time (~95 min/epoch); it was
run on Kaggle instead — see `kaggle/train/`. The trained 5-epoch weights are at
`../models/chunkybert_5ep.pt`.

Note: the paper trained on a Tesla T4 (16GB). On smaller GPUs, reduce
`--batch-size` and/or `--max-length`; chunking (up to 5000/512 ≈ 10 chunks per
document) is memory-heavy since a whole document's chunks are encoded in one
BERT forward pass.
