# Code guide

A guided tour of the evaluation code, for studying it. Read `EVALUATION_WALKTHROUGH.md` first for
what the code is *for*; this document explains *how* it works.

Assumed: Python and basic PyTorch (`nn.Module`, a training loop, a loss). Everything else is
explained where it first matters, mostly in §2.

---

## 1. The shape of the codebase

All the code fits one pipeline:

```
 raw data ─► prepare ─► train a classifier ─► weights ─► score texts ─► analyse
                                                │
                       (Kaggle for training; the laptop for scoring and analysis)
```

| Stage | ChunkyBERT | Rebuilt ChunkyBERT models | Baly | Jev (zero-shot) |
|---|---|---|---|---|
| Prepare data | `chunkybert/prepare_data.py` | (same CSV) | `baly/baly.py prep` | `jev_classify.load_items` |
| Model | `chunkybert/model.py` | `replication/variants.py` | `baly/baly.py: build` | hosted API (no model code) |
| Train | `chunkybert/train.py` | `replication/run_variant.py` | `baly/baly.py: run` | none |
| Score `stories_10k` | `evaluation/chunkybert_eval.ipynb` | `evaluation/best_variant_eval.ipynb` | `evaluation/baly_eval.ipynb` | `jev/jev_classify.py run` |
| Score rewrite sets | `evaluation/long10k/score.py` | same | same | `jev/score_neutral.py` |

Code size is small: about 2,400 lines of Python plus notebooks. Four files hold most of the ideas:
`chunkybert/model.py`, `replication/variants.py`, `baly/baly.py` and `jev/jev_classify.py`.

---

## 2. Background you need

**Tokenizer output.** `tok(texts, truncation=True, max_length=N, padding=..., return_tensors="pt")`
returns:
- `input_ids`: integer token IDs, shape `(batch, seq_len)`;
- `attention_mask`: 1 for real tokens, 0 for padding.

Every model here takes both, because texts in a batch differ in length and get padded to match.
Always ask: does this code ignore the padding positions, or treat them as text?

**BERT special tokens.** A tokenised text looks like `[CLS] tok tok … tok [SEP] [PAD] [PAD]`.
- `[CLS]` is a summary slot at position 0.
- `[SEP]` marks the end.
- BERT accepts at most 512 positions.

**BERT's two outputs.**
- `last_hidden_state`: one 768-dim vector per token, shape `(batch, seq_len, 768)`.
- `pooler_output`: the `[CLS]` vector passed through one extra dense layer, shape `(batch, 768)`.

ChunkyBERT uses the first; Baly's models use the second.

**Mixed precision.** `torch.autocast` runs most operations in 16-bit floats for speed and memory.
`GradScaler` multiplies the loss up before `backward()`, so small gradients don't round to zero, then
divides back down. This is why the training loops contain `scaler.scale(loss).backward()` and
`scaler.unscale_(opt)` before gradient clipping.

**DataParallel.** On Kaggle's two GPUs, `nn.DataParallel(model)` splits each batch across them. Its
weights live under `model.module`. Save `model.module.state_dict()`, or the parameter names get a
`module.` prefix.

---

## 3. Suggested reading order

1. `chunkybert/model.py`: the core architecture, about 150 lines.
2. `chunkybert/train.py`: a standard training loop.
3. `replication/variants.py`: the same model, made configurable.
4. `baly/baly.py`: a second family of models, including the two de-biasing tricks.
5. One evaluation notebook, e.g. `evaluation/baly_eval.ipynb`: how trained models score new data.
6. `evaluation/long10k/score.py`: the same scoring, as a resumable script.
7. `jev/jev_classify.py`: a completely different approach, an async API client with a spending cap.
8. Skim the rest: `bias_types.py`, `vad_scores.py`, `led_eval/`, `qa_eval/`.

Each section below ends with a few questions to check understanding.

---

## 4. ChunkyBERT

### 4.1 `chunkybert/prepare_data.py` (50 lines)

Reads the released dataset (one JSON file per article) and writes one CSV:
`data/article_bias_prediction.csv`. It keeps `content` (article text) and `bias` (0 Left, 1 Centre,
2 Right), plus metadata. Nothing subtle.

### 4.2 [`chunkybert/model.py`](chunkybert/model.py) — read this one closely

This is the authors' code, unchanged. One class, `ChunkedBERTClassifier`, with two methods that
matter.

**`__init__`** builds four parts:
- `self.bert`: pretrained `bert-base-uncased`.
- `self.transformer`: two extra `nn.TransformerEncoderLayer`s on top of BERT. These are the layers
  the paper claims are essential; the replication found they aren't.
- `self.attention`: a small MLP (`Linear → GELU → Linear`) that gives each token a single score,
  used for attention pooling.
- `self.classifier`: `Dropout → Linear(768, 3)`.

**`chunk_text_batched`** ([line 50](chunkybert/model.py#L50)) turns a batch of long documents into
a batch of 512-token chunks:
1. For each document, keep only the real tokens (`attention_mask[i].sum()`) and strip `[CLS]` and
   `[SEP]`.
2. Cut the rest into pieces of 510 tokens ([line 59](chunkybert/model.py#L59)), leaving room to
   re-add `[CLS]` and `[SEP]` to each piece.
3. Re-wrap each piece as `[CLS] … [SEP]`, pad it to 512, and build its mask.
4. Record which document each chunk came from in `chunk_to_batch_idx`.

So a batch of 2 documents with 1,200 and 300 real tokens becomes 3 + 1 = 4 chunks, shape `(4, 512)`.

**`forward`** ([line 113](chunkybert/model.py#L113)):

```
chunks (C, 512) ──BERT──► (C, 512, 768) ──2 Transformer layers──► (C, 512, 768)
   ──attention pooling──► (C, 768) ──mean over each document's chunks──► (B, 768) ──classifier──► (B, 3)
```

- **Attention pooling** ([line 133](chunkybert/model.py#L133)): score every token, `softmax` over the
  512 positions, take the weighted sum of the token vectors.
- **Per-document mean:** a Python loop over the batch, selecting chunks with `chunk_to_batch == i`.

**Things to notice:**
- **Padding is not masked** in either the extra Transformer layers (no `src_key_padding_mask`) or the
  pooling softmax. A short final chunk with 100 real tokens and 412 `[PAD]`s spreads attention over
  the padding. This is the authors' behaviour; `replication/variants.py` fixes it, which is why the
  two code paths give slightly different numbers.
- **`forward` takes the tokenizer as an argument**, only to look up the special-token IDs.
- **Chunking is a Python loop on the GPU**, which is slow but correct.

### 4.3 `chunkybert/train.py` (573 lines, mostly plotting)

The authors' training code with three changes, listed in its docstring:
- a `--data-csv` flag instead of a hard-coded Kaggle path;
- plots saved to files instead of displayed;
- a missing `)` fixed.

Pieces:
- **`TextClassificationDataset`** ([line 45](chunkybert/train.py#L45)) tokenises each article with
  `padding='max_length'` and `max_length=5000`. Every article becomes a 5,000-long tensor, mostly
  padding. That is wasteful but harmless, because `chunk_text_batched` drops the padding anyway.
- **`train()`** ([line 155](chunkybert/train.py#L155)) is a standard loop:
  - AdamW with weight decay 0.01;
  - linear warm-up for the first 10% of steps, then linear decay;
  - mixed precision;
  - gradient clipping at 1.0;
  - cross-entropy loss;
  - after each epoch, accuracy, AUC and plots on train and validation.
- **`main()`**: an 80/20 stratified split with seed 42, then `run()`.

The version actually run on Kaggle is a single-file copy:
`chunkybert/kaggle/train/chunkybert_train.py`. Same code, merged so Kaggle can run it as one script,
plus a line that saves the weights.

**Check yourself**
- A 5,000-token document: how many chunks? (⌈4,998 / 510⌉ = 10.)
- Why does `train.py` padding to 5,000 not change the model's predictions?
- Where would you add `src_key_padding_mask` to stop the extra layers attending to padding?

---

## 5. The rebuilt comparison models: [`replication/variants.py`](replication/variants.py)

The paper's comparison models (ablations and GRU/TCN baselines) have no released code. Instead of 14
separate classes, one class, `VariantClassifier`, takes switches, and each paper model is a setting
of those switches.

| Switch | Values | Meaning |
|---|---|---|
| `backbone` | `bert` / `embedding` | Pretrained BERT, or a word-embedding table trained from scratch |
| `encoder` | `transformer` / `gru` / `tcn` / `none` | What runs on top of the backbone |
| `pooling` | `attention` / `mean` | How a chunk's tokens become one vector |
| `chunk_agg` | `mean` / `max` | How a document's chunk vectors become one vector |
| `chunking` | `True` / `False` | Split into 512-token chunks, or not |
| `freeze_backbone` | bool | Stop BERT's weights from training |
| `simple_head` | bool | Classifier = bare `Linear`, no dropout |

The paper's models are listed as dictionaries of these switches at the bottom: `TABLE8`
([line 178](replication/variants.py#L178)) and `TABLE10` ([line 189](replication/variants.py#L189)).
For example, `"Freeze BERT": dict(freeze_backbone=True)`. Unlisted switches keep their defaults,
which reproduce ChunkyBERT exactly apart from the padding mask.

Methods:
- **`chunk_text_batched`**: copied from the authors' `model.py`.
- **`encode`** ([line 129](replication/variants.py#L129)): runs the backbone, then the chosen encoder.
- **`pool`** ([line 140](replication/variants.py#L140)): **masks padding.** Padding scores are set to
  the minimum float before the softmax ([line 144](replication/variants.py#L144)), so they get zero
  weight. The mean option divides by the count of real tokens.
- **`forward`** ([line 148](replication/variants.py#L148)): the no-chunking branch
  ([line 152](replication/variants.py#L152)) means different things per backbone:
  - BERT can't read past 512 tokens, so it truncates to the first 512.
  - A GRU or TCN has no length limit, so it reads the whole 5,000 tokens.

**The two helper classes:**
- **`TCN`:** a stack of 1-D convolutions with dilation 1, 2, 4, 8.
- **`Chomp`:** trims the extra padding on the right, so each output position sees only earlier
  tokens (a "causal" convolution).

### `replication/run_variant.py`

Trains one variant for one epoch (the paper's protocol), then evaluates on validation and appends one
row to a results CSV. `--variant "Freeze BERT"` picks the configuration by name. The training loop
matches `train.py`'s, minus the plots.

Shell scripts in `replication/sweeps/` loop over variants. Kaggle versions of the same code are in
`replication/kaggle/`.

**Check yourself**
- Which switch values give the paper's "no Transformers" model? Which give the no-BERT model that
  actually matches the paper's number? (`encoder="none"`; `backbone="embedding", encoder="none"`.)
- Why must the GRU baseline without chunking read all 5,000 tokens, while BERT without chunking reads
  512?

---

## 6. Baly et al.: [`baly/baly.py`](baly/baly.py) (351 lines)

One file, six models, selected by a mode argument (`python baly/baly.py <mode>`):

| Mode | Model |
|---|---|
| `prep` | Build `data/baly_articles.csv` from the released JSON |
| `majority` | Always predict the most common training label |
| `bert --split media\|random` | Plain BERT classifier |
| `medium` | BERT predicting the outlet (73 classes) instead of the ideology |
| `aa` | BERT + adversarial adaptation |
| `tlp` | Triplet pre-training, then the plain classifier |

Read it top to bottom; the sections are marked with comment banners.

**Data** (lines 25–84):
- `find()`: locates files by walking directories, because Kaggle mounts datasets at unpredictable
  paths.
- `load()`: joins the article table with Baly's official split files (`train/valid/test.tsv`, keyed
  by article ID).
- `medium_split()`: builds a new stratified split for the outlet task over the 73 largest outlets.
  The paper's version wasn't released.
- `sample_triplets()` ([line 63](baly/baly.py#L63)): for each article (the "anchor"), finds:
  - a *positive*: same ideology, different outlet;
  - a *negative*: different ideology, same outlet;
  - all three on the same topic.

  Only anchors with both a positive and a negative are kept (9,220 of 26,590).

**`metrics()`** (line 87): macro-F1, accuracy, and MAE. MAE is computed only for the 3-class tasks,
because outlet IDs have no order.

**`build()`** ([line 97](baly/baly.py#L97)) is the most interesting part.
- `BertClf`: BERT → `pooler_output` → dropout → linear.
- When `n_media > 0`, a second linear head (`media_head`) guesses the outlet, but it receives the
  BERT vector through `GradReverse`.
- **`GradReverse`** (lines 101–109) is an autograd function that does nothing on the way forward and
  multiplies the gradient by **−λ** on the way back:
  - the media head is trained normally to guess the outlet;
  - the gradient reaching BERT from that head is flipped, so BERT is pushed to make the outlet *harder*
    to guess.

  That is adversarial adaptation in nine lines.

**`run()`** ([line 133](baly/baly.py#L133)), the training loop:
- **`batches()`** tokenises on the fly with `padding=True`, padding to the longest text in the batch.
  That is cheaper than `train.py`'s pad-to-5,000.
- **Loss** ([line 183](baly/baly.py#L183)): `ce(ideology) + ce(outlet)` when adversarial, otherwise
  just `ce(ideology)`. Only this sum is logged, which is why the walkthrough says the outlet loss was
  *inferred*.
- **Early stopping** (lines 194–201): keep the weights with the best validation macro-F1; stop after
  `patience` epochs without improvement. At the end, reload the best weights and predict the test set.

**`pretrain_triplets()`** ([line 211](baly/baly.py#L211)):
- Trains only `model.bert` with `nn.TripletMarginLoss`, which pushes the anchor at least `margin`
  closer to its positive than to its negative.
- Anchor, positive and negative are encoded in one batch (`flat` = a, p, n, a, p, n, …) and reshaped
  to `(n, 3, 768)`.
- It returns only the `bert.*` weights. `run()` then loads them with `strict=False`, so the classifier
  head starts fresh.

**`main()`** ([line 263](baly/baly.py#L263)): parses arguments, then dispatches on the mode.
- The argument list is split into settings the paper reports and settings it doesn't (assumptions).
  The assumptions are written into every result row.
- Outputs go to `--out`:
  - `results.jsonl`: one row per run;
  - `<name>_epochs.csv`: per-epoch log;
  - `<name>_test_probs.csv`: probability vectors;
  - optionally `<name>.pt`: weights.

**The Kaggle copies** (`baly/kaggle/main/run.py`, `baly/kaggle/tlp/run.py`) are generated from
`baly.py` with a short driver appended that runs several modes in sequence. Edit `baly.py`, not the
copies.

**Check yourself**
- What would happen with `GradReverse` returning `+λ·g` instead of `−λ·g`?
- Why is `media_head` given `GradReverse.apply(h)` while `head` gets `h` directly?
- Why does the AA checkpoint need `strict=False` when loaded into a plain `build(3)` model?
  (It holds extra `media_head.*` weights.)

---

## 7. Applying trained models to new data

### 7.1 The evaluation notebooks (`evaluation/*.ipynb`)

`chunkybert_eval`, `best_variant_eval` and `baly_eval` share a structure:
1. **Load** the model class and the saved weights from `models/`.
2. **Flatten** `data/stories_10k.jsonl`:
   - into one row per labelled article: `story_id, slot, label, publisher, text`, using the article
     **body only**;
   - plus one row per neutral summary.
3. **`score(texts)`**: tokenise in batches of 16 with `padding=True`, run the model under `autocast`,
   apply `softmax`, collect an `(N, 3)` array of probabilities.
4. **`attach(df, probs)`** adds the derived columns:
   - `p_left, p_centre, p_right`;
   - `pred_id` (the argmax) and `confidence` (its probability);
   - `margin`: top probability minus the second;
   - `entropy`: normalised so 0 = certain and 1 = uniform.
5. **Analyse** with sklearn: confusion matrices, per-class reports, per-outlet accuracy.
6. **Save** per-article scores to `evaluation/scores/*.csv`.

Notebooks are run headless with
`jupyter nbconvert --to notebook --execute --inplace X.ipynb`, so their outputs are saved in the file.

**Matching rows across models.** Rows are matched by `story_id + ":" + slot`, not by position. The
"compare with ChunkyBERT" cell in `best_variant_eval` shows the pattern.

### 7.2 `evaluation/long10k/score.py`: scoring as a script

Written later, to score the team's Nemotron rewrite sets (originals and rewrites) with every model. It
uses the notebooks' scoring logic as a resumable command-line tool:
- **`build`** writes `texts.jsonl`: one row per text, with `id = story:side:kind`.
- **`load_model`** returns a uniform tuple `(model, tokenizer, max_len, needs_tokenizer, vocab)` for
  every model type. This hides that ChunkyBERT-style models take the tokenizer in `forward` and Baly
  models don't.
- **`score`**:
  - skips IDs already in the output file, so a crashed run resumes;
  - sorts texts by length, which keeps padding small;
  - appends one JSON line per text.
  - For the outlet classifier it stores the top-5 outlets instead of 73 probabilities.
- **`vad`**: the same lexicon method as `evaluation/vad/vad_scores.py`.

`build_notebook.py` generates the analysis notebooks (`short_eval.ipynb`, `long10k_eval.ipynb`), and
`run_all.sh` runs every model over both datasets.

**Check yourself**
- Why sort texts by length before batching?
- How does `score()` know where to resume?

---

## 8. Jev, zero-shot: [`jev/jev_classify.py`](jev/jev_classify.py)

No model weights; every prediction is an HTTP request to TypeSafe's API, via their SDK.

- **Questions** (`QUESTIONS`, top of file): each is a typed object, e.g. `Choice(instructions=...,
  criteria={"left": ..., "centre": ..., "right": ...})`. The API returns a probability per option.
  - Four designs are defined; `outlet_style` won.
  - `probs()` maps any answer to `[p_left, p_centre, p_right]`. The five-point `score5` design is
    collapsed to three classes.
  - The SDK returns integer keys for that design; the documentation shows strings.
- **`state_for()`** decides what the API sees: `{"article": body}` or
  `{"title": ..., "article": body}`.
- **`Ledger`** ([line 103](jev/jev_classify.py#L103)) is a spending cap for concurrent requests:
  - Before a request it *reserves* an estimated token count. If that would exceed the cap, the request
    is refused.
  - After the response it *settles*: releases the reservation and adds the actual tokens.
  - An `asyncio.Lock` keeps concurrent updates consistent.
  - The running total is written to `results/spend.json` after every request, so it survives across
    runs.
- **`score_all()`** runs up to `concurrency` requests at once:
  - an `asyncio.Semaphore` limits concurrency;
  - a `stop` event halts everything once the budget is reached;
  - IDs already in the output file are skipped, so runs resume;
  - each result is one JSON line.
- **`main()`**:
  - `dev`: score 450 balanced validation articles with every design, with and without titles.
  - `run`: score everything except those 450 with one design. Summaries are shuffled first, so a
    budget cut leaves a random sample.

The API key is read from `.env` by `api_key()`. That file isn't in git; never print it.

**`jev/evaluate.py`** has the metrics plus the "prior correction":
- `fit_prior` searches a grid for two log-offsets (Centre fixed at 0) that maximise macro-F1 on the
  dev set;
- `apply_prior` adds them to the log-probabilities.

It is the standard way to correct a model that over-predicts one class. Here it didn't help on the
full data.

**`jev/bias_types.py`** reuses `Ledger` and `score_all`, with a different question set: twelve yes/no
`Noul` questions (spin, loaded words, …), five direction `Choice`s, and `outlet_style`, all in one
request per text.

**`jev/score_neutral.py`** reuses the same runner to score rewrites and summaries, without titles.

**Check yourself**
- Why reserve an *estimate* before the request rather than just checking the total afterwards? (With
  16 requests in flight, all would pass a check made before any of them finished.)
- What makes rerunning `run` after a crash cheap?

---

## 9. The remaining files

- **`evaluation/vad/vad_scores.py`** (47 lines): lowercases, splits into words with a regex, looks up
  each word in the Warriner lexicon, and averages valence, arousal and dominance over the words found.
- **`led_eval/`** (in progress), the evaluation of the team's LED rewriters:
  - `build_texts.py` builds one table of every text to score: originals, centre article, LLM summary,
    Nemotron rewrite and each model's rewrite. Each row has a hash, so each distinct text is scored
    once.
  - `generate.py` runs the LED models: input `title </s> body`, 512 tokens, greedy decoding. Output is
    appended per batch, so a rerun resumes.
- **`qa_eval/qa_eval.py`** (run on Kaggle 1–2 Oct; results in `qa_eval/README.md`), measuring whether rewrites keep the content. Three stages:
  1. `gen`: generate rewrites.
  2. `nli`: does each reference fact still follow from the rewrite, according to an entailment model?
  3. `qe`: QuestEval, which asks questions of one text and answers them from the other.

  It has never been run in full.

---

## 10. Conventions used throughout

- **Labels:** `0 = Left, 1 = Centre, 2 = Right` everywhere. MAE depends on this order.
- **Probability vectors, not just labels.** Every scorer saves the full `[p_left, p_centre, p_right]`,
  so analyses can measure shifts in probability, not only flips in the argmax.
- **Body text only** when scoring, so originals and rewrites (which have no title) are comparable.
- **Resumable outputs:** long jobs append one JSON line per item and skip IDs already present.
- **Paths from the repo root:** scripts compute `ROOT = Path(__file__).resolve().parents[n]`, so they
  run from any working directory.
- **Paper settings vs assumptions** are kept separate in argument lists and written into result rows.
- **Not in git:** `data/`, `models/`, `external/`. `CLAUDE.md` explains how to regenerate or download
  each.

---

## 11. Known quirks (worth knowing before you change anything)

| Where | Quirk |
|---|---|
| `chunkybert/model.py` | No padding mask in the extra layers or the pooling (authors' behaviour, kept on purpose) |
| `chunkybert/train.py` | Pads every article to 5,000 tokens; slow and memory-hungry, but correct |
| `replication/variants.py` | Masks padding, so its "Full Model" is not numerically identical to `model.py` |
| `replication/results/results_table8_lr1e-3.csv` | Rows have different column counts (a column was added midway); read it by position |
| `baly/baly.py` | Logs only the summed loss for adversarial training, so the outlet loss can't be read directly |
| `baly/baly.py: medium` | Outlet = web domain, top 73 by size; 1,436 articles from smaller outlets excluded |
| Loading the AA checkpoint | Needs `strict=False` because of its extra outlet head |
| Kaggle copies | `chunkybert/kaggle/`, `replication/kaggle/`, `baly/kaggle/` are generated or merged copies; edit the source files |
| Jev SDK | Answer probabilities come back with integer keys for the five-point design |

---

## 12. Running things

Quick checks that run on the laptop in a minute or two (use `--limit` to subsample):

```bash
python chunkybert/train.py --data-csv data/article_bias_prediction.csv --limit 40 \
       --num-epochs 1 --batch-size 2 --max-length 512 --no-plots --output-dir /tmp/cb
python replication/run_variant.py --variant "Freeze BERT" \
       --data-csv data/article_bias_prediction.csv --limit 24 --batch-size 2 --max-length 300
python baly/baly.py bert --split media --limit 24 --max-len 128 --batch-size 2 --epochs 1 --out /tmp/baly
python evaluation/long10k/score.py time --dataset short --model baly_media
```

Full training runs on Kaggle (see "Kaggle mechanics" in `CLAUDE.md`). The local GPU has only 4 GB.
