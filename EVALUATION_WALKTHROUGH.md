# Evaluation walkthrough

A technical walkthrough of the evaluation side of the project, written for someone presenting it
without prior context. It covers what was built, why, how each piece was checked, and what the
results mean.

Sources: `CONTEXT.md` (all results), `replication/REPLICATION.md` and `baly/REPLICATION.md` (full
replication logs), and the notebooks in `evaluation/` and `jev/`. Numbers not taken from those were
measured directly for this document; they are marked *(measured here)*.

---

## 0. Summary

The team is building models that rewrite partisan news so it reads neutrally. Claiming a rewrite is
"more neutral" needs instruments that measure political slant. This work builds those instruments,
checks each against its paper, and applies them to the team's data.

Four findings:

1. **Published ideology classifiers rely heavily on the outlet.** The same BERT model scores 82
   macro-F1 when test articles come from outlets seen in training, and 37 when they don't. Every
   80%+ figure we reproduced comes from that easier setting. The 37 is still well above the
   majority baseline's 21, so some ideology signal exists, but most of the headline accuracy is
   outlet recognition.
2. **On the team's data, every trained classifier scores 36–44% accuracy** on three balanced classes
   (chance 33%). All of them rarely predict Centre and over-predict Right.
3. **A zero-shot model (Jev) beat every fine-tuned classifier** at 47.8% accuracy, with the opposite
   bias: it over-predicts Centre.
4. **Replication found two claims in the ChunkyBERT paper that its own evidence doesn't support.**
   Its key ablation ("removing the Transformer layers drops accuracy to 41%") almost certainly
   measured a model with BERT removed instead. Its "attention is essential" gap most likely comes
   from baseline runs that never trained.

The practical consequence: no single classifier here can certify that a rewrite is neutral. A rewrite
that loses its outlet's style will move these classifiers whether or not its slant changed. Section 7
describes how to use them despite that.

---

## 1. Where this fits in the project

**Team goal.** Take partisan coverage of a news event and produce a neutral version: either a
summary across several outlets or a single article rewritten neutrally. The current models are two
LED (Longformer encoder-decoder) rewriters trained by teammates, one without and one with an added
triplet-loss term (triplet weight λ = 0 and λ = 0.35).

**This work's scope** is the evaluation layer listed in `eval_metrics.md`:

| Instrument | Status |
|---|---|
| VAD (valence / arousal / dominance) | Built (word-list based) and run |
| Publisher classifier, to check publisher leakage | Partly: an outlet classifier exists (Baly, 73 outlets, 78% accuracy). Not yet applied to the team's data or model outputs |
| Ideology classifier: Baly et al. 2020 | Rebuilt; 5 of 6 models run |
| Ideology classifier: ChunkyBERT (2026) | Built from the authors' code; reproduces |
| Added: Jev zero-shot ideology and AllSides bias types | Built and run |
| Added: POLITICS (2022) | Scoped and data located; not run |

The working rule: **before trusting an instrument on our data, reproduce its paper's results on the
paper's data.** Otherwise a poor instrument and a poor rewrite look the same.

---

## 2. Concepts

**Labels track the outlet almost perfectly.** How labels were assigned differs by dataset, but in
practice the label is nearly a function of the publisher:
- *Baly corpus:* per Baly et al., AllSides staff rated each article individually. Even so, only 42
  of 37,554 articles (0.11%) carry a label different from their outlet's most common one, and only 7
  of 491 outlets have more than one label (ProPublica is the main mixed one: 38 Centre, 34 Left)
  *(measured here)*. The paper reports 3.11% differing from AllSides' *published* outlet ratings; we
  can't recompute that without those ratings.
- *`stories_10k`:* the label is exactly a function of the publisher. Each of its 11 publishers appears
  in only one slot: every Fox article is Right, every The Hill article Centre *(measured here)*. How
  the slots were assigned isn't documented.

So identifying the outlet is enough to get the label right. Whether models actually do that is a
separate question, answered by the evidence in §4.3, not by the labels alone.

**Random split vs media split** (the idea behind Baly et al. 2020):
- *Random split:* articles are shuffled into train and test, so the same outlets appear in both. A
  model can learn "Breitbart's style → Right".
- *Media split:* train and test use different outlets. The model has to generalise ideology to
  outlets it hasn't seen. Much harder, and the more meaningful test.

**Publisher leakage.** If a rewrite still lets a classifier identify the original outlet, the rewrite
kept that outlet's style. A publisher classifier detects this.

**Metrics.**
- *Accuracy.*
- *Macro-F1:* F1 averaged over the three classes, so ignoring a class is penalised.
- *MAE:* mean distance on the ordinal scale Left=0, Centre=1, Right=2 (Baly's metric). Calling a
  Left article Right costs 2; calling it Centre costs 1.
- *One-vs-rest AUC.*
- *Per-class recall:* where most of the findings show up.

**Fine-tuned vs zero-shot.** The BERT models are trained on labelled articles. Jev is a hosted model
that answers a typed question about a text and returns option probabilities; it was not trained on
our labels.

**Statistical precision.** On the 30,000 `stories_10k` articles, an accuracy near 40% has a 95%
interval of about ±0.6 points, so gaps of 1+ point there are real. On Baly's 1,300-article test set
it is about ±2.7 points. That set also covers only 13 outlets, so differences of a few points there
are not meaningful.

---

## 3. Data

### 3.1 Article-Bias-Prediction (Baly corpus) — what the classifiers are trained on

- 37,554 articles from AllSides: Left 13,005, Centre 10,815, Right 13,734.
- Full article text: median 1,018 BERT tokens, and 88.6% are longer than BERT's 512-token limit
  *(measured here)*. That length is why ChunkyBERT splits documents into chunks.
- Ships with official random and media splits. The released data differs from the paper's:
  37,554 vs 34,737 articles, and a media split of 26,590 / 2,356 / 1,300 vs 22,969 / 5,098 / 1,200.
  The released media split does keep outlets separate: no web domain appears in both train and test.
- ChunkyBERT and Baly both train on this corpus, as their papers state.

### 3.2 `stories_10k` — the team's data, what the classifiers are evaluated on

10,000 news stories (8,000 train / 1,000 val / 1,000 test splits), each with four texts:

| Field | What | Label |
|---|---|---|
| `left` | article from a left outlet | Left |
| `center` | article from a centre outlet | Centre |
| `right` | article from a right outlet | Right |
| `neutral` | LLM-written neutral summary (Nemotron-3-Ultra) | none |

- That is **30,000 labelled articles** (10,000 per class) and **10,000 unlabelled summaries**. Only
  the `neutral` field is neutral; the other three are ordinary partisan news.
- **11 publishers**, with a lopsided Centre class:

| Side | Publishers (articles) |
|---|---|
| Left | NYT 2,854 · HuffPost 2,827 · Daily Kos 2,257 · Washington Post 1,199 · CNN 863 |
| Centre | **The Hill 6,089** · USA Today 2,001 · AP 1,910 |
| Right | Breitbart 4,644 · Washington Times 2,814 · Fox 2,542 |

- **The articles are short:** median 135 words (175 BERT tokens), and none exceeds 250 words, which
  suggests a length cap *(measured here)*. Compare 1,018 tokens for Baly articles. Summaries median
  273 tokens.
- **Overlap with the classifiers' training data:**
  - Negligible at the article level: 71 of 30,000 titles (0.2%) also appear in Baly's corpus.
  - Large at the outlet level: 10 of the 11 publishers are in Baly's corpus; HuffPost isn't.

---

## 4. The instruments

For each: what it is, how it was built, and whether it reproduced.

### 4.1 ChunkyBERT (Loiya et al., *Scientific Reports* 2026)

**What it is.** BERT reads at most 512 tokens. ChunkyBERT handles documents up to 5,000 tokens:
1. Split the document into 512-token chunks.
2. Encode each chunk with BERT.
3. Pass the token vectors through a 2-layer Transformer encoder.
4. Attention-pool each chunk into a single vector.
5. Average the chunk vectors and classify Left / Centre / Right.

**How it was built.** The authors released a single notebook. Their model and training code are used
verbatim (`chunkybert/model.py`, `train.py`), with two changes: a data-path option and a fix for a
syntax error that stopped the notebook from running. Trained on Kaggle (2×T4 GPUs), 5 epochs, the
paper's hyperparameters, about 30 minutes per epoch.

**Reproduces ✅**

| | Paper | Run 1 | Run 2 |
|---|---|---|---|
| Best val accuracy | 86.22% | 86.58% | 86.87% |
| Best val AUC | 0.962 | 0.9636 | 0.9651 |
| Epoch-1 val accuracy | 83.91% | 82.93% | 83.01% |

These are random-split numbers, the easier setting described in §2. ChunkyBERT was never evaluated on
a media split.

### 4.2 ChunkyBERT's comparison models: rebuilt, and where the paper doesn't hold

The paper argues its architecture is needed using two tables of comparison models (its Tables 8 and
10), but released no code for them. We rebuilt all 14 as one configurable model
(`replication/variants.py`) and trained each for one epoch, as the paper did. Two choices differ
from the authors' code, so these runs are comparable with each other but not exactly with the
authors' own ChunkyBERT:
- Pooling ignores padding tokens; the authors' code includes them.
- In this setup the full model scored 80.99% after one epoch, against 82.93% from the authors' code.

**Table 10, ablations.** Five of six variants land within 1.2 points of the paper. The sixth is off
by 41 points.

| Variant | Paper | Measured | Gap |
|---|---|---|---|
| Full model | 84 | 80.99 | −3.0 |
| No attention pooling | 82 | 81.43 | −0.6 ✅ |
| Simplified classifier head (bare linear layer) | 82 | 82.36 | +0.4 ✅ |
| Max- instead of mean-pooling over chunks | 81 | 81.84 | +0.8 ✅ |
| No chunking (first 512 tokens only) | 79 | 77.81 | −1.2 ✅ |
| BERT frozen | 58 | 58.15 | +0.2 ✅ |
| **No Transformers (2-layer encoder removed)** | **41** | **82.31** | **+41.3 ❌** |

The paper's strongest evidence is "removing the Transformer layers drops accuracy to 41%". Removing
them actually costs nothing. Two observations point to what the paper measured instead:
- **Training time.** The paper lists 5 minutes per epoch for this variant against 27 for the full
  model. Removing two small layers from a model whose cost is dominated by BERT (110M parameters)
  can't make it 5× faster; our version took 24 minutes.
- **Removing BERT itself** gives 37.2% in about 2 minutes on our hardware: close to the paper's 41%
  and in the right range of time.

So that row almost certainly reports a model without BERT. The rest of the table agrees that BERT
does the work:
- Freezing BERT costs 23 points.
- Every other change moves accuracy by at most about 3 points (dropping chunking costs 3.2).
- Four simplified variants score slightly *above* the full model. The gaps (≤1.4 points) are within
  single-run noise, so the honest reading is that the added layers make no measurable difference.

**Table 8, GRU/TCN baselines (partly reproduces).** The paper doesn't say what input these baselines
use. We tried the cheapest reading first: word embeddings trained from scratch, with no BERT.
- **At the paper's learning rate (2e-5)**, which suits fine-tuning BERT but is very low for a model
  trained from scratch, nothing trains:
  - TCN predicts a single class (36.57%).
  - GRU reaches 37–39%.
  - The paper's two GRU-without-attention rows (39.2, 36.4) match this. Most likely those runs
    failed to train as well.
- **At a standard from-scratch learning rate (1e-3):**
  - Both GRU-with-attention rows match the paper (77.4 vs 73.3; 75.7 vs 74.8).
  - GRU without attention reaches 69–72%, not 36–39%. The paper's 34–38 point "attention is
    essential" gap shrinks to about 6 points once the model trains.
- **TCN doesn't reproduce at either setting** (12–18 points low). We stopped there rather than
  search for settings that happen to hit the paper's number.
- **This rests on our reading of an underspecified baseline.** The alternative reading (BERT feeding
  a GRU/TCN) was not tested.

**Why this matters:** the replication caught real errors, and it separates what to trust in
ChunkyBERT (the BERT backbone and its in-distribution accuracy) from what not to (its claims about
which components matter).

### 4.3 Baly et al. 2020 — the outlet-vs-ideology test

**What it is.** The paper that introduced the media split. It shows BERT collapses on unseen outlets
and proposes two remedies:
- **Adversarial adaptation:** train BERT while a second head tries to predict the outlet, and
  reverse that head's gradient so the shared encoder unlearns outlet identity.
- **Triplet-loss pre-training:** pull same-ideology articles from different outlets together, and
  push different-ideology articles from the same outlet apart.

**How it was built.** No model code was released, so all six models were rebuilt from the paper in
one file (`baly/baly.py`).
- *Paper's settings:* BERT-base, 512 tokens, learning rate 2e-5, gradient clipping 1, early stopping
  on validation macro-F1, adversarial weight λ = 0.7.
- *Gradient reversal:* checked numerically; the encoder receives exactly −0.70× the outlet gradient.
- *Settings the paper omits*, recorded as assumptions: batch size 16, at most 5 epochs, triplet
  margin 1.0.

**Results** (test set; macro-F1 / accuracy / MAE; single run each):

| Model | Paper | Measured | |
|---|---|---|---|
| Majority-class baseline | 19.6 / 41.7 / 0.92 | 21.0 / 46.1 / 0.85 | ✅ |
| BERT, **random** split | 80.2 / 79.8 / 0.33 | 82.2 / 81.9 / 0.30 | ✅ |
| BERT, **media** split | 35.5 / 36.8 / 0.90 | 37.1 / 42.1 / 0.82 | ✅ |
| BERT + adversarial adaptation | 43.9 / 46.2 / 0.59 | 33.7 / 36.8 / 0.84 | ❌ |
| BERT + triplet pre-training | 48.3 / 51.4 / 0.51 | not run (on hold) | — |
| BERT outlet classifier (73 outlets) | 59.7 F1 / 80.1 acc | 48.3 F1 / 78.4 acc | ✅ accuracy |

- **The central claim reproduces:** 82 → 37 macro-F1 when test outlets are unseen. On unseen outlets
  BERT's accuracy even falls *below* the majority baseline (42.1% vs 46.1%), as in the paper. Its
  macro-F1 (37) is still well above the majority baseline's (21): the model predicts all three
  classes, just not reliably.
- **Outlet classifier:**
  - It names which of 73 outlets wrote an article 78% of the time, so outlet identity is easy to
    recover from text.
  - Its macro-F1 is 11 points below the paper's. It was still improving at the 5-epoch cap, and our
    73-outlet set includes smaller outlets than the paper's.
- **Adversarial adaptation didn't reproduce:** 3.4 F1 *below* plain BERT instead of 8 points above.
  - The outlet-predicting head does appear to have worked: its loss settled near chance level,
    inferred from the logged total loss.
  - So outlet information was suppressed, yet accuracy on unseen outlets didn't improve.
  - With one run and a validation set of only 12 outlets, on which accuracy stayed below chance
    throughout, this counts as "not reproduced", not "disproven".

### 4.4 Jev zero-shot ideology (TypeSafe)

**What it is.** A hosted model: send a text and a typed question with named options, and get back a
probability for each option. It was not trained on our labels.

**How it was built** (`jev/jev_classify.py`).
- **Choosing the question:** four question designs × {body only, title + body} were compared on 450
  held-out dev articles. The winner asks *"what kind of outlet most likely published this?"* (with
  the title), not *"what is this article's leaning?"*. That fits labels that track the outlet.
- **Prior correction:** a per-class correction fitted on the dev set didn't help on the full data,
  so the raw predictions are reported.
- **Cost:** $0.95 for about 38,500 requests.

### 4.5 Jev AllSides bias types

**What it is.** AllSides describes 16 types of media bias. Twelve can be judged from a single text,
and each became a yes/no question to Jev:
- spin, sensationalism, loaded words, subjective adjectives;
- opinion presented as fact, unsubstantiated claims, unattributed sources;
- mudslinging, mind reading, negativity, one-sidedness, flawed logic.

Five more questions ask about direction: whose vocabulary is used, which side the slant favours,
whose viewpoint comes first, who is attacked, and elite vs populist.

**Why it's here.** It asks about features of the writing rather than about the outlet, which the
ideology classifiers can't do. These are Jev's judgments; they have not been checked against human
annotation.

**Scope.** Run on 750 test stories × 4 texts = 3,000 texts, for $0.19. Not covered:
- story choice, placement and photo bias, which a single text can't show;
- omission, which needs a story's three articles compared side by side (not yet run).

### 4.6 VAD

Word-list scores from Warriner et al. (2013): 13,915 English words rated 1–9 for valence, arousal and
dominance, averaged over the listed words in each text (`evaluation/vad/vad_scores.py`). Built
separately from the replication work.

### 4.7 POLITICS (Liu et al. 2022): scoped, not run

**What it is.** RoBERTa further pretrained on about 2.3 million articles from the same 11 outlets as
`stories_10k`, with a triplet loss over same-story articles.

**What's available.** The pretrained model is public; the fine-tuning code isn't. Redoing the
pretraining would cost 160–240 GPU-hours.

**Agreed scope.** Reproduce the paper's ideology fine-tuning results (its Table 3) with the released
model against RoBERTa. Of the six ideology datasets:
- Congress Speech and Hyperpartisan are listed as openly downloadable (not yet downloaded).
- AllSides is available on request only.
- The YouTube and Twitter datasets can't be shared.

**Caveat for any later use on our data.** POLITICS was pretrained on our 11 publishers, so it may
have seen `stories_10k` articles.

---

## 5. Results on the team's data (`stories_10k`)

### 5.1 The 30,000 labelled articles

| Model | Trained on labels? | Accuracy | Macro-F1 | Recall L / C / R | Share predicted Right |
|---|---|---|---|---|---|
| **Jev zero-shot** (`outlet_style`)* | no | **47.8%** | **45.8** | 0.37 / **0.80** / 0.26 | 13% |
| ChunkyBERT (5 epochs) | yes | 44.3% | 40.7 | 0.49 / 0.13 / 0.71 | 55% |
| Baly BERT, random split | yes | 42.0% | 37.9 | 0.39 / 0.11 / 0.76 | 60% |
| ChunkyBERT, simplified head (1 epoch) | yes | 40.4% | 33.7 | 0.34 / 0.05 / 0.83 | 70% |
| Baly BERT + adversarial | yes | 37.0% | 35.5 | 0.27 / 0.25 / 0.59 | 49% |
| Baly BERT, media split | yes | 36.0% | 31.7 | 0.31 / 0.08 / 0.69 | 60% |

\* Jev is scored on 29,550 articles; its 450 dev articles are excluded. Chance is 33.3%.

Confusion matrix for ChunkyBERT (rows = true label, columns = predicted):

| | Left | Centre | Right |
|---|---|---|---|
| **Left** | 4,866 | 1,210 | 3,924 |
| **Centre** | 3,324 | 1,324 | 5,352 |
| **Right** | 1,783 | 1,118 | 7,099 |

Every model's matrix is in `evaluation/{chunkybert_eval,best_variant_eval,baly_eval}.ipynb` and
`jev/jev_eval.ipynb`. Per-article probability vectors are in `evaluation/scores/` and
`jev/results/jev_scores.csv`.

### 5.2 Accuracy by outlet *(measured here)*

| Outlet (label) | ChunkyBERT | Simplified head | Baly random | Baly media | Baly AA | Jev |
|---|---|---|---|---|---|---|
| CNN (L) | 82 | 76 | 78 | 82 | 79 | 12 |
| NYT (L) | 69 | 41 | 43 | 7 | 7 | 16 |
| Washington Post (L) | 40 | 25 | 26 | 29 | 23 | 15 |
| Daily Kos (L) | 37 | 27 | 41 | 33 | 27 | 75 |
| HuffPost (L) | 31 | 22 | 26 | 38 | 33 | 46 |
| The Hill (C) | 10 | 3 | 9 | 5 | 17 | 74 |
| USA Today (C) | 27 | 13 | 26 | 21 | 53 | 84 |
| AP (C) | 9 | 2 | 5 | 2 | 19 | 98 |
| Washington Times (R) | 80 | 90 | 81 | 90 | 83 | 19 |
| Breitbart (R) | 69 | 82 | 82 | 65 | 56 | 36 |
| Fox (R) | 65 | 78 | 60 | 56 | 39 | 16 |

### 5.3 What the numbers show

- **The trained models and Jev fail on almost opposite sets of outlets.** The trained models get the
  Washington Times, CNN and Breitbart right and miss every Centre outlet. Jev gets the Centre outlets
  and Daily Kos right and misses most Right outlets, CNN and NYT.
- **Centre fails for every trained model.** All three Centre outlets are mostly missed: The Hill at
  3–17% and AP at 2–19%. The Hill alone is 61% of the Centre class, so it dominates Centre recall.
- **For the Baly models, having seen an outlet in training barely helps on these short texts.** The
  media-split model scores 36.5% on outlets it trained on and 35.3% on outlets it didn't. Two
  qualifications:
  - The two groups contain different outlets, so this isn't a controlled comparison.
  - NYT is a clear exception: the model trained on NYT scores 43% on it, the two that weren't score 7%.
  - A plausible explanation, not tested: 135-word texts carry little of an outlet's house style.
- **The biases run in opposite directions.** Trained models predict Right 49–70% of the time; Jev
  predicts Centre 68% of the time. When instruments with opposite biases agree, the agreement is
  more informative than either alone.

### 5.4 The 10,000 neutral summaries (no labels)

| Model | Predicted Left / Centre / Right |
|---|---|
| ChunkyBERT | 38% / 4% / 58% |
| ChunkyBERT, simplified head | 10% / 0% / 90% |
| Baly BERT, random split | 7% / 5% / 88% |
| Baly BERT, media split | 22% / 3% / 75% |
| Baly BERT + adversarial | 25% / 21% / 54% |
| Jev (8,001 summaries scored) | 5% / 95% / 1% |

The trained models call most summaries Right; Jev calls nearly all of them Centre. That is each
model's overall bias from §5.1 showing up again, so these distributions can't be read as evidence
about the summaries. This is the clearest illustration of why one classifier can't certify
neutrality.

### 5.5 Bias types (Jev, 750 test stories)

- **Overall bias:** summaries score lowest, but only just below the centre articles (0.266 vs
  0.276). Story by story, the summary scores below the left article 71% of the time, the right
  article 59%, and the centre article 52% (no better than a coin flip).
- **The clearest improvement is one-sidedness:** 0.30 for summaries against 0.48–0.55 for the source
  articles.
- **Where summaries do worse than centre articles:** unattributed sources (+0.08) and mudslinging
  (+0.04). One possible reason: summarising drops names but keeps charged quotes.
- **By side:** left articles score highest on almost every type; right articles are close to centre
  ones.
- **Combining all 17 signals** in a logistic regression (cross-validated, with no story split across
  folds) predicts the label at 51.0% accuracy / 51.0 macro-F1 on these 2,250 labelled articles. This
  is the best ideology result on this data, but it is supervised.
- **Direction:** Jev rates the summaries' slant as favouring the left more often than the right (mean
  probability 0.40 vs 0.18).

### 5.6 VAD

The four text types are indistinguishable (*measured on all 40,000 texts, 1–9 scale*):

| | Valence | Arousal | Dominance |
|---|---|---|---|
| Range across the four text types | 5.39–5.43 | 4.03–4.04 | 5.47–5.49 |

Word-level emotional tone doesn't separate partisan from neutral text here.

A related warning comes from the team's earlier analysis (`docs/eda-evaluation.md`, branch
`data-finding-out`). It found the framing-bias metric used in this literature correlates 0.88 with
the amount of text scored, so a model can improve it simply by writing less.

---

## 6. The common thread: the outlet, more than the ideology

Several independent results point the same way:

1. **Baly media split:** BERT drops from 82 to 37 macro-F1 on unseen outlets.
2. **Outlet classifier:** text identifies its outlet 78% of the time across 73 outlets.
3. **`stories_10k`:** ChunkyBERT and the random-split Baly model, at 82–87% on Baly's random split,
   score 42–44% here.
   - Accuracy is high for a few distinctive outlets (CNN, Washington Times) and near zero for others
     (AP, The Hill).
4. **Jev:** asking "which kind of outlet?" outperformed asking "which leaning?".

**The counterpoint.** On unseen outlets BERT still beats the majority baseline on macro-F1 (37 vs
21), so the classifiers do pick up some ideology signal. The accurate statement is that **most** of
their accuracy comes from recognising the outlet.

This matters directly for the team's goal. A rewrite that removes an outlet's style will move these
classifiers whether or not it changes the slant.

---

## 7. Using these instruments to evaluate a neutraliser

Next step (in progress, no results yet): evaluate the two LED rewriters on the 1,000 held-out test
stories. Those 2,000 left and right articles were never seen in training. The approach:

- **Paired, not absolute.** Score each original and its rewrite with the same instrument and report
  the change: does P(original side) fall and P(Centre) rise? Each model's fixed bias applies to both
  texts, so comparing the pair reduces it.
- **Several instruments with opposite biases.** A shift toward Centre that appears under both the
  BERT models and Jev is more credible than one under either alone.
- **Label-recovery accuracy** (lower is better) measures how much ideology or outlet signal the
  rewrite still carries.
- **Slant from the bias types.** For each type (spin, loaded words, one-sidedness, …), how often the
  rewrite scores better than its own original.
- **Reference points.** The human centre article, the LLM summary and the Nemotron rewrites the
  models were trained to imitate, so "more neutral" is measured against something.
- **Content preservation**, measured separately with question-answering coverage on DiverseSumm and
  QuestEval, because the easiest way to look neutral is to say less.

---

## 8. Limitations

- **Labels follow the outlet:** 99.9% of Baly labels match the outlet's usual label, and 100% in
  `stories_10k`. An article counted as misclassified may simply be milder than its outlet.
- **One run per model** for all Baly and ablation results. Baly's media-split test set covers 13
  outlets, so differences of a few points there are not meaningful.
- **The released Baly data isn't the paper's data.** Results are comparable in kind, not exactly.
- **`stories_10k` texts are short** (median 135 words). ChunkyBERT's long-document design never
  matters here, and there is little text to classify.
- **Centre is dominated by The Hill** (61% of the class), so Centre recall mostly reflects The Hill.
- **Table 8 conclusions depend on our reading** of baselines the paper leaves unspecified.
- **Jev's bias-type scores are answers to individual questions,** not validated rates. Compare them
  across text types rather than reading the absolute values.
- **Jev's ideology run scored the original articles with their titles;** rewrites have none. Paired
  comparisons need the originals re-scored on body text only.
- **POLITICS may have been pretrained on `stories_10k` articles** (unverified).

---

## 9. Repository map

| Path | Contents |
|---|---|
| `eval_metrics.md` | The original list of required instruments |
| `chunkybert/` | ChunkyBERT from the authors' code, plus the Kaggle training kernel |
| `replication/` | Rebuilt ChunkyBERT comparison models; `REPLICATION.md` is the full log |
| `baly/` | Baly et al. rebuild (`baly.py`) and Kaggle kernels; `REPLICATION.md` is the full log |
| `evaluation/` | Notebooks scoring each model on `stories_10k`; `scores/` holds per-article probabilities; `vad/` |
| `jev/` | Jev ideology (`jev_eval.ipynb`) and bias types (`bias_types.ipynb`); `results/spend.json` tracks API spend |
| `led_eval/`, `qa_eval/` | Rewriter evaluation, in progress: inputs prepared, no results |
| `CONTEXT.md` | All results in one place |
| `CLAUDE.md` | How to run things, where data and weights live, compute rules |
| `data/`, `models/`, `external/` | Not in git: datasets, trained weights, upstream repositories (`CLAUDE.md` explains how to regenerate them) |

Compute: training ran on Kaggle (2×T4). A 4 GB laptop GPU was used for smoke tests and inference.
Jev spend to date is $1.14.

---

## 10. Likely questions

**"Isn't 44% accuracy just a weak model?"** The same models reach 82–87% when test outlets were seen
in training, and BERT falls to 37 macro-F1 on Baly's unseen-outlet split. The drop on our data
matches that pattern. Baly et al. report the same collapse.

**"Why believe your replications over the papers?"** Each disagreement has supporting evidence, not
just a different number:
- *The 41% ablation:* the paper's own 5-minute training time is impossible for the model it
  describes, and removing BERT comes close on both accuracy (37 vs 41) and training time.
- *The GRU gap:* the paper's no-attention numbers match our runs that verifiably failed to train.

**"Why does a zero-shot model beat fine-tuned ones?"** We don't know for certain. A plausible account
is that the fine-tuned models learned outlet cues from long articles, and the short excerpts here
carry few of them, while Jev wasn't trained on those cues. Jev has its own strong bias (toward
Centre), which is why we combine instruments rather than pick one.

**"Are the neutral summaries neutral?"** The classifiers can't settle it; they disagree in opposite
directions. The bias-type analysis says the summaries are much less one-sided than the source
articles and overall about as biased as the centre articles, with somewhat more unattributed claims
and mudslinging.

**"What's next?"**
- Paired evaluation of the LED rewriters (§7).
- Several runs of Baly's adversarial model.
- The triplet pre-training model.
- POLITICS Table 3.
