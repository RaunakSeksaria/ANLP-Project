# Results — all evaluation work to date

Every measured number from this branch in one place, as of 2026-10-13. Background and method:
`EVALUATION_WALKTHROUGH.md`. Full logs: `replication/REPLICATION.md`, `baly/REPLICATION.md`.
Single seed throughout unless stated. Labels: 0 = Left, 1 = Centre, 2 = Right (outlet-level).

**Contents**
1. Replications (do the instruments work on their own papers' data?)
2. `stories_10k`: the 30,000 labelled source articles
3. `stories_10k`: the 10,000 neutral summaries
4. Nemotron single-article rewrites (short and long sets)
5. LED neutraliser outputs
6. Bias types and VAD
7. Status: what is done, partial, or pending
8. Appendix: confusion matrices

---

## 1. Replications

### 1.1 ChunkyBERT (Loiya et al. 2026), authors' code, random 80/20 split

| | Paper | Run 1 | Run 2 |
|---|---|---|---|
| Best val accuracy | 86.22% | 86.58% | 86.87% |
| Best val AUC | 0.962 | 0.9636 | 0.9651 |
| Epoch-1 val accuracy | 83.91% | 82.93% | 83.01% |

✅ Reproduces.

### 1.2 ChunkyBERT Table 10, ablations (1 epoch, rebuilt — no upstream code)

| Variant | Paper | Measured |
|---|---|---|
| Full model | 84 | 80.99 |
| No attention | 82 | 81.43 ✅ |
| No classifier (bare linear head) | 82 | 82.36 ✅ |
| Max pool | 81 | 81.84 ✅ |
| No chunking | 79 | 77.81 ✅ |
| Freeze BERT | 58 | 58.15 ✅ |
| No Transformers | 41 | 82.31 ❌ |
| *No BERT at all (our test of what the paper measured)* | *—* | *37.16, 1.9 min/epoch* |

The paper's "No Transformers = 41%, 5 min" matches a model with BERT removed, not the ablation described.

### 1.3 ChunkyBERT Table 8, GRU/TCN baselines (rebuilt, standalone embeddings)

| Model | Paper | lr 2e-5 (paper's) | lr 1e-3 |
|---|---|---|---|
| GRU, no chunk, no attn | 39.16 | 37.68 ✅ | 71.88 |
| GRU, no chunk, attn | 73.29 | 39.25 | 77.42 ✅ |
| GRU, chunk, no attn | 36.44 | 37.32 ✅ | 69.26 |
| GRU, chunk, attn | 74.82 | 38.82 | 75.68 ✅ |
| TCN, no chunk, no attn | 71.11 | 36.57 | 58.85 |
| TCN, no chunk, attn | 68.02 | 36.57 | 53.72 |
| TCN, chunk, no attn | 72.28 | 36.57 | 54.48 |
| TCN, chunk, attn | 69.62 | 36.57 | 54.88 |

GRU rows reproduce only with different learning rates per row; TCN does not reproduce. 36.57% is the
majority-class share (a model predicting one class).

### 1.4 Baly et al. 2020 (rebuilt — no upstream model code)

| Model | Paper F1 / Acc / MAE | Measured F1 / Acc / MAE | |
|---|---|---|---|
| Majority class | 19.6 / 41.7 / 0.92 | 21.0 / 46.1 / 0.85 | ✅ |
| BERT, random split | 80.2 / 79.8 / 0.33 | 82.2 / 81.9 / 0.30 | ✅ |
| BERT, media split (unseen outlets) | 35.5 / 36.8 / 0.90 | 37.1 / 42.1 / 0.82 | ✅ |
| BERT + adversarial adaptation | 43.9 / 46.2 / 0.59 | 33.7 / 36.8 / 0.84 | ❌ |
| BERT + triplet pre-training | 48.3 / 51.4 / 0.51 | not run | — |
| BERT outlet classifier (73-way) | 59.7 F1 / 80.1 acc | 48.3 F1 / 78.4 acc | ✅ acc |

### 1.5 POLITICS (Liu et al. 2022), released checkpoint fine-tuned on Baly's corpus

Pretrained checkpoint `launch/POLITICS`; no fine-tuning code was released, so head and optimiser follow
the paper's Table A7 (epochs capped at 5, patience 2, to match the Baly models). This is **not** the
paper's Table 3 (that replication is pending); it is POLITICS used as a classifier on Baly's benchmark.

| Model | Split | Macro-F1 | Acc | MAE |
|---|---|---|---|---|
| **POLITICS** | random | **89.2** | **88.9** | **0.18** |
| BERT (ours) | random | 82.2 | 81.9 | 0.30 |
| **POLITICS** | media | **52.0** | **53.4** | **0.55** |
| BERT (ours) | media | 37.1 | 42.1 | 0.82 |

⚠️ POLITICS was pretrained on 3.6M articles from the same 11 outlets as `stories_10k`, including
Breitbart, AP and CNN, which are in Baly's media-split test set. Its "unseen outlet" score is not
unseen for it. Treat POLITICS numbers as an upper bound.

---

## 2. `stories_10k`: 30,000 labelled source articles

10,000 stories × {left, centre, right} article, short excerpts (~200 tokens), balanced classes,
11 publishers. Chance = 33.3%.

| Model | Trained on | Acc | Macro-F1 | MAE | Recall L / C / R | Predicted Right |
|---|---|---|---|---|---|---|
| **POLITICS, media split** ⚠️ | Baly media | **66.5%** | **66.0** | **0.38** | 0.72 / 0.48 / 0.79 | 40% |
| **POLITICS, random split** ⚠️ | Baly random | 63.2% | 59.0 | 0.42 | 0.83 / 0.22 / 0.84 | — |
| Jev zero-shot (`outlet_style`)* | — | 47.8% | 45.8 | — | 0.37 / 0.80 / 0.26 | — |
| ChunkyBERT (5 epochs) | Baly random | 44.3% | 40.7 | 0.75 | 0.49 / 0.13 / 0.71 | 55% |
| Baly BERT, random split | Baly random | 42.0% | 37.9 | 0.76 | 0.39 / 0.11 / 0.76 | 60% |
| No Classifier variant (1 epoch) | Baly random | 40.4% | 33.7 | 0.81 | 0.34 / 0.05 / 0.83 | 70% |
| Baly BERT + adversarial | Baly media | 37.0% | 35.5 | 0.83 | 0.27 / 0.25 / 0.59 | 49% |
| Baly BERT, media split | Baly media | 36.0% | 31.7 | 0.86 | 0.31 / 0.08 / 0.69 | 60% |
| Jev bias types + logistic regression (supervised, CV)† | — | 51.0% | 51.0 | — | — | — |

\* 29,550 articles (450 dev excluded); title + body. † 750 test stories, story-grouped 5-fold CV.

**Per-publisher accuracy, selected models** (centre is where everything fails):

| Publisher (side) | ChunkyBERT | Baly random | Baly media | POLITICS media | POLITICS random |
|---|---|---|---|---|---|
| The Hill (C), 6,089 articles | 10.1 | 8.5 | 5.3 | 47.6 | 18.1 |
| USA Today (C) | 26.9 | 25.6 | 20.6 | 65.6 | 37.8 |
| AP (C) | 8.8 | 4.7 | 2.2 | 31.3 | 17.0 |
| NYT (L) | 68.9 | 42.5 | 7.4 | 61.5 | 90.6 |
| Breitbart (R) | 69.0 | 81.8 | 64.5 | 84.5 | 91.3 |

Outlet seen in training barely helps the BERT models on these excerpts: Baly media scores 36.5% on
outlets it trained on vs 35.3% on unseen ones. NYT is the exception (42.5% when seen vs 7% when not).

---

## 3. `stories_10k`: 10,000 neutral summaries (no labels)

LLM-written (Nemotron-3-Ultra, prompt v3) from each story's left + right articles.

| Model | Predicted Left / Centre / Right | Mean P(L / C / R) |
|---|---|---|
| ChunkyBERT | 38.0 / 4.2 / 57.8% | .38 / .07 / .56 |
| No Classifier variant | 10.0 / 0.3 / 89.7% | .21 / .06 / .73 |
| Baly BERT, random split | 7.2 / 5.1 / 87.6% | .10 / .17 / .73 |
| Baly BERT, media split | 22.3 / 2.5 / 75.2% | .24 / .11 / .65 |
| Baly BERT + adversarial | 24.9 / 20.9 / 54.3% | .27 / .30 / .43 |
| POLITICS, media split ⚠️ | 21.2 / 57.4 / 21.4% | .25 / .53 / .22 |
| POLITICS, random split ⚠️ | 30.8 / 27.2 / 42.0% | .32 / .28 / .41 |
| **Jev zero-shot** | 4.5 / **94.6** / 1.0% | .11 / .84 / .05 |

The verdict depends on the classifier more than on the summaries: BERT models lean Right, Jev
leans Centre, POLITICS depends on its training split.

---

## 4. Nemotron single-article rewrites

Two sets of Raunak's data, each pairing an original left or right article with a Nemotron rewrite
(prompt n2, "remove framing, add nothing"). No centre articles.

| Set | Source | Pairs | Words, original → rewrite |
|---|---|---|---|
| Short | `origin/data` `{left,right}_neutral_dedup.jsonl` (LED training targets) | 20,000 | 163 → 142 |
| Long | HF `anlp-project-datasets` `data/long/` ("full1024") | 19,922 | 432 → 375 |

Pairs whose original also appears in Baly's corpus (the classifiers' training data) are excluded:
30 short, 139 long.

### 4.1 Fine-tuned classifiers, paired original → rewrite (ChunkyBERT and Baly random only so far)

"Acc" = share classified as the article's true side.

| Set | Model | Acc orig → rewrite | Mean P(L/C/R) orig → rewrite | Moved to Centre |
|---|---|---|---|---|
| Short | ChunkyBERT | 61.5% → 56.5% | .34/.11/.55 → .26/.09/.65 | 4.5% |
| Short | Baly random | 58.8% → 52.0% | .27/.17/.55 → .12/.16/.72 | 4.0% |
| Long | ChunkyBERT | 64.3% → 57.8% | .34/.16/.51 → .28/.12/.60 | 6.3% |
| Long | Baly random | 59.1% → 51.2% | .25/.17/.57 → .10/.17/.73 | 4.7% |

By side, predicted L / C / R, original → rewrite:

| Set | Model | Left articles | Right articles |
|---|---|---|---|
| Short | ChunkyBERT | 51/11/38% → 35/8/57% | 17/10/72% → 16/5/78% |
| Short | Baly random | 40/17/42% → 13/9/78% | 11/12/77% → 5/4/91% |
| Long | ChunkyBERT | 55/15/30% → 40/12/47% | 12/15/73% → 16/9/76% |
| Long | Baly random | 39/17/45% → 11/11/78% | 8/12/80% → 3/5/91% |

The BERT classifiers move the rewrites toward **Right**, not Centre. This matches their general
bias on plain text (§3) more than anything about the rewrites.

### 4.2 Jev zero-shot on the rewrites (body only)

| Texts | n | Predicted L / C / R | Left articles → L/C/R | Right articles → L/C/R | Still own side |
|---|---|---|---|---|---|
| Nemotron rewrites, short | 20,000 | 12.8 / 78.0 / 9.2% | 18 / 78 / 4% | 7 / 78 / 15% | 16.6% |
| Nemotron rewrites, long | 19,922 | 11.8 / 79.3 / 8.9% | 19 / 79 / 3% | 5 / 80 / 15% | 16.8% |
| *Reference: `stories_10k` original L/R articles (title + body)* | 19,700 | 22.8 / 61.4 / 15.8% | 37 / 57 / 6% | 8 / 66 / 26% | 31.6% |

The reference row is not a strict pairing (different articles, scored with titles). Re-scoring
the short/long originals body-only would make it exact (~$1.25).

---

## 5. LED neutraliser outputs

Team models `RaunakSeksaria/anlp-neutral-led`, λ0 and λ0.35, run on the 2,000 held-out
`stories_10k` test left/right articles (never seen in LED training). Greedy decoding, 512 tokens in
and out; none truncated or cut off.

**Jev zero-shot:**

| Model | n | Predicted L / C / R | Left articles → L/C/R | Right articles → L/C/R | Still own side |
|---|---|---|---|---|---|
| LED λ0 | 2,000 | 13.2 / 77.1 / 9.7% | 19 / 76 / 5% | 7 / 79 / 14% | 16.9% |
| LED λ0.35 | 2,000 | 13.0 / 77.5 / 9.6% | 19 / 76 / 5% | 7 / 79 / 14% | 16.7% |
| *Nemotron targets, short set (§4.2)* | 20,000 | 12.8 / 78.0 / 9.2% | 18 / 78 / 4% | 7 / 78 / 15% | 16.6% |

On this measure the LED outputs are indistinguishable from their Nemotron training targets, and
λ0 vs λ0.35 makes no difference. Fine-tuned classifiers have not been run on the LED outputs yet.

---

## 6. Bias types and VAD

### 6.1 AllSides bias types (Jev, 750 test stories × 4 texts)

- Neutral summaries have the lowest overall bias score, barely below centre articles (0.266 vs
  0.276). The summary scores lower than the story's left article in 71% of stories, the right
  article in 59%, the centre article in 52%.
- Largest gain: one-sidedness, 0.30 vs 0.48–0.55 for source articles.
- Summaries are worse than centre articles on unattributed sources (+0.08) and mudslinging (+0.04).
- Left articles score most biased on almost every type; right articles sit near centre.
- `slant_side` is the most informative direction question (41.1 macro-F1).

### 6.2 VAD (Warriner lexicon, 1–9 scale)

| Texts | Valence | Arousal | Dominance |
|---|---|---|---|
| `stories_10k` left | 5.425 | 4.035 | 5.480 |
| `stories_10k` centre | 5.434 | 4.030 | 5.492 |
| `stories_10k` right | 5.418 | 4.036 | 5.479 |
| `stories_10k` neutral summaries | 5.391 | 4.034 | 5.466 |
| Short set: original → rewrite | 5.429 → 5.413 | 4.038 → 4.023 | 5.478 → 5.478 |
| Long set: original → rewrite | 5.402 → 5.392 | 4.057 → 4.046 | 5.453 → 5.453 |

Word-level affect barely separates anything.

---

## 7. Status

| Item | State |
|---|---|
| Replications §1 | Done, except Baly triplet pre-training (on hold) and POLITICS Table 3 (Congress Speech / Hyperpartisan; data located, not run) |
| Classifiers on `stories_10k` §2–3 | Done for all 8 classifiers |
| Classifiers on Nemotron rewrite sets §4.1 | ChunkyBERT + Baly random done on both sets. **Paused:** Baly media (long set 51% done), Baly AA, outlet classifier, No Classifier, on both sets |
| Classifiers on LED outputs | Not started |
| Jev on all neutral texts §3–5 | Done |
| Publisher-leakage (outlet classifier on rewrites) | Not started (part of the paused queue) |
| Jev spend | $2.43 total (cap $2.60) |

---

## 8. Appendix: confusion matrices (`stories_10k`, 30,000 labelled articles)

Rows = true, columns = predicted (Left, Centre, Right).

| Model | True Left | True Centre | True Right |
|---|---|---|---|
| POLITICS, media split | 7,200 / 2,050 / 750 | 2,023 / 4,807 / 3,170 | 445 / 1,618 / 7,937 |
| POLITICS, random split | 8,344 / 943 / 713 | 3,687 / 2,182 / 4,131 | 740 / 822 / 8,438 |
| Jev zero-shot (29,550) | 3,684 / 5,595 / 571 | 1,210 / 7,898 / 742 | 802 / 6,500 / 2,548 |
| ChunkyBERT | 4,866 / 1,210 / 3,924 | 3,324 / 1,324 / 5,352 | 1,783 / 1,118 / 7,099 |
| No Classifier variant | 3,354 / 1,167 / 5,479 | 2,231 / 456 / 7,313 | 1,028 / 662 / 8,310 |
| Baly BERT, random split | 3,877 / 1,838 / 4,285 | 2,651 / 1,122 / 6,227 | 1,176 / 1,232 / 7,592 |
| Baly BERT, media split | 3,075 / 2,226 / 4,699 | 2,899 / 776 / 6,325 | 1,755 / 1,297 / 6,948 |
| Baly BERT + adversarial | 2,669 / 3,375 / 3,956 | 2,622 / 2,475 / 4,903 | 1,870 / 2,187 / 5,943 |

Nemotron rewrite sets, originals only (true Left / true Right rows):

| Set | Model | True Left | True Right |
|---|---|---|---|
| Long | ChunkyBERT | 5,486 / 1,508 / 2,943 | 1,182 / 1,429 / 7,235 |
| Long | Baly random | 3,841 / 1,666 / 4,430 | 789 / 1,216 / 7,841 |

**Where to find per-text scores:** `evaluation/scores/` (`stories_10k`), `evaluation/{short,long10k}/scores/`
(rewrite sets), `jev/results/` (all Jev), `politics/kaggle/{media,random}/output/results/` (POLITICS),
`led_eval/out/` (LED outputs).
