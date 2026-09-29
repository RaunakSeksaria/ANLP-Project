# Project context

State of branch `harshit` as of 2026-09-27. How to run things: `CLAUDE.md`. Open work: `TODO.md`.

## Goal

`eval_metrics.md` lists the evaluation tools the team needs: VAD scores, a publisher classifier (to
check publisher leakage), and political-ideology classifiers (Baly et al.; ChunkyBERT). This branch
builds the ideology classifiers, checks that each reproduces its paper, and applies them to the
team's `stories_10k` data (10,000 stories × {left, centre, right source article + LLM neutral summary}).

---

## 1. ChunkyBERT reproduced from the authors' code — `chunkybert/`

Loiya et al., *Sci. Rep.* 16, 15323 (2026). The model is BERT over 512-token chunks of documents up
to 5,000 tokens, then a 2-layer Transformer encoder, attention pooling and mean over chunks.

- Code is the authors' own notebook (`github.com/dloiya/chunkybert`), split into `model.py` and
  `train.py`. Two changes: a data-path flag, and a fix for an unterminated `print(` that stopped the
  notebook running.
- Data: Baly et al.'s Article-Bias-Prediction corpus, as the paper states. Class counts match the
  paper exactly.
- Trained twice on Kaggle (2×T4), 5 epochs, paper hyperparameters, 80/20 random stratified split.

| | Paper | Run 1 | Run 2 |
|---|---|---|---|
| Best val accuracy | 86.22% | 86.58% | 86.87% |
| Best val AUC | 0.962 | 0.9636 | 0.9651 |
| Epoch-1 val accuracy | 83.91% | 82.93% | 83.01% |

**Reproduces.** Run 2's final-epoch weights: `models/chunkybert_5ep.pt`, also uploaded to Kaggle as
`harshitlalwani/chunkybert-political-bias` (public visibility not verified).

## 2. ChunkyBERT on the team's data — `evaluation/chunkybert_eval.ipynb`

- **30,000 labelled source articles: 44.3% accuracy** (vs ~86% in-distribution). Macro-F1 0.41,
  macro AUC 0.66.
  - Recall is Left 0.49, Centre 0.13, Right 0.71.
  - Centre collapses: half of the centre articles are predicted Right.
- **10,000 neutral summaries** (no labels, so predicted classes only): 57.8% Right, 38.0% Left,
  4.2% Centre.
- Per-article probabilities: `evaluation/scores/source_article_scores.csv` and
  `evaluation/scores/neutral_summary_scores.csv`.

## 3. Rebuilding ChunkyBERT's comparison models — `replication/`

The authors released no code for Table 8 (GRU/TCN baselines) or Table 10 (ablations), so all were
rebuilt from the paper's text as one parameterised class. Full log: `replication/REPLICATION.md`.

### Table 8 — GRU/TCN baselines (partly reproduced)

- At the paper's learning rate (2e-5), every from-scratch baseline fails to train; most predict the
  majority class (36.57%).
- At a normal from-scratch learning rate (1e-3):
  - Both **GRU + attention** rows match: 77.4 vs 73.3, and 75.7 vs 74.8.
  - **GRU without attention** reaches 69–72%, where the paper reports 36–39%. The paper's
    "attention is essential" gap looks like two runs that never trained. Removing attention really
    costs about 6 points, not about 34.
- **TCN does not reproduce**: 12–18 points below the paper at every setting, and deeper TCNs were
  worse. Tuning stopped there rather than searching for settings that happened to hit the target.

### Table 10 — ablations (6 of 7 reproduced)

| Variant | Paper | Measured |
|---|---|---|
| Full Model | 84 | 80.99 |
| No Attention | 82 | 81.43 ✅ |
| No Classifier | 82 | **82.36** ✅ best |
| Max Pool | 81 | 81.84 ✅ |
| No Chunking | 79 | 77.81 ✅ |
| Freeze BERT | 58 | 58.15 ✅ |
| No Transformers | 41 | **82.31** ❌ |

**Key finding.** The paper's main architectural evidence, "No Transformers" at 41%, does not hold.
Removing the 2-layer Transformer costs nothing (82.31%).

- The paper's own timing gives it away: 5 min per epoch versus 27 for the full model, which is only
  possible if BERT itself was removed.
- Removing BERT reproduces the paper's number: 37.2% in about 2 min.
- So BERT does the work. The extra encoder layers presented as the contribution add nothing
  measurable, and the full ChunkyBERT scored below every simplified variant.

## 4. Best rebuilt variant on the team's data — `evaluation/best_variant_eval.ipynb`

"No Classifier" (ChunkyBERT with the head reduced to a bare Linear layer), retrained for 1 epoch with
weights saved (`models/no_classifier_1ep.pt`, 82.21% val).

| | pred Left | pred Centre | pred Right |
|---|---|---|---|
| true Left | 3354 | 1167 | 5479 |
| true Centre | 2231 | 456 | 7313 |
| true Right | 1028 | 662 | 8310 |

- Accuracy is **40.4%**, below ChunkyBERT's 44.3%. The comparison isn't clean: this model had 1
  epoch, ChunkyBERT had 5.
- Both models fail on Centre and lean heavily Right. That points to the training data rather than
  the architecture.

## 5. VAD scores — `evaluation/vad/` (added by a teammate, not in this line of work)

Word-list valence/arousal/dominance (Warriner et al. 2013, mean over words that appear in the list)
for all 40,000 texts. The four text types are indistinguishable:

| Text type | Valence | Arousal | Dominance |
|---|---|---|---|
| Left | 5.425 | 4.035 | 5.480 |
| Centre | 5.434 | 4.030 | 5.492 |
| Right | 5.418 | 4.036 | 5.479 |
| Neutral | 5.391 | 4.034 | 5.466 |

---

## 5b. Jev zero-shot (TypeSafe System One) — `jev/` (side experiment)

Zero-shot classification with TypeSafe's Jev model: one typed Choice question per article returns
probabilities for Left, Centre and Right. Four prompt designs were compared on 450 dev articles,
and "what kind of outlet most likely published this", with title + body, won. The labels are
outlet-level, so that framing matches how they were made.

| Model (29,550 labelled articles) | Accuracy | Macro-F1 | Recall L / C / R |
|---|---|---|---|
| **Jev zero-shot** | **47.8%** | **45.8** | 0.37 / 0.80 / 0.26 |
| ChunkyBERT (5 epochs) | 44.3% | 40.7 | 0.49 / 0.13 / 0.71 |
| No Classifier (1 epoch) | 40.4% | 33.7 | 0.34 / 0.05 / 0.83 |

- **Jev, never trained on these labels, beats both fine-tuned models.** Its bias is the mirror
  image of theirs: it over-predicts Centre, while they almost never predict it.
- 94.5% of the neutral summaries (8,001 scored) come out Centre.
- Cost $0.95 of a $1 budget. The spend ledger is in `jev/results/spend.json`.

## 5c. AllSides bias types with Jev — `jev/bias_types.ipynb`

750 test-split stories × 4 texts (left, centre and right articles plus the neutral summary) = 3,000
texts. One request per text carries 18 questions: 12 presence checks for the AllSides types a
single text can show, 5 direction questions, and `outlet_style`. Cost $0.19.

- **Neutral summaries have the lowest overall bias score, but only barely below the centre
  articles** (0.266 vs 0.276). The score is lower than the story's own left article in 71% of
  stories, the right article in 59%, and the centre article in 52% (a coin flip).
  - The big gain is **one-sidedness**: 0.30 vs 0.48–0.55.
  - They score *higher* than the centre articles on **unattributed sources** (+0.08) and
    **mudslinging** (+0.04), plausibly because summarising drops names and keeps the charged
    quotes.
- **Left articles score as the most biased on almost every type** (spin, opinion as fact, mind
  reading, sensationalism). Right articles sit close to the centre ones.
- **Direction questions:** `slant_side` is the most useful (41.1 macro-F1, balanced recall).
  `word_choice_side` and `attack_target` mostly answer "neither"; the excerpts rarely contain
  contested terms.
- **All signals + `outlet_style` in a logistic regression** (5-fold CV, grouped by story): **51.0
  accuracy / 51.0 macro-F1**, against 46.0 / 43.5 for `outlet_style` alone on the same articles. This
  step is supervised, since the regression learns from the labels.
- Jev reads the neutral summaries' slant as leaning left (0.40 vs 0.18 right), and more often has
  them disparaging the right (0.20) than the left (0.07).

## Cross-cutting takeaway

Every ChunkyBERT number above 80% comes from a **random** split, where the same outlets appear in
training and test. On the team's data, accuracy falls to 40–44%. The model has most likely learned
**outlet style rather than ideology**, which is exactly the failure Baly et al. 2020 designed its
media-based split to expose. That is the motivation for the Baly replication below, and for the
still-open publisher-leakage classifier.

## 6. Baly et al. 2020 replication — `baly/` (5 of 6 done)

"We Can Detect Your Bias" (arXiv 2010.05338). Agreed scope: **BERT models only**, the paper's
reported hyperparameters (no search), no Twitter-bio or Wikipedia media representations. No model
code was released, so all six are rebuilt from the paper (`baly/baly.py`). Full log:
`baly/REPLICATION.md`.

Test set, single seed. Media split unless noted.

| Model | Paper F1 / Acc / MAE | Measured F1 / Acc / MAE | |
|---|---|---|---|
| Majority | 19.61 / 41.67 / 0.92 | 21.03 / 46.08 / 0.85 | ✅ |
| BERT (random split) | 80.19 / 79.83 / 0.33 | 82.20 / 81.92 / 0.30 | ✅ |
| BERT (media split) | 35.53 / 36.75 / 0.90 | 37.13 / 42.08 / 0.82 | ✅ |
| BERT + adversarial adaptation | 43.87 / 46.22 / 0.59 | 33.68 / 36.85 / 0.84 | ❌ |
| BERT + triplet-loss pre-training | 48.26 / 51.41 / 0.51 | on hold | — |
| BERT outlet classifier (73-way) | 59.72 / 80.12 / — | 48.31 / 78.41 / — | ✅ acc, ❌ F1 |

- **The central claim reproduces.** BERT drops from 82.2 to 37.1 macro-F1 when test outlets are
  unseen, and on those outlets its accuracy is *below* the majority baseline — as in the paper. An
  outlet classifier names the source of an article 78% of the time. This confirms the cross-cutting
  takeaway above with a controlled experiment: BERT bias classifiers on this corpus mostly learn
  the outlet.
- **Adversarial adaptation did not reproduce.** It scored below plain BERT (−3.4 F1), not +8.3 above
  it, even though its adversary did strip outlet information (outlet loss settled at chance).
- **Single-seed results on this split are noisy.** Validation and test are 12–13 outlets each, and
  validation accuracy sat below chance for the whole run. Differences of a few points between two
  models here are within that noise.
**On the team's data** (`evaluation/baly_eval.ipynb`, 30,000 labelled source articles):

| Model | Macro-F1 | Acc | MAE | Recall L / C / R | Share predicted Right |
|---|---|---|---|---|---|
| Baly BERT (random split) | 37.9 | 42.0 | 0.76 | 0.39 / 0.11 / 0.76 | 60% |
| Baly BERT (media split) | 31.7 | 36.0 | 0.86 | 0.31 / 0.08 / 0.69 | 60% |
| Baly BERT + AA | 35.5 | 37.0 | 0.83 | 0.27 / 0.25 / 0.59 | 49% |
| ChunkyBERT (for comparison) | 40.7 | 44.3 | 0.75 | 0.49 / 0.13 / 0.71 | 55% |

- All Baly models land near their own unseen-outlet score (31–38 F1), even though 7–10 of the 11
  publishers here were in their training data. Seeing an outlet in training barely helps on these
  short texts: the media-split model scores 36.5% on outlets it trained on and 35.3% on ones it didn't.
- The exception is NYT: 42.5% for the random-split model, which trained on NYT, against 7% for the
  two media-split models, which did not.
- Centre fails for every model, mostly because The Hill (6,089 of the 10,000 centre articles) is
  classified correctly only 5–17% of the time. AA is the only model that predicts Centre with any
  regularity (25% recall), at the cost of Left and Right.
- On the unlabelled neutral summaries, the random-split model predicts Right for 88% of them.

- The released data is not the paper's (37,554 vs 34,737 articles; media split 26,590 / 2,356 / 1,300
  vs 22,969 / 5,098 / 1,200). The media split is still outlet-disjoint by domain.

## Open issues

- Publisher-leakage classifier (`eval_metrics.md` item 2) not built.
- Nothing is committed to git yet.
- Table 8 under the "BERT feeds the GRU/TCN" reading was never tested.
