# Baly et al. 2020 replication — progress log

Baly, Da San Martino, Glass & Nakov, *We Can Detect Your Bias: Predicting the Political Ideology
of News Articles*, EMNLP 2020 (arXiv 2010.05338).

**Status: 5 of 6 models done. TLP (triplet-loss pre-training) on hold pending a decision.**

## Scope (agreed)

BERT models only. The paper's reported hyperparameters, with no search. No LSTM models. No
Twitter-bio or Wikipedia media-level representations (Table 6 rows 3–8).

| # | Model | Paper table | Code |
|---|---|---|---|
| 1 | Majority class | 3 | `baly.py majority` |
| 2 | BERT, media-based split | 3, 5 ("None") | `baly.py bert --split media` |
| 3 | BERT, random split | 3 | `baly.py bert --split random` |
| 4 | BERT outlet classifier | 4 | `baly.py medium` |
| 5 | BERT + adversarial media adaptation | 5 | `baly.py aa` |
| 6 | BERT + triplet-loss pre-training | 5 | `baly.py tlp` |

No model code was released (the official repo holds only data and splits), so all six are rebuilt
from Sections 4–5 of the paper.

## Setup

**From the paper (Sec 5.1):** `bert-base-uncased`, 512-token input, lr 2e-5, gradient clipping 1.0,
early stopping on validation macro-F1, AA λ = 0.7, 35,017 triplets sharing a topic. Metrics are
macro-F1, accuracy and MAE on the test set.

**Not stated by the paper, so assumed:**

| Setting | Value |
|---|---|
| Batch size | 16 |
| Max epochs | 5 (early stopping patience 2) |
| Schedule | AdamW, weight decay 0.01, linear warmup 10% |
| Classifier head | `pooler_output` → dropout 0.1 → linear (standard BERT fine-tuning) |
| Triplet margin ε | 1.0 |
| Triplet pre-training epochs | 1 |
| Triplet batch | 8 triplets |
| Seed | 42 |

**Implementation checks:**
- Gradient reversal verified numerically: the gradient reaching the encoder from the media loss is
  exactly −0.70× the direct gradient.
- Triplet sampling follows the paper's rule: positive = same ideology, *different* outlet; negative
  = different ideology, *same* outlet; all three share a topic. 9,220 of 26,590 training articles
  qualify as anchors. Negatives come mostly from the 288 articles (1.1%) whose label differs from
  their outlet's usual one, which is inherent to the design.

## Data differences from the paper

The released corpus and splits are not the ones in the paper's Table 2.

| | Paper | Released (used here) |
|---|---|---|
| Corpus | 34,737 articles, 73 media | 37,554 articles, 467 domains |
| Media split train / valid / test | 22,969 / 5,098 / 1,200 | 26,590 / 2,356 / 1,300 |
| Media in media split | 46 / 15 / 12 | 54 / 12 / 13 |
| Random split train / valid / test | 26,828 / 6,709 / 1,200 | 27,978 / 6,996 / 1,300 |

- The released media split is still outlet-disjoint by **domain** (`source_url`): 0 domains are
  shared between train and test, or between train and valid. By the `source` name it is not
  (12 names shared), so "outlet" here means domain throughout, including AA's adversary.
- **The media split is domain-disjoint, but not organisation-disjoint.** `www.cnn.com` (2,584
  articles) and `www.money.cnn.com` are in train, while `www.politicalticker.blogs.cnn.com` is in
  test (100 of the 1,300 test articles). So 7.7% of the "unseen outlet" test set comes from an
  organisation the model trained on.
- **Outlet classifier (Table 4):** the paper's split isn't released. As the paper describes, I made
  a stratified random 80/10/10 split over the **73 most frequent domains**, to match its 73-way
  task. The 1,436 articles from smaller domains (3.8%) are excluded.

## Results

Test set, media split unless noted. Single seed (42). Δ = measured − paper, in points.

| Model | Paper F1 | Paper Acc | Paper MAE | F1 | Acc | MAE | Δ F1 | Verdict |
|---|---|---|---|---|---|---|---|---|
| Majority | 19.61 | 41.67 | 0.92 | 21.03 | 46.08 | 0.85 | +1.4 | ✅ |
| BERT (random split) | 80.19 | 79.83 | 0.33 | **82.20** | **81.92** | **0.30** | +2.0 | ✅ |
| BERT (media split) | 35.53 | 36.75 | 0.90 | **37.13** | **42.08** | **0.82** | +1.6 | ✅ |
| BERT + adversarial adaptation | 43.87 | 46.22 | 0.59 | 33.68 | 36.85 | 0.84 | −10.2 | ❌ |
| BERT + triplet-loss pre-training | 48.26 | 51.41 | 0.51 | — | — | — | — | on hold |
| BERT outlet classifier (73-way) | 59.72 | 80.12 | — | 48.31 | 78.41 | — | −11.4 | ✅ acc / ❌ F1 |

Every run trained 5 epochs at 10.5–11.7 min each on 2×T4. No run triggered early stopping before
the epoch cap.

### What reproduces

**The paper's central claim reproduces cleanly.** The same BERT scores 82.2 macro-F1 when test
outlets were seen in training and 37.1 when they were not — a 45-point collapse, matching the
paper's 80.2 → 35.5. BERT on unseen outlets is barely above the majority baseline (37.1 vs 21.0 F1),
and in accuracy it falls *below* it (42.1% vs 46.1%) — as in the paper (36.8% vs 41.7%). Most of what
a BERT bias classifier learns on this corpus is outlet identity.

The outlet classifier confirms it from the other side: 78.4% accuracy at naming which of 73 outlets
wrote an article (paper 80.1%). Its macro-F1 is 11 points lower, but that metric was still climbing
at the epoch cap (val 21.5 → 46.0 over 5 epochs) and weights the smallest outlets equally; our
top-73 set reaches further into the corpus's long tail (smallest class 44 articles).

### What does not

**Adversarial adaptation did not help — it scored 3.4 F1 *below* plain BERT** (33.7 vs 37.1), where
the paper reports +8.3. The adversary itself worked: the combined training loss settled near 4.25,
and with the ideology loss at ~0.1–0.2 that leaves the outlet loss at ≈ ln 54 = 3.99, chance level
for 54 training outlets (inferred from the logged total; the two terms were not logged separately).
So outlet information *was* stripped from the representation, yet ideology prediction on unseen
outlets did not improve.

**How much to trust a single run here.** The media-split validation set is 12 outlets and its
accuracy stayed *below chance* (22–30%) in every epoch of both runs, while the 13-outlet test set
gave 37–42%. With so few outlets per set, which outlets land in validation vs test swings results by
several points, and early stopping on that validation set is close to noise. A 3–10 point gap
between two single-seed runs is inside that range. The fair statement is that AA's gain **was not
reproduced** here, not that it is disproven; confirming either way needs several seeds.
## Run log

- **2026-09-27** — Read the paper; confirmed no model code was released. Verified that the released
  splits differ from Table 2 but the media split is domain-disjoint. Scope cut to BERT only, no
  hyperparameter search, no Twitter/Wikipedia.
- **2026-09-27** — Wrote `baly.py` (all six models, one file). Majority baseline run. Smoke-tested
  all five BERT models locally, which found and fixed: the outlet classifier accidentally getting
  an adversarial head, MAE being reported for the 79-way task, and TLP failing on tiny subsets.
  Settled "outlet" as domain and the outlet task as top-73 domains.
- **2026-09-27** — Uploaded private dataset `harshitlalwani/baly-article-bias` (article table +
  `splits/`). Pushed `kaggle/main/` as kernel `baly-bert-main`: BERT media → BERT+AA → BERT random →
  outlet classifier, up to ~4 h. Per the user, TLP (`kaggle/tlp/`, ~1.7 h) waits until these
  results are in.
- **2026-09-27 16:22 → 18:31** — Kernel sat in Kaggle's T4 queue ~2 h (user chose to keep waiting
  rather than delete and move to P100), then ran ~3.8 h cleanly.
- **2026-09-29** — Results in. Random-split and media-split BERT, majority and outlet-classifier
  accuracy reproduce. AA does not (−10 F1 vs paper, −3.4 vs our own baseline). Outputs in
  `kaggle/main/output/`; model weights left on Kaggle (4 × ~440 MB, not downloaded).
- **2026-09-29** — Downloaded the three ideology models' weights to `models/baly/` and scored the
  team's `stories_10k` in `evaluation/baly_eval.ipynb`. Checked contamination first: only 71 of
  30,000 `stories_10k` titles (0.2%) occur in Baly's corpus. Outlet overlap is large, though: 7 of
  its 11 publishers are in the media-split training set, 10 in the random split's, and HuffPost is
  in neither.
