# POLITICS ideology classifier, version 2 (Left / Centre / Right)

A news-article ideology classifier: the POLITICS language model (Liu et al., 2022) fine-tuned on the
Article-Bias-Prediction corpus (Baly et al., 2020). Input is an article body; output is
P(Left), P(Centre), P(Right).

**Version 2** replaces the single `politics_random.pt` / `politics_media.pt` of version 1 with
4 random-split seeds trained with sliding windows, 10-epoch training with early stopping, per-epoch
validation and test metrics, and temperature calibration. The two version-1 files are still included
unchanged (see Files).

## What changed from version 1, and why

| Issue in version 1 | Fix in version 2 |
|---|---|
| Media-split checkpoint chosen on a validation set that was at chance every epoch (val macro-F1 32.8), so the choice was effectively random | Random split only. Its validation set contains the same outlets as test, so validation tracks test (see per-epoch table) |
| Max 5 epochs, patience 2 | Max 10 epochs, patience 4 on validation macro-F1 (paper, Table A7) |
| Articles truncated at 512 tokens | 512-token sliding windows with 64-token overlap (paper, Table A7) for **training and inference**; article logits = mean of window logits. Up to 8 windows (~3,600 tokens) per article |
| One seed; a rerun moved predicted-Centre share on neutral summaries by ~11 points | 4 seeds (42, 43, 44, 45), mean ± std reported, plus a seed ensemble |
| Only the best checkpoint, and `politics_random.pt` missing | First, best and last checkpoint for every seed |
| No per-class results | Per-class precision / recall / F1 and confusion matrices, every epoch |
| Over-confident probabilities | Temperature scaling fitted on validation; ECE before and after; reliability plot |

Not changed: the encoder, head, optimiser and data. The data is used as released (no extra masking of
outlet names); see Caveats for pretraining overlap.

## Files

| File | What |
|---|---|
| `seed<S>_best.pt` | Best epoch by validation macro-F1, one per seed. **Use these.** |
| `seed<S>_first.pt`, `seed<S>_last.pt` | Epoch 1 and the final epoch, per seed |
| `calibration.json` | Temperature per seed and for the ensemble |
| `politics_classifier.py` | Model class, sliding-window inference, ensembling, metrics |
| `report.json`, `summary.md`, `epoch_curves.json` | All metrics: per seed, per epoch, per class, ensemble, calibration |
| `epoch_curves.png`, `reliability.png` | Validation/test macro-F1 per epoch; ensemble reliability before/after calibration |
| `stories_10k_metrics.json` | Ensemble scores on the team's stories_10k (see Caveats) |
| `politics_media.pt`, `politics_random.pt` | **Legacy version-1 models**, unchanged, kept so existing scripts that load them by name keep working. Superseded by the seed files; `politics_media.pt`'s checkpoint was selected on a chance-level validation set |

Checkpoints are `state_dict`s for the same `PoliticsClassifier` class as version 1, stored in fp16 to halve
the download. `load_state_dict` casts them into the fp32 model, so loading code is unchanged.

## Usage

```bash
pip install torch transformers numpy
```

```python
from politics_classifier import load, predict

# recommended: calibrated ensemble of all seeds, sliding windows
models, temps = load(["seed42_best.pt", "seed43_best.pt", "seed44_best.pt", "seed45_best.pt"], "calibration.json")
probs = predict(models, ["...article body...", "...another..."], temperature=temps["ensemble"])
# probs[i] = [P(Left), P(Centre), P(Right)]

# one seed
models, temps = load("seed42_best.pt", "calibration.json")
probs = predict(models, texts, temperature=temps["seed42"])
```

Version-1-style code also works with these files (`PoliticsClassifier`, `torch.load` + `load_state_dict`,
512-token truncation), but loses the sliding windows and calibration.

- Pass the **article body only** (no title), as in training.
- Labels: 0 Left, 1 Centre, 2 Right.
- Loading prints warnings about unused `lm_head.*` and missing `pooler.*` weights. Both are expected.

## How it was trained

| Setting | Value | Source |
|---|---|---|
| Base model | `launch/POLITICS` (RoBERTa-base further pretrained on ~3.6M news articles) | paper |
| Classifier input | `<s>` vector of the last layer, per window | paper §5.4 |
| Head | Dropout 0.1 → Linear(768, 768) → Tanh → Dropout 0.1 → Linear(768, 3) | Table A7; Tanh ours |
| Long inputs | 512-token windows, 64-token overlap, mean of window logits; ≤ 8 windows per article | Table A7 |
| Data | Article-Bias-Prediction, released **random** split: 27,978 / 6,996 / 1,300 (train / val / test); body only | Baly repo |
| Optimiser | AdamW, lr 2e-5, weight decay 0.001, betas (0.9, 0.999) | Table A7 |
| Schedule | 6% warm-up, linear decay | Table A7 |
| Epochs | Max 10, early stopping patience 4 on validation macro-F1; best epoch kept | Table A7 |
| Batch | 16 articles per optimiser step (all their windows), gradient clipping 1.0, fp16 autocast | ours |
| Seeds | 42, 43, 44, 45 | ours (paper: 5 runs) |
| Calibration | One temperature per seed and one for the ensemble, minimising validation NLL | ours |
| Hardware | One RTX 2080 Ti per seed (IIIT-H Ada cluster), ~18 min per epoch | — |

## Performance (Article-Bias-Prediction test set, 1,300 articles)

| Model | Best epoch / run | Val F1 | Test F1 | Test acc | Test MAE | Test recall L / C / R | Test ECE raw → calibrated (T) |
|---|---|---|---|---|---|---|---|
| seed 42 | 10 / 10 | 93.19 | 91.94 | 92.15 | 0.118 | 89.8 / 96.7 / 91.5 | 0.062 → 0.016 (3.57) |
| seed 43 | 10 / 10 | 92.86 | 92.00 | 92.31 | 0.112 | 89.3 / 96.7 / 92.2 | 0.062 → 0.024 (3.74) |
| seed 44 | 7 / 10 | 92.55 | 91.16 | 91.23 | 0.132 | 92.3 / 96.0 / 88.2 | 0.068 → 0.015 (3.47) |
| seed 45 | 10 / 10 | 92.86 | 92.33 | 92.46 | 0.116 | 90.8 / 96.3 / 91.7 | 0.059 → 0.028 (3.66) |
| **mean ± std** | | 92.86 ± 0.26 | 91.86 ± 0.49 | 92.04 ± 0.55 | 0.12 ± 0.01 | 90.55 ± 1.32 / 96.41 ± 0.32 / 90.86 ± 1.83 | 0.06 ± 0.00 |
| **ensemble of 4** | | 93.71 | 93.09 | 93.15 | 0.106 | 92.3 / 97.0 / 91.8 | 0.032 → 0.018 (3.08) |

Ensemble confusion matrix, test (rows true, columns predicted):

| | Left | Centre | Right |
|---|---|---|---|
| **Left** | 371 | 13 | 18 |
| **Centre** | 3 | 290 | 6 |
| **Right** | 31 | 18 | 550 |

Version 1 `politics_random.pt` (truncated, 1 seed): macro-F1 89.1, accuracy 89.3, MAE 0.16.

Per-epoch macro-F1 (val / test):

| Epoch | seed 42 | seed 43 | seed 44 | seed 45 |
|---|---|---|---|---|
| 1 | 90.21 / 87.95 | 88.53 / 85.98 | 90.04 / 87.44 | 89.94 / 87.56 |
| 2 | 90.70 / 91.03 | 91.94 / 90.26 | 91.67 / 92.10 | 90.23 / 89.60 |
| 3 | 92.42 / 92.87 | 90.88 / 89.10 | 92.04 / 93.68 | 92.21 / 92.10 |
| 4 | 90.93 / 91.35 | 92.10 / 91.98 | 92.33 / 93.47 | 92.43 / 91.45 |
| 5 | 91.61 / 92.40 | 92.15 / 91.28 | 92.34 / 92.19 | 92.13 / 93.09 |
| 6 | 92.93 / 91.81 | 92.77 / 92.87 | 92.53 / 92.19 | 92.03 / 91.13 |
| 7 | 91.99 / 91.38 | 92.67 / 92.30 | 92.55 / 91.16 ★ | 92.33 / 92.15 |
| 8 | 92.76 / 91.42 | 92.51 / 92.08 | 92.20 / 91.81 | 92.70 / 92.72 |
| 9 | 92.92 / 92.42 | 92.67 / 92.48 | 92.48 / 91.75 | 92.70 / 92.84 |
| 10 | 93.19 / 91.94 ★ | 92.86 / 92.00 ★ | 92.50 / 91.34 | 92.86 / 92.33 ★ |


★ = epoch selected by validation macro-F1.

What the per-epoch table shows:
- **Selection is meaningful now.** On the random split, validation and test move together (both 88–93), unlike
  the media split's chance-level validation in version 1. Validation is still noisy at the ±1-point level,
  so the best validation epoch is not always the best test epoch (e.g. seed 44 peaks on test at epoch 3).
- **Training had converged.** Validation sits at 92.0–93.2 from epoch 6 on. Three of four seeds set their
  best at epoch 10, the cap, by margins of 0.1–0.3 points, which is within noise.
- **Seed variance on test is small** (± 0.5 macro-F1). The **ensemble adds ~1.2 points** over the mean seed.
- **Calibration:** every seed is strongly over-confident (fitted temperatures 3.5–3.7). Temperature scaling
  cuts test ECE from ~0.06 to ~0.02; the ensemble needs T = 3.08.
  See `reliability.png`.
- Centre is the strongest class on this test set (recall 96–97%), but the weakest on the team's data (below).

## Team `stories_10k` (30,000 labelled articles, calibrated ensemble)

| | Macro-F1 | Accuracy | MAE | Recall L / C / R | ECE |
|---|---|---|---|---|---|
| Ensemble of 4 (calibrated) | 60.7 | 64.2 | 0.41 | 81.4 / 25.6 / 85.5 | 0.199 |
| Single seeds (macro-F1) | 61.7 / 60.7 / 61.1 / 58.6 | | | | |
| Version 1 `politics_random.pt` | 61.0 | 63.9 | 0.41 | ~83 / ~22 / ~84 | — |

Neutral summaries (10,000, unlabelled): predicted L / C / R = 26.5% / 20.6% / 52.9%.

Two things stand out. The sliding windows do not help here: stories_10k articles are short (median ~180
tokens, none over 512), so they fit in one window anyway. And the temperature fitted on the corpus's
validation set does **not** calibrate this data (ECE 0.20): the classifier is still over-confident
on text unlike its training distribution. Treat probabilities on new domains as ranks, not rates.

## Caveats

- **Pretraining overlap (not fixed here).** POLITICS was pretrained on BigNewsAlign, from 11 US outlets:
  Daily Kos, HuffPost, CNN, Washington Post, NYT, USA Today, AP, The Hill, Washington Times, Fox and
  Breitbart. The team's datasets (stories_10k, the rewrite sets) are drawn from BigNewsAlign, so the
  encoder has likely seen those original articles. Their scores, including the stories_10k numbers above,
  are likely inflated, and a rewrite can look "harder" partly because it is new text. Use the classifier
  for relative comparisons, ideally against a paraphrase control.
- **Random split = outlet recognition.** Train and test share outlets, so part of the accuracy is
  recognising an outlet's house style. That is fine for measuring whether a rewrite removes outlet-style
  cues, but it is not evidence of article-level ideology detection.
- **Labels are outlet-level.** Each article inherits its outlet's AllSides rating.
- **Centre is the weakest class on the team's data** (recall ~26% on stories_10k).

## License

POLITICS is CC BY-NC-SA 4.0; the fine-tuned weights are a derivative under the same terms.

## Links

| What | Link |
|---|---|
| POLITICS paper | https://arxiv.org/abs/2205.00619 |
| POLITICS code (pretraining only) | https://github.com/launchnlp/POLITICS |
| Pretrained checkpoint | https://huggingface.co/launch/POLITICS |
| Article-Bias-Prediction corpus + splits | https://github.com/ramybaly/Article-Bias-Prediction |
| Training code | `politics/train_v2.py`, `politics/politics_classifier.py`, `politics/aggregate_v2.py` (team repo, branch `harshit`) |
