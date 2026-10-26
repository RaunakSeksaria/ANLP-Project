# POLITICS ideology classifier: issues found and requested fixes

*For Harshit. These are notes on `politics_media.pt` (Kaggle: `harshitlalwani/politics-ideology-classifier`),
checked against the POLITICS paper (Liu et al. 2022, `papers/politics.pdf`) and against how we plan to
use the classifier.*

## How we plan to use it

The classifier is an **evaluation metric**, not something we train on. We score each original
article and its neutralised rewrite, and expect the rewrite to be **harder to classify**: lower
accuracy at recovering the source side, flatter probabilities, and more mass on Centre. That gives
us a number for (a) the quality of our LLM-generated targets and (b) the trained LED model's
outputs.

So we don't need the best possible absolute accuracy. We need the classifier to be **stable and
uncontaminated**, so that a difference between original and rewrite (or between two prompt
variants) means something.

## What we checked: the fine-tuning against the paper

The head and optimiser match the paper (Table A7): `<s>` token → 2-layer FFNN (768 hidden,
dropout 0.1), AdamW, lr 2e-5, weight decay 0.001, betas (0.9, 0.999), 6% warm-up with linear
decay. Tanh is your choice; the paper doesn't specify the activation, which is fine. Loading with
`weights_only=True` works, and all weights load as expected.

Deviations that matter:

| | Paper (Table A7, §5.4) | `politics_media.pt` |
|---|---|---|
| Epochs / patience | 10 / 4 | 5 / 2 |
| Inputs over 512 tokens | sliding window, size 512, overlap 64 | truncated at 512 |
| Runs | average of 5 | 1 seed |
| Checkpoint selection | best validation F1 | best validation F1, but that best was **32.81** (see 1) |

## Issues, in priority order

### 1. Checkpoint selection is effectively random (media model)
The media checkpoint was chosen at epoch 3 with **validation macro-F1 32.81**, which is chance
level for 3 classes. Test macro-F1 is 53.1. Validation and test disagree because Baly's media
split uses **different outlets** in train, validation and test. Validation never rose above chance
in any epoch, so "best epoch by validation F1" didn't select anything meaningful. The test number
also rests on one 1,300-article set and one seed.

The random-split model doesn't have this problem (validation 90.85 at epoch 5).

**Please:**
- train for up to **10 epochs with patience 4**, as in the paper;
- select on a validation set that behaves like the test set: for example, hold out a slice of the
  *training* outlets' articles as validation, or use the random-split validation for selection
  and report media-split test;
- report **validation and test F1 for every epoch**, so we can see whether the choice matters;
- upload the **per-epoch checkpoints** (or at least first, best and last).

### 2. Long articles are truncated
Our new long dataset (`reframing_{left,right}_long` on HF, inputs up to 1,024 LED tokens) has
about **half of its articles over 512 tokens** (median about 530–580). Truncation means those are
judged on their opening only. The paper uses a **512-token sliding window with 64 overlap**.

**Please:** add sliding-window inference (average the logits over windows), and ideally train with
the same windowing, so training and inference match.

### 3. Single seed
The paper averages 5 runs. Your README already reports that a rerun moved the predicted-Centre
share on neutral summaries by **about 11 points**. That's bigger than the effects we want to
measure between prompt variants.

**Please:** train **3–5 seeds** and release them, or at least report mean ± std per metric. We
can then average predictions across seeds.

### 4. Leakage from POLITICS pretraining (can't be fixed by fine-tuning)
POLITICS is RoBERTa further pretrained on **BigNewsAlign**, with an **ideology triplet loss on the
outlets' left/right labels** (paper §3.2, §4.1), across exactly our 11 outlets (Table A4). **Every
one of our datasets is drawn from BigNewsAlign.** So the encoder has effectively seen our original
articles and was trained to separate them by side. Original articles therefore look easier to
classify than unseen text would, and any rewrite looks "harder" partly just because it's new text.

There are two workarounds:
- **Mask outlet clues in the fine-tuning (and scoring) text:** outlet names, "(CNN)"-style
  datelines, "told Fox News", bylines and the boilerplate first/last sentences, as POLITICS's own
  preprocessing did (Appendix A, "Removing Media Leaking Phrases"). Without this the model can
  read the outlet off the page. In our 6-article smoke test a HuffPost article scored
  **Right 0.99** and a Fox article **Centre 0.96**.
- **Train the identical recipe on plain `roberta-base`** (skipping POLITICS) as an uncontaminated
  comparison. If both classifiers tell the same story about our rewrites, the leakage isn't
  driving the result.

### 5. Missing file
The README lists `politics_random.pt`, but only `politics_media.pt` is on Kaggle. Please upload
it. It's useful as an **outlet-style detector** (89 F1 because it can recognise house style), and
comparing its drop with the media model's separates "style changed" from "framing changed".

### 6. Smaller things (nice to have)
- **Per-class results** on the media test set (confusion matrix): Centre recall is the weakest
  class (about 0.2–0.5), and our metric leans on it.
- **Calibration:** outputs are very confident (many 0.99s). A reliability plot or temperature
  scaling on validation would make "probability shifts toward Centre" more meaningful.
- **The 63.2 F1 on `stories_10k`** comes from BigNewsAlign too, so it's inflated by point 4. It's
  worth saying so in the README.

## What would help most, in order
1. A media model selected properly (point 1), with all per-epoch checkpoints.
2. Sliding-window inference (point 2).
3. A `roberta-base` version with the same recipe and outlet masking (point 4).
4. 3–5 seeds (point 3) and `politics_random.pt` (point 5).

Please keep the same release format (`state_dict` for the same `PoliticsClassifier` class, with
labels 0 Left / 1 Centre / 2 Right), so our scoring code doesn't change.

## What we're doing on our side regardless
- Masking outlet names in everything we score.
- Adding a **paraphrase control**: the same LLM rewords articles while explicitly keeping the
  framing. The drop from original → paraphrase is the "new wording" effect. The drop from
  original → neutral rewrite is new wording plus neutralisation, and the difference between the two
  is what we attribute to neutralisation.
- Scoring the same events' **centre-outlet articles** (AP, The Hill, USA Today) as a "what neutral
  looks like" reference.
- Using the classifier for **relative comparisons only** (original vs rewrite, variant vs variant),
  with confidence intervals over hundreds of articles.
