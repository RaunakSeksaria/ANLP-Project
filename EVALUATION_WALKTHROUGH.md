# Evaluation walkthrough

This document explains the evaluation side of the project from scratch. It assumes you know basic
machine learning (classifiers, training, accuracy) but nothing about this project or the papers it
builds on. Every claim is stated in full where it is used.

Sources: `CONTEXT.md` (all results), `replication/REPLICATION.md` and `baly/REPLICATION.md` (full
logs), and the notebooks in `evaluation/` and `jev/`.

---

## Contents

1. What the project is, and why evaluation is the hard part
2. Background: how the data is labelled, two ways to test, metrics
3. The two datasets
4. Instrument 1: ChunkyBERT
5. Instrument 2: Baly et al. 2020
6. Instruments 3 and 4: Jev, a zero-shot model
7. Instrument 5: VAD
8. Instruments not finished
9. All instruments applied to the team's data
10. The common thread
11. How to use all this to evaluate the team's rewriters
12. Limitations
13. Repository map
14. Likely questions

---

## 1. What the project is, and why evaluation is the hard part

**The project.** News outlets cover the same event with a political slant: Fox News and the New York
Times describe the same bill differently. The team trains models that take a slanted article (or
several) and produce a neutral version. The current models are two rewriters based on LED, a
Transformer that turns one text into another (an "encoder-decoder"). They take one article in and
write a neutral rewrite out.

**The evaluation problem.** To claim "our rewrite is more neutral than the original", you need a way
to measure political slant. The obvious tool is a classifier that reads a text and outputs
Left / Centre / Right, applied to the original and the rewrite to see whether the rewrite moved
toward Centre.

That only works if the classifier is actually detecting *slant*. This work checks whether it is.
Doing so meant:

1. building the classifiers that published papers recommend;
2. checking that each reproduces its paper's results, so we know it is built correctly;
3. applying them to the team's data to see what they really measure.

**The short answer:** the published classifiers mostly detect *which outlet wrote the article*, not
its slant. That makes them unreliable on their own for judging the team's rewrites. Section 11
explains how to use them anyway.

---

## 2. Background

### 2.1 How the data is labelled

Every article in these datasets carries a label: Left, Centre or Right. In practice that label is
almost always just **the political rating of the outlet that published it**:
- in the main training dataset, 99.9% of articles have the same label as the rest of their outlet's
  articles (only 42 of 37,554 differ);
- in the team's dataset, 100% do. Every Fox article is Right and every Associated Press article is
  Centre.

**Why it matters:** a classifier can get the label right without understanding slant at all. Learning
to recognise the outlet's writing style is enough ("this reads like Breitbart → Right"). High
accuracy therefore doesn't prove the classifier detects slant.

### 2.2 Two ways to test a classifier

- **Random split.** Shuffle all articles and split them into training and test sets. The same outlets
  appear on both sides, so a classifier that has memorised outlet styles scores well.
- **Media split.** Put each outlet's articles entirely in training *or* entirely in test. The test
  outlets are then ones the classifier has never seen, so recognising style can't help and it has
  to generalise ideology.

**The gap between the two scores measures how much of a classifier's accuracy came from recognising
outlets.**

### 2.3 Metrics

- **Accuracy:** fraction labelled correctly. Chance is 33% with three balanced classes.
- **Macro-F1:** F1 computed per class, then averaged, so a model that ignores a class scores badly.
  We report it ×100 (e.g. 37.1).
- **MAE:** mean distance between predicted and true label on the scale Left = 0, Centre = 1,
  Right = 2. Calling a Left article Right costs 2; calling it Centre costs 1. Lower is better.
- **Recall per class:** of the true Centre articles, what fraction were predicted Centre. This is
  where most of the interesting failures show up.

**How precise the numbers are.** On the team's 30,000 articles, an accuracy near 40% is pinned down to
about ±0.6 points (95% interval), so a 1-point difference is real. On the main dataset's 1,300-article
media-split test set it is about ±2.7 points. That test set also contains only 13 outlets, so
differences of a few points there mean little.

---

## 3. The two datasets

### 3.1 Article-Bias-Prediction: used to *train* the classifiers

Released by Baly et al. (2020), collected from AllSides, a website that rates news outlets' political
leaning.
- 37,554 full news articles: Left 13,005, Centre 10,815, Right 13,734.
- Long: median 1,018 tokens, and 88.6% exceed 512 tokens, the most a standard BERT model can read.
- Comes with official random and media splits.
- The released version differs from the one in Baly's paper (37,554 vs 34,737 articles; different
  split sizes), so results are comparable in kind, not to the decimal. Its media split does keep
  outlets separate: we checked, and no outlet appears in both training and test.

### 3.2 `stories_10k`: the team's data, used to *evaluate*

10,000 news events ("stories"), each covered by four texts:

| Field | What it is | Label |
|---|---|---|
| `left` | an article on the event from a left-leaning outlet | Left |
| `center` | an article on the event from a centrist outlet | Centre |
| `right` | an article on the event from a right-leaning outlet | Right |
| `neutral` | a neutral summary of the event written by an LLM (Nemotron-3-Ultra) | none |

That gives **30,000 labelled articles** (10,000 per class) and **10,000 unlabelled neutral
summaries**. The articles come from 11 outlets, and the Centre class is lopsided:

| Side | Outlets (number of articles) |
|---|---|
| Left | New York Times 2,854 · HuffPost 2,827 · Daily Kos 2,257 · Washington Post 1,199 · CNN 863 |
| Centre | **The Hill 6,089** · USA Today 2,001 · Associated Press 1,910 |
| Right | Breitbart 4,644 · Washington Times 2,814 · Fox 2,542 |

Two properties matter later:
- **The articles are short:** median 135 words and never more than 250, against ~1,000 tokens for
  the training articles. This is by construction: the events come from BigNewsAlign, a corpus of
  news articles grouped by event, exported with articles capped at 250 words, and the 10,000
  *shortest* events were chosen (`data_generation/scripts/make_clusters.py`). There is far less
  text to go on.
- **Same outlets as the training data, different articles.** 10 of these 11 outlets also appear in
  the training dataset (all but HuffPost), but only 71 of the 30,000 articles (0.2%) do. The
  classifiers have seen these outlets, but not these articles.

---

## 4. Instrument 1: ChunkyBERT

### What it is

A 2026 paper (Loiya et al., *Scientific Reports*) presenting a classifier for long news articles.
Standard BERT reads at most 512 tokens, so ChunkyBERT:
1. cuts the article (up to 5,000 tokens) into 512-token pieces ("chunks");
2. runs BERT on each chunk;
3. passes BERT's output through two extra Transformer layers;
4. compresses each chunk to one vector with "attention pooling", a learned weighted average over
   tokens;
5. averages the chunk vectors and predicts Left / Centre / Right.

The paper reports 86% accuracy on the Article-Bias-Prediction dataset (random split).

### Does our copy work? Yes.

The authors published their code. We used it unchanged, apart from fixing a syntax error that stopped
it running, and trained it twice on Kaggle's free GPUs with the paper's settings.

| | Paper | Our run 1 | Our run 2 |
|---|---|---|---|
| Best validation accuracy | 86.22% | 86.58% | 86.87% |
| Accuracy after 1 epoch | 83.91% | 82.93% | 83.01% |

Note this is a **random split** (§2.2), so outlet memorisation is allowed.

### The paper's two architecture claims, and why they don't hold

The paper argues that each part of its design is necessary. It does so with two experiments: remove
parts one at a time (an "ablation study") and show accuracy falls, and compare against simpler older
models. The authors released no code for either, so we rebuilt them from the paper's description and
reran them, training each model for one epoch as the paper did.

**Claim 1: "The two extra Transformer layers are essential. Removing them drops accuracy from 84% to
41%."** This is the paper's main evidence that its design matters.

When we removed those two layers, accuracy did not drop at all:

| Model | Paper | Ours |
|---|---|---|
| Full ChunkyBERT | 84% | 81.0% |
| Without the two extra Transformer layers | **41%** | **82.3%** |

Why we think the paper's 41% comes from a different experiment:
- **The training time doesn't fit.** The paper reports this variant trained in 5 minutes against 27
  for the full model. The two extra layers are tiny next to BERT (about 110 million parameters), so
  removing them can't make training 5× faster. Ours took 24 minutes.
- **Removing BERT itself reproduces their numbers.** Without BERT (just learned word vectors, attention
  pooling and a classifier), we get 37% in about 2 minutes. That is close to the paper's 41% and
  similarly fast.

So the paper's "no Transformers" result most likely measured a model with BERT removed, while the
text describes removing only the two extra layers. Our other ablations agree that BERT does the work:

| What was changed | Paper | Ours |
|---|---|---|
| Nothing (full model) | 84 | 81.0 |
| Plain average instead of attention pooling | 82 | 81.4 |
| Simpler final classification layer | 82 | 82.4 |
| Take the max over chunk vectors instead of the average | 81 | 81.8 |
| No chunking: read only the first 512 tokens | 79 | 77.8 |
| BERT frozen (not trained) | 58 | 58.2 |

- These five match the paper to within about 1 point.
- Freezing BERT costs 23 points; every other change costs at most about 3.
- Several simplified versions score slightly *above* the full model. Those gaps (≤1.4 points) are
  within run-to-run noise, so the fair conclusion is that the components the paper adds on top of
  BERT make no measurable difference.

**Claim 2: "Attention is essential for simpler models too."** The paper compares ChunkyBERT with two
older model types, GRU and TCN (two standard ways of reading a text word by word), each with and
without attention. It reports that a GRU without attention scores 36–39%, barely above chance,
while with attention it scores 73–75%. That is a 34–38 point effect.

The paper doesn't specify these models fully, so we built the simplest version, trained from scratch:
- **With the learning rate the paper states** (very small, suited to fine-tuning BERT rather than
  training from scratch), no model learns anything: they all score 37–39%. That matches the paper's
  "without attention" numbers, which suggests those runs simply failed to train.
- **With a normal learning rate for training from scratch**, the GRU with attention matches the paper
  (77% and 76% vs 73% and 75%). Without attention it reaches 69–72%, not 36–39%. Attention is worth
  about 6 points, not 34.
- The TCN variants never matched the paper (12–18 points low). We stopped there rather than keep
  adjusting settings until a number happened to match.
- Caveat: this depends on our reading of a baseline the paper leaves underspecified.

**What to take from ChunkyBERT:** the model works as a classifier and our copy reproduces its accuracy.
Its claims about *why* it works don't hold up. In practice it is BERT applied to chunks.

---

## 5. Instrument 2: Baly et al. 2020

### What the paper claims

Baly et al. (EMNLP 2020) created the media split (§2.2) and the Article-Bias-Prediction dataset. They
make three claims:

1. **Classifiers mostly learn the outlet.** BERT does well on a random split and collapses on a media
   split.
2. **The outlet is easy to identify from text.** A BERT model can name the outlet from the article.
3. **Two training tricks reduce the problem**, improving accuracy on unseen outlets:
   - *Adversarial adaptation.* During training, add a second small classifier that tries to guess
     *which outlet* wrote each article from BERT's internal representation. Then train BERT to make
     that guesser *fail*. BERT should stop encoding outlet identity, forcing the main
     Left/Centre/Right classifier to rely on something else, hopefully the slant.
   - *Triplet pre-training.* Before the main training, teach BERT that two articles with the same
     ideology from *different* outlets should look similar, and two with different ideologies from
     the *same* outlet should look different.

The authors released only data, no model code. We rebuilt all the models from the paper's description
using its stated settings, and wrote down the settings it omits (e.g. batch size) as assumptions.

### What we found

**Claims 1 and 2 reproduce.** Results on Baly's test set:

| Model | Paper: F1 / accuracy | Ours: F1 / accuracy |
|---|---|---|
| Always predict the most common class | 19.6 / 41.7% | 21.0 / 46.1% |
| BERT, random split (outlets seen) | 80.2 / 79.8% | 82.2 / 81.9% |
| BERT, media split (outlets unseen) | 35.5 / 36.8% | 37.1 / 42.1% |
| BERT guessing the outlet (73 outlets) | 59.7 / 80.1% | 48.3 / 78.4% |

- **Unseen outlets.** The same BERT drops from 82 to 37 macro-F1 once test outlets are unseen. On
  those outlets its accuracy is even *below* always guessing the most common class, in our run and
  in the paper's.
- **Not zero ideology signal.** Its macro-F1 (37) is still well above the baseline's (21), so it
  isn't predicting one class blindly; it is just unreliable.
- **Outlet identification.** BERT names which of 73 outlets wrote an article 78% of the time. Outlet
  identity is easy to read off the text.

**Claim 3 (adversarial adaptation) does not reproduce.**
- The paper reports it raises F1 on unseen outlets from 35.5 to 43.9 (+8). We got 33.7, *below*
  plain BERT's 37.1.
- To rule out a broken implementation, we checked whether the outlet-guesser was actually defeated.
  With 54 training outlets, a guesser that knows nothing has a loss of about 3.99 (the loss of a
  uniform guess, ln 54). Our guesser's loss stayed at that level, so BERT did hide outlet identity.
- That check is indirect: only the combined loss was logged, and we subtracted the main classifier's
  typical loss.
- So the fix did what it was designed to do, yet accuracy on unseen outlets did not improve. Hiding
  the outlet didn't make BERT find the slant.
- Caveat: this was one training run, and the media split's validation set has only 12 outlets
  (accuracy on it stayed below chance throughout). Read this as "not reproduced", not "disproven".
- The triplet pre-training model is built and tested but not yet run.

**What to take from Baly:** a controlled confirmation that these classifiers mostly learn the outlet,
and that the published fix didn't help in our hands.

---

## 6. Instruments 3 and 4: Jev, a zero-shot model

### What Jev is

Jev is a hosted model from TypeSafe. You send it a text and a question with named answer options; it
returns a probability for each option. It is "zero-shot": it was never trained on our labels. That
makes it a useful contrast with the BERT classifiers, which learned from outlet-labelled data.

### Instrument 3: Jev as an ideology classifier

- **Choosing the question.** On 450 held-out articles we compared four phrasings, each with and
  without the headline. The best asked *"What kind of outlet most likely published this?"*, not
  *"What is this article's political leaning?"*. That fits §2.1: the labels describe the outlet.
- **Cost:** $0.95 for about 38,500 articles and summaries.

### Instrument 4: Jev for AllSides bias types

AllSides lists 16 kinds of media bias. Twelve can be judged from a single text, and each became a
yes/no question to Jev:
- spin and sensationalism;
- loaded words and subjective adjectives;
- opinion presented as fact, unsupported claims, unnamed sources;
- personal attacks ("mudslinging"), claiming to know someone's thoughts ("mind reading");
- unwarranted negativity, presenting only one side, and faulty reasoning.

Five more questions ask about direction: whose vocabulary is used, which side the selection of facts
favours, whose view comes first, which side is attacked, and whether experts or ordinary people are
deferred to.

**Why this matters:** these questions ask about *how the text is written*, not *who published it*,
which the ideology classifiers can't do. They are Jev's judgments and haven't been checked against
human annotators. Run on 750 stories × 4 texts = 3,000 texts, for $0.19.

---

## 7. Instrument 5: VAD

VAD scores a text's emotional tone using a published word list (Warriner et al., 2013) of 13,915
English words rated 1–9 on three scales:
- **valence:** unpleasant to pleasant;
- **arousal:** calm to excited;
- **dominance:** controlled to in control.

A text's score is the average over the listed words it contains. It was built separately from the
classifier work.

---

## 8. Instruments not finished

- **Publisher-leakage check.** The outlet-guessing model from §5 exists, but it hasn't yet been run
  on the team's data or on rewrites. The question it would answer: can you still tell which outlet
  an article came from after it was rewritten?
- **POLITICS (Liu et al., 2022).** A version of RoBERTa (a BERT variant) that the authors trained
  further on 2.3 million articles from the same 11 outlets as the team's data, specifically to
  understand ideology.
  - The trained model is public; redoing that training would cost 160–240 GPU-hours.
  - The agreed plan is to reproduce the paper's ideology results with the public model. Only two of
    its six test datasets are publicly downloadable.
  - The team's articles come from BigNewsAlign, the aligned corpus released *with* POLITICS, which
    POLITICS was trained on. Any POLITICS score on the team's data is therefore very likely
    contaminated.

---

## 9. All instruments applied to the team's data

### 9.1 Ideology classifiers on the 30,000 labelled articles

| Model | Trained on labels? | Accuracy | Macro-F1 | Recall L / C / R | % predicted Right |
|---|---|---|---|---|---|
| Jev (zero-shot)* | no | **47.8%** | **45.8** | 0.37 / **0.80** / 0.26 | 13% |
| ChunkyBERT | yes | 44.3% | 40.7 | 0.49 / 0.13 / 0.71 | 55% |
| Baly BERT, random split | yes | 42.0% | 37.9 | 0.39 / 0.11 / 0.76 | 60% |
| ChunkyBERT, simplified head | yes | 40.4% | 33.7 | 0.34 / 0.05 / 0.83 | 70% |
| Baly BERT + adversarial | yes | 37.0% | 35.5 | 0.27 / 0.25 / 0.59 | 49% |
| Baly BERT, media split | yes | 36.0% | 31.7 | 0.31 / 0.08 / 0.69 | 60% |

\* Jev is scored on 29,550 articles; the 450 used to choose its question are excluded. Chance is 33%.

- **Every BERT model trained on labels scores 36–44%.** ChunkyBERT and the random-split Baly model
  reach 82–87% on the training dataset, so the drop is roughly 40 points. Expected, given §5: the
  articles are new, short, and lean on outlet style less.
- **The zero-shot model beats all of them** (47.8%).
- **Their mistakes run in opposite directions.**
  - The BERT models almost never predict Centre and predict Right about half the time or more.
  - Jev predicts Centre for about two-thirds of articles.

Confusion matrix for ChunkyBERT (rows = true label, columns = prediction). Most Centre articles are
called Right, and many Left ones too:

| | → Left | → Centre | → Right |
|---|---|---|---|
| **Left** | 4,866 | 1,210 | 3,924 |
| **Centre** | 3,324 | 1,324 | 5,352 |
| **Right** | 1,783 | 1,118 | 7,099 |

### 9.2 Accuracy per outlet

This table shows *what* each model has learned:

| Outlet (true label) | ChunkyBERT | Simplified head | Baly random | Baly media | Baly adversarial | Jev |
|---|---|---|---|---|---|---|
| CNN (L) | 82% | 76% | 78% | 82% | 79% | 12% |
| New York Times (L) | 69% | 41% | 43% | 7% | 7% | 16% |
| Washington Post (L) | 40% | 25% | 26% | 29% | 23% | 15% |
| Daily Kos (L) | 37% | 27% | 41% | 33% | 27% | 75% |
| HuffPost (L) | 31% | 22% | 26% | 38% | 33% | 46% |
| The Hill (C) | 10% | 3% | 9% | 5% | 17% | 74% |
| USA Today (C) | 27% | 13% | 26% | 21% | 53% | 84% |
| Associated Press (C) | 9% | 2% | 5% | 2% | 19% | 98% |
| Washington Times (R) | 80% | 90% | 81% | 90% | 83% | 19% |
| Breitbart (R) | 69% | 82% | 82% | 65% | 56% | 36% |
| Fox (R) | 65% | 78% | 60% | 56% | 39% | 16% |

- **The BERT models and Jev succeed on almost opposite outlets.** The BERT models are right about the
  Washington Times, CNN and Breitbart, and almost never right about any centrist outlet. Jev is right
  about the centrist outlets and Daily Kos, and wrong about most right-leaning outlets.
- **Centre failure isn't one outlet.** All three centrist outlets are mostly missed by the BERT
  models. The Hill matters most: it is 61% of all Centre articles, and the BERT models get only
  3–17% of it right.
- **A clue that outlet familiarity matters.** On the New York Times, the Baly model trained on a
  random split (which included NYT) gets 43% right; the two media-split models, which never saw NYT,
  get 7%.

### 9.3 The 10,000 neutral summaries

These have no label, so we can only report what each model predicts:

| Model | Left / Centre / Right |
|---|---|
| ChunkyBERT | 38% / 4% / 58% |
| ChunkyBERT, simplified head | 10% / 0% / 90% |
| Baly BERT, random split | 7% / 5% / 88% |
| Baly BERT, media split | 22% / 3% / 75% |
| Baly BERT + adversarial | 25% / 21% / 54% |
| Jev | 5% / 95% / 1% |

The BERT models call the summaries mostly Right; Jev calls nearly all of them Centre. Each model is
repeating its general tendency from §9.1. This tells us about the models, not the summaries, and it is
the clearest example of why a single classifier can't certify that a text is neutral.

### 9.4 Bias types (Jev, 750 stories)

Each text gets a probability for each bias type; the "overall bias score" is their average.

- **Neutral summaries are the least biased text type, but only just.**
  - Overall: 0.266, against 0.276 for the centre articles.
  - Story by story, the summary scores below the left article 71% of the time, the right article 59%,
    and the centre article 52% (a coin flip).
- **Their clear improvement is balance.** "Presents only one side" scores 0.30 for summaries against
  0.48–0.55 for the source articles.
- **They are slightly worse than centre articles** on unnamed sources (+0.08) and personal attacks
  (+0.04). One possible reason: summarising drops names but keeps charged quotes.
- **Left articles score highest** on almost every bias type; right articles are close to centre ones.
- **Direction:** Jev judges the summaries' selection of facts as favouring the left more often than
  the right (average probability 0.40 vs 0.18).
- **Combined signal:** feeding all 17 answers into a small classifier trained to predict the label
  reaches 51% accuracy, the best on this data. That step uses the labels, so it isn't zero-shot.

### 9.5 VAD

Valence, arousal and dominance are practically identical across left, centre and right articles and
the neutral summaries, all within 0.05 on a 1–9 scale. Emotional word choice doesn't distinguish
partisan from neutral text here.

A related warning from the team's earlier analysis (`docs/eda-evaluation.md` on branch
`data-finding-out`): the "framing bias" score used in earlier neutral-summarisation papers correlates
0.88 with the amount of text scored, so a model can improve it just by writing less.

---

## 10. The common thread

Several independent results say the same thing: **these classifiers mostly recognise the outlet, not
the slant.**

1. BERT drops from 82 to 37 when tested on unseen outlets (§5).
2. BERT can name the outlet of an article 78% of the time (§5).
3. On the team's data the classifiers score 36–44%, and they succeed or fail by outlet: high on CNN
   and the Washington Times, near zero on the Associated Press and The Hill (§9).
4. Jev does best when asked "which outlet?" rather than "which leaning?" (§6).

Against that, the classifiers do pick up *some* real signal: on unseen outlets BERT's macro-F1 is 37,
against 21 for blind guessing.

**Why this matters for the team:** a rewrite that removes an outlet's writing style will change these
classifiers' outputs whether or not it changes the slant. A "more neutral" score from them alone
doesn't show the rewrite is more neutral.

---

## 11. How to use all this to evaluate the team's rewriters

Not yet run; this is the plan, and the reasoning behind each choice. The test set is the 2,000 left and
right articles from 1,000 events that the rewriters never saw during training.

- **Compare each article with its own rewrite.** Score both and report the change: did the
  probability of the original side fall, and Centre rise? Each classifier's built-in lean affects
  both texts, so comparing the pair partly cancels it.
- **Use classifiers with opposite leanings.** A move toward Centre seen by both the BERT models
  (which lean Right) and Jev (which leans Centre) is more convincing than either alone.
- **Use the bias-type questions for slant itself.** For example: is the rewrite less one-sided than
  its original, with fewer loaded words?
- **Check the outlet can't be recovered** from the rewrite, using the outlet-guessing model.
- **Include reference points.** The human-written centre article, the LLM summary, and the rewrites
  the models were trained to imitate, so "more neutral" is measured against something.
- **Check information is kept.** The easiest way to look neutral is to say less, so measure content
  preservation separately, e.g. whether questions about the original can still be answered from the
  rewrite.

---

## 12. Limitations

- **Labels describe the outlet, not the article.** An article counted as wrong may just be milder than
  its outlet.
- **One training run per model** in the replications. Small differences on the 13-outlet media-split
  test set are not meaningful.
- **Released data isn't the papers' data.** Comparisons are close, not exact.
- **The team's articles are short excerpts** (median 135 words), unlike the long articles the
  classifiers were trained on.
- **Centre is mostly The Hill** (61% of it), so Centre results largely reflect one outlet.
- **The GRU/TCN conclusions rest on our reading** of models the paper only partly describes.
- **Jev's bias-type answers aren't validated** against humans. Compare them between text types; don't
  read the absolute values as rates.
- **Jev's ideology run saw headlines; rewrites have none.** Comparisons with rewrites must re-score
  the originals without headlines.

---

## 13. Repository map

| Path | Contents |
|---|---|
| `eval_metrics.md` | The original list of required instruments |
| `chunkybert/` | ChunkyBERT, from the authors' code |
| `replication/` | Rebuilt ChunkyBERT comparison models; `REPLICATION.md` is the full log |
| `baly/` | Rebuilt Baly models; `REPLICATION.md` is the full log |
| `evaluation/` | Notebooks applying each model to the team's data; `scores/` holds per-article probabilities; `vad/` |
| `jev/` | Jev ideology (`jev_eval.ipynb`) and bias types (`bias_types.ipynb`) |
| `led_eval/`, `qa_eval/` | Rewriter evaluation. `qa_eval/`: QuestEval + DiverseSumm NLI coverage on LED λ0 / λ0.35, done (see `qa_eval/README.md`) |
| `CONTEXT.md` | All results in one place |
| `CLAUDE.md` | How to run things and where data and model weights live |
| `data/`, `models/`, `external/` | Not in git: datasets, trained weights, other projects' code (`CLAUDE.md` explains how to get them) |

Training ran on Kaggle (free 2×T4 GPUs); inference and testing on a 4 GB laptop GPU. Jev has cost
$1.14 in total.

---

## 14. Likely questions

**"44% accuracy — isn't that just a weak model?"** No. The same models score 82–87% when test outlets
were seen in training. On unseen outlets, BERT drops to 37 macro-F1 in both Baly's paper and our
reproduction. The low score on the team's data fits that pattern: the models rely on recognising
outlets, and that works much less well on short excerpts.

**"Why believe your result over the ChunkyBERT paper?"** We don't rely on our number alone. The
paper's own reported training time (5 minutes) is impossible for the model it describes. A model with
BERT removed reproduces both their accuracy (37% vs their 41%) and their speed. For the attention
claim, their "without attention" numbers match our runs that verifiably didn't train.

**"Why does a zero-shot model beat the trained ones?"** Not known for certain. A plausible
explanation: the trained models learned to spot outlet style in long articles, and short excerpts
carry little of it, while Jev never learned to rely on outlet style. Jev has its own strong lean
toward Centre, which is why we combine instruments.

**"So are the neutral summaries neutral?"** The classifiers can't tell us; they disagree in opposite
directions. The bias-type questions say the summaries are much more balanced than the source articles
and about as biased overall as the centrist articles, with slightly more unnamed sources and personal
attacks.

**"What's next?"**
- Evaluate the team's rewriters as in §11.
- Re-run the adversarial model several times to settle whether it really fails.
- Run the triplet pre-training model.
- Reproduce the POLITICS results.
