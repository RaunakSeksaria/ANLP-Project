# Reading the EDA: what the interim plan can and cannot become

Companion to `eda/REPORT.md` (corpus profiling) and `eda/metric_gold_baseline.md`
(the framing-bias metric). This is the judgement layer: what the numbers mean for
the proposal.

---

## 1. The headline

Three things came out of the profiling that the interim proposal could not have
known, in descending order of how much they change the plan.

**The evaluation metric this literature reports is confounded by output length.**
The NeuS framing-bias metric correlates 0.88 with the number of calibrated
tokens, and per-token affect is flat at ~0.178 across left, centre, right,
concatenations, and even a different corpus from a different decade. It ranks
partisan text above centrist text — but only because partisan text diverges more
in vocabulary from the reference, not because that vocabulary is more
affect-laden. **A model lowers its score by generating less.** That is precisely
the "neutrality by deletion" failure mode the interim proposal set out to avoid,
and the headline metric is blind to it.

**The corpus that has everything is very small, and its articles are excerpts.**
`framing-bias-metric-main` supplies event id, publisher and neutral summary
together — the only one that does. But it is 3,095 stories, the released split
is 2,452/307/307, and AllSides publishes an ~85-word excerpt per outlet rather
than the article (hard cap 95 words, measured). The full three-source input is
~241 words. Compute is therefore cheap; claims about *event information
retention* are correspondingly weak, because there is very little event
information present to retain.

**The released split leaks publishers.** `data_preprocessing.ipynb` cell 26 uses
`train_test_split(random_state=42)` over stories, and 92 of 99 test publishers
also appear in train. A publisher-recoverability probe evaluated on the stock
split is scored almost entirely on outlets it has already seen — the easy
setting, and the exact thing Baly's media-based split exists to prevent.

---

## 2. On WCEP-10 being the best corpus

Your instinct is right on the axes it is right on, and the profiling backs it up.
WCEP-10 beats NeuS on **everything structural**:

| | WCEP-10 | NeuS |
| --- | --- | --- |
| clusters | 10,200 | 3,095 |
| sources per event | ~9 | exactly 3 |
| document length | ~337 words (real article text) | ~85 words (excerpt) |
| event grouping | native | curated |

If the project were only "event-aware representation learning", WCEP-10 would be
the obvious and correct choice, and NeuS would be a toy beside it.

It fails on the two axes this *particular* proposal is built from.

**No publisher labels.** PRIMERA's preprocessing kept only `document` and
`summary`; the raw WCEP release retains source URLs, these files do not. They are
partly recoverable — 38–42 % of documents name their outlet in the body text —
but that same fact poisons the measurement: a leakage probe would score high off
literal strings like `(AP)` or the Guardian's `This article is more than 4 months
old` rather than off learned style. Any leakage number from unscrubbed WCEP text
is measuring string matching. The triplet objective's whole point is that anchor
and negative share a publisher; without reliable publisher identity, that
constraint cannot be imposed.

**The targets were not written to be neutral.** A 25-word Wikipedia Current
Events blurb is a factual one-liner. Nothing in it was authored to reconcile
competing framings. Training against it teaches compression, not neutralisation —
and the framing-bias metric has essentially no surface to measure on a 25-word
reference. Running the metric on WCEP confirms this: documents score 17.6
positive arousal against the NeuS articles' 3.2, entirely because they are 5.6×
longer after calibration. Per token, 0.177 versus 0.179. The metric is measuring
the length gap between a 337-word article and a 25-word blurb and nothing else.

**So: complementary, not competing.** The useful split of labour:

- **WCEP-10 → representation pretraining at scale.** Extension 1 (predict one
  report's latent from another's, same event) needs *only* same-event pairs. No
  publisher labels, no neutral targets. WCEP-10 supplies 10,200 clusters of
  exactly that, natively. It is the right corpus for that extension and there is
  no substitute in the collection.
- **NeuS → supervised neutralisation and all evaluation.** It is the only place
  the publisher-adversarial negative, the neutral target, and the metric all
  exist together, and it is what the papers you are citing report on.

There is also a weaker event-contrastive objective — positive = same event,
negative = *any* different event — that needs no publisher labels at all and runs
on WCEP-10 at full scale. That is a legitimate stage-one pretraining objective;
the publisher-adversarial refinement then happens on NeuS. Worth considering as
the two-stage design.

---

## 3. The interim plan, item by item

**Core: event-aware triplet + neutral generation — feasible, keep it.**
153 of 250 publishers appear in ≥2 distinct stories, making 9,188 of 9,285
article slots usable as a same-publisher, different-event negative. The sampling
requirement is satisfiable with no additional data. Publisher labels for the
released splits are recoverable at 100 % coverage by normalising headlines
against the union of all three crawl files (`neus_title_index()` in
`eda/profile_datasets.py`) — note that the obvious single-file join silently
misses ~13 %, which come from `headline_11_2021-10-19_crawled_all.jsonl`.

**Backbone: BART — fine here, unlike elsewhere.** The 3-article input is ~241
words (~310 BPE), comfortably inside 1,024. Note this is only true of NeuS: an
AllSides article alone is a median 916 words, and a WCEP-10 cluster is ~4,400
tokens, which would force LED/PRIMERA.

**Evaluation axis 1 — publisher leakage: keep, but fix the split.** As shipped it
is scored on seen publishers. Build a media-disjoint split (hold out whole
outlets, as Baly does) and report both. This is cheap — seconds of CPU — and it
converts a weak number into a real generalisation claim.

**Evaluation axis 2 — event retention via DiverseSumm QA: weaken or replace.**
The evaluation assumes articles rich enough to draw distinct questions from.
Here the input is three ~85-word excerpts summarised into ~57 words. There is
little room for a question answerable from one source and not another. Either
scale the claim down honestly, or replace it with a cheaper content-preservation
measure and say why.

**Extension 1 — cross-source pretraining: keep, scope it to WCEP-10.** Good fit,
but it is the one item needing real compute (~15–20 GPU-hours on a 40 GB card;
see `docs/ada-compute-requirements.md`). Cut first if allocation is tight.

**Extension 2 — FactorCL shared/unique factorisation: drop.** Two extensions plus
a core contribution plus a metric critique is more than a course project can
carry, and this is the one with the least support from the data on hand.

**Beware `romil-2.pdf`** (Lei & Huang, event relation graphs for framing-bias
mitigation). It works the same NeuS setup toward the same goal and partly
occupies the space. Read it before committing, and be explicit about the
delta — "representation-level constraint" versus "explicit event graph" is a
defensible distinction, but it has to be stated.

---

## 4. The metric finding is the cheapest real contribution available

It costs no GPU, it is already done, and it is a genuine problem with a NAACL
2022 metric that this literature still reports. Three levels of ambition:

1. **Report the per-token rate alongside the raw score.** Minimum viable
   honesty; makes the confound visible rather than load-bearing.
2. **Length-match, or regress the score on length and report the residual.**
3. **Make it a stated contribution:** *"the standard framing-bias metric is
   confounded by generation length; here is a normalised variant and what
   changes under it."*

One substantive lead for a replacement: MPQA strong-subjectivity density is the
only lexical signal measured that is *not* flat under length — 0.0633 / 0.0557 /
0.0631 hits per calibrated token for left / centre / right, a ~13 % relative gap,
against VAD arousal's 0.2 %. If a lexical neutrality measure is wanted, that is
the better candidate, and swapping it in is a one-line change to
`eda/framing_bias.py`.

Also note the baseline table: **copy-the-centre-article already scores below both
partisan articles** on the raw metric, at zero training cost. Any proposed model
must beat that, and it should appear in the results table as a baseline.

---

## 5. Recommended shape

**Primary: NeuS as the supervised corpus, triplet loss on the encoder, evaluated
on a media-disjoint split, with the metric critique as a stated secondary
contribution.** This is fully supported by the data on hand, needs ~20–25
GPU-hours, and every claim in it is one the corpus can actually license.

Concretely, in order:

1. Build the publisher-labelled, media-disjoint NeuS splits (CPU, minutes).
2. Reproduce the NeuS baseline on the modern stack (~1.5 GPU-hours), copying the
   `extra_task` / `task_loss_ratio` design at
   `src/transformers/models/bart/modeling_bart.py:1377-1533` and
   `src/finetune.py:203-224` — that hook is exactly where the triplet loss
   attaches.
3. Add the triplet objective, sweep weight and margin (~10 GPU-hours).
4. Probe publisher recoverability on frozen representations, both splits.
5. Report raw *and* length-normalised framing bias, against the copy-centre and
   concat baselines.

**If time allows:** WCEP-10 event-contrastive pretraining as stage one, then
fine-tune on NeuS — this is where the user's WCEP-10 instinct pays off, and it
directly tests extension 1's label-efficiency question.

**Fallback if GPU time does not materialise:** the metric critique plus a
representation-only study on AllSides — its media-based split is already built,
its publisher labels are real, and probes on cached vectors run on a laptop. No
generation, smaller claims, but a complete and honest paper.

---

## 6. Open risks

- **2,452 training examples is small** for fine-tuning BART-large. Expect high
  variance across seeds; report mean and spread over ≥3 seeds, not a single run.
  This is a reviewer's first question.
- **AllSides bias labels are publisher-level** — only 7 of 491 sources ever carry
  more than one label. "Predict ideology" and "predict publisher" are nearly the
  same task there, so do not report them as independent evaluation axes.
- **The centre "publisher" is sometimes an author.** Some NeuS centre slots have
  `newsSource` values like `Henry A. Brechter` — AllSides' own staff blog rather
  than an outlet. Filter or fold these before treating publisher as a clean
  categorical, or the leakage probe will have a degenerate class.
- **Date fields in AllSides are dirty**: 4,407 missing, 37 in `M/D/YY`, plus junk
  years (`0001`, `2050`). Any date-keyed clustering silently drops ~12 %.
- **NeuS's own metric code does not run** (`eda/metric_gold_baseline.md` lists the
  six defects). Numbers reported as "the NeuS metric" from now on are from our
  reimplementation; say so, and cite the paper's published table for comparison.
