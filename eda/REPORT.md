# Corpus profiling

Every number here is reproducible with `uv run python eda/profile_datasets.py`
(CPU only, no network). Sections below pair the generated figures with a short
reading of what each corpus can actually carry.

An event-aware triplet objective needs three labels on the same example:

1. an **event** grouping (to pick the positive),
2. a **publisher** identity (to pick the negative, and to probe for leakage),
3. a **neutral summary** target (to train the decoder).

| corpus | scale | event grouping | publisher label | neutral target |
| --- | --- | --- | --- | --- |
| **NeuS / framing-bias-metric** | 3,095 stories × 3 articles | ✅ curated story ids | ✅ 250 outlets | ✅ AllSides roundup, ~57 w |
| AllSides / Article-Bias-Prediction | 37,554 articles | ❌ ~1.1 k mineable clusters | ✅ 491 outlets | ❌ |
| WCEP-10 | 10,200 clusters × ~9 docs | ✅ native | ❌ ~40 % guessable in-text | ⚠️ 25-word blurb |
| WNC | 181 k sentence pairs | ❌ | ❌ | ⚠️ sentence-level Wikipedia |

Only NeuS carries all three at once, and it is by far the smallest. Everything
else buys scale by giving up an axis.

---

## NeuS / framing-bias-metric (Lee et al. 2022)

```
stories 3,095 (unique ids 3,095)   publishers 250   dates 2016-05-13 → 2021-04-21
stance_specific  center 3075 · lean left 2214 · lean right 2135 · right 960 · left 881 · mixed 12 · not rated 8
article body words   median 85   p90 90   max 95
roundup words        median 57   p90 150  max 301
3-article concat     median 241  p90 270  max 275
released splits      train 2,452 · val 307 · test 307   (exactly 3 articles each)
```

**This is the corpus the interim plan assumed existed.** `raw_crawled/` gives a
story id, a publisher (`newsSource`), a stance (`stance`, plus a five-way
`stance_specific`), and a human-written neutral `roundup`, all on one record.

Three findings that shape everything downstream.

**The "articles" are excerpts.** Body length is capped at 95 words — AllSides
publishes a snippet per outlet, not the article. So the whole three-source input
is ~241 words (~310 BPE), which fits BART's 1,024 window with room to spare.
Good news for compute; bad news for any claim about *event information
retention*, because there is very little event information present to retain.
DiverseSumm-style QA evaluation over an 85-word excerpt is not really the same
evaluation the DiverseSumm paper describes.

**Triplet mining is feasible.** 153 of 250 publishers appear in ≥2 distinct
stories, which makes 9,188 of 9,285 article slots usable as a same-publisher,
different-event negative. The objective's sampling requirement is satisfiable
without any extra data.

**The released split leaks publishers.** `data_preprocessing.ipynb` cell 26 uses
`train_test_split(..., test_size=0.2, random_state=42)` — a random split over
*stories*. Joining the released `.source` files back to the crawl metadata
(100 % coverage, see below) shows **92 of 99 test publishers also occur in
train**. A publisher-recoverability probe scored on this split is scored almost
entirely on outlets it has already seen. That is the easy setting, and it is the
opposite of what Baly's media-based split was built for. Any leakage number
reported on the stock split needs that caveat, or a second publisher-disjoint
split alongside it.

### Joining the splits back to publisher metadata

The released train/val/test files contain only `TITLE=> … ARTICLE=> …` text with
the left/centre/right order shuffled — the labels are stripped. They can be
recovered by normalising the headline (casefold, strip non-alphanumerics) and
looking it up in the crawl.

The filtered crawl alone covers ~87 %. The union of all three crawl files —
`all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl` (3,095),
`all_crawled_combined.jsonl` (5,602) and
`headline_11_2021-10-19_crawled_all.jsonl` (506) — reaches **100 %**:

```
train  7,356/7,356 articles joined   2,452 stories   194 publishers
val      921/921                       308 stories   103 publishers
test     921/921                       307 stories    99 publishers
```

(`val` resolves to 308 stories for 307 examples: one headline collides across
two stories. Negligible, but worth knowing before anyone asserts a 1:1 join.)

Roughly 13 % of the released examples come from the headline crawl rather than
the filtered file, which is why the obvious single-file join silently drops
them. `neus_title_index()` in `eda/profile_datasets.py` does this correctly.

---

## AllSides / Article-Bias-Prediction (Baly et al. 2020)

```
articles 37,554   sources 491   topics 108
bias  left 13,005 · right 13,734 · center 10,815
sources carrying >1 bias label   7/491
dates  iso 33,110 · missing 4,407 · other-format 37
content words  median 916  p90 1,827  max 9,703
mined event clusters 1,160 → multi-source 1,124 → all three leanings 64
articles inside multi-source clusters  2,483 (6.6 % of corpus)
random split  train 27,978 · valid 6,996 · test 1,300
media  split  train 26,590 · valid 2,356 · test 1,300
```

**Bias is a publisher attribute, not an article annotation.** Only 7 of 491
sources ever carry more than one label; the rest inherit a single outlet-level
leaning across every article. Two consequences. First, "predict the article's
ideology" and "predict the publisher" are close to the same task on this data,
so reporting publisher-leakage and ideology-leakage as independent evaluation
axes would be double-counting. Second, the media-based split is the only
non-trivial generalisation test available here, and it is the one thing AllSides
offers that NeuS does not.

**No event ids.** Clustering by (topic, date) plus ≥50 % title-token overlap
recovers 1,124 multi-source clusters — and they are genuinely event-like
(2018-06-18 immigration groups The Hill / Daily Kos / Reuters on family
separation). But that is 6.6 % of the corpus, mostly pairs, with only 64
clusters spanning all three leanings. Enough to mine extra triplet negatives;
nowhere near enough to train a generator, and there is no summary target anyway.

**Length is a real constraint here, unlike in NeuS.** Median 916 words (~1,190
BPE) means most single articles already exceed BART's window, and framing
concentrates in the lead — so truncation is not a neutral operation.

Two data-quality traps: 4,407 articles have no date and 37 use `M/D/YY`, so any
date-keyed clustering silently drops ~12 % of the corpus; and a handful of
records carry junk years (`0001`, `2050`).

---

## WCEP-10 (Ghalandari et al. 2020, PRIMERA preprocessing)

```
train 8,158 clusters · val 1,020 · test 1,022     ~9 documents per cluster
doc words      median ~337  p90 ~820
summary words  median 25–28
docs naming a publisher in body text   38.0 % / 40.3 % / 42.1 %
in-text cues   AP 11,646 · Reuters 11,432 · AFP 5,373 · CNN 3,891 · boilerplate 3,542 · BBC 2,682
```

By far the best event structure and the only corpus at real scale — 10,200
genuine multi-document clusters with ~9 sources each, against NeuS's 3,095 × 3.
If the project were only about event-aware representation learning, this would
be the obvious choice.

Two things stop it being sufficient on its own.

**The targets were not written to be neutral.** A 25-word Wikipedia Current
Events blurb is a factual one-liner. Nothing in it was authored to reconcile
competing framings, so training against it teaches compression, not
neutralisation — and the framing-bias metric has almost no surface to measure on
a 25-word reference.

**Publisher labels were dropped.** PRIMERA's preprocessing kept only `document`
and `summary`; the raw WCEP release retains source URLs. They are partly
recoverable — 38–42 % of documents name their outlet in the body — but that cuts
both ways. Those same strings are a confound: a leakage probe would score high
off literal tokens like "(AP)" or the Guardian's "This article is more than 4
months old" rather than off learned style. Any leakage figure computed on
un-scrubbed WCEP text is measuring string matching.

Best role: **scale pretraining** for the event-aware objective (extension 1's
cross-source prediction needs unlabelled same-event pairs, which is exactly what
this is), with the supervised neutralisation done on NeuS.

---

## WNC / neutralising-bias (Pryzant et al. 2020)

```
biased.full        181,496 rows      neutral   385,527 rows
biased.word.train   53,803 · dev 700 · test 1,000
real_world_samples  ibc_left/right, news_left/right, news_sensationalist_dramatic, speeches — 200 each
format  TSV: id / src_tok / tgt_tok / src_raw / tgt_raw / src_POS / tgt_parse
```

Wikipedia NPOV revision pairs: lowercased, WordPiece-split (`##`),
sentence-level, single-source, encyclopedic. It shares no structure with the
multi-source task — no events, no publishers, no documents.

Useful for exactly two things: an auxiliary sentence-level neutralisation signal
(pretrain or multi-task the decoder), and as a source of bias-word lexicons. The
1,200 `real_world_samples` are eval-only probes on out-of-domain text.
