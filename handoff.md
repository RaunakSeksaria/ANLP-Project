
## Chat: evaluate the new LED neutralisers (branch `harshit`), handed off 2026-10-01

### Task (user-approved; stopped mid-way)
Rerun every evaluation this branch has done on the two new single-article neutralisers in
`RaunakSeksaria/anlp-neutral-led` (subfolders `side-neutral-full/lambda0` and `lambda035`; LED-base;
one partisan article in, the same article rewritten neutrally out). "Where applicable", also run the
scorers on Raunak's new training data. The user asked for:
- **one unified progress bar** to track everything, and
- **a live markdown file** that gets each evaluation's results as soon as it finishes.

**Approved by the user:**
- **Jev:** about $0.50 more spend (ledger `jev/results/spend.json` stands at $1.1412). Keep the
  total for this effort at or under about $0.55.
- **Kaggle:** pushing kernels, about 4–6 GPU h of the 30 h weekly quota.
- **Nemotron reference row:** add the training-target rewrites as a comparison row.

### Established facts (checked, don't redo)
- **Input format** (from `origin/manas:src/data/triplet_dataset.py` `format_article`):
  `f"{title.strip()} {tok.sep_token} {' '.join(body.split())}"`, truncated at 512 tokens.
  - Raunak's data card says the `source` field uses `TITLE=> … . ARTICLE=> …`. That is *not* what the
    LED was trained on.
- **Decoding** (from the checkpoint's `metrics/training_config.json`): greedy, `num_beams=1`,
  `max_new_tokens=512`, global attention on token 0. Card test numbers: F1 0.810 for both λ.
- **The model's test set is held out.** Its 1,775 articles are the left (868) + right (907) test
  splits of `left/right_neutral_dedup.jsonl` on `origin/data`.
  - All 2,000 left/right articles of the 1,000 test stories in `data/stories_10k.jsonl` are absent
    from model train/val.
  - The 225 test articles that don't appear by id are text-duplicates; matching on text recovers
    all 2,000 Nemotron rewrites.
  - Train-split stories are not usable: about 8,300 of those articles were training inputs.
- **`data/own_test.jsonl` is the same set:** the same 2,000 test articles, with the human centre
  article as `target_text`. `qa_eval/qa_eval.py` uses it, so one held-out set feeds every
  evaluation.
- **`qa_eval/qa_eval.py`** (untracked) was written in an earlier chat for the *old*
  `Manas206/led-neutral-*` models. It has never been run beyond a local smoke test of the gen/nli
  stages, so there are no old QA numbers.
  - To reuse it: swap `MODELS` to the new checkpoints; change `fmt` to `title </s> body`, `MAX_IN` to
    512 and `GEN` to greedy.
  - It needs `diverse_summ.json` (at `external/DiverseSumm/data/`) and `own_test.jsonl`; on Kaggle
    both come in as an uploaded dataset.
  - QuestEval needs `pip install questeval --no-deps` plus the `datasets.load_metric = evaluate.load`
    shim already in the script. That path is untested.
  - Add `nemotron` as a QuestEval system alongside `reference`.
- **How the classifiers read text:** every classifier notebook scores `body` only, which is
  like-for-like with rewrites.
  - Jev ideology (`outlet_style`) and bias types originally scored originals *with title*. For
    paired comparisons, re-score the originals body-only.
  - Jev costs about 560 tokens per ideology request and about 1,500 per bias-type request (18
    questions), at $0.042/M.
- **Weights** (local sizes; none of them needs to be on the cloud box if GPU work stays on Kaggle):
  - LED λ0 and λ0.35: 618 MB each (`model.safetensors`); public HF repo
    `RaunakSeksaria/anlp-neutral-led`, loaded with `subfolder="side-neutral-full/lambda0"` etc.
  - ChunkyBERT `models/chunkybert_5ep.pt`, 449 MB: Kaggle dataset
    `harshitlalwani/chunkybert-political-bias`, or the `chunkybert-full-train` kernel output.
  - No-Classifier `models/no_classifier_1ep.pt`, 449 MB: output of the `chunkybert-best-variant`
    kernel (`best_variant_no_classifier.pt`).
  - Baly ×3 (`models/baly/`, 1.3 GB): **not used in this effort.**

### Done
- `led_eval/build_texts.py` → `data/led_eval/texts.jsonl`: 44,874 rows, 40,781 distinct texts.
  Each row has `id, panel, story_id, split, side, kind, publisher, h (sha1[:16] of text), text`.
  - `test` panel: 1,000 test stories × {original L/R, nemotron L/R, centre, llm_summary}, plus
    `led_lambda0` and `led_lambda035` rows once `led_eval/out/gen_<model>.jsonl` exists (rows
    `{id: "<story_id>:<side>", output}`). Rerun the script after generating.
  - `data` panel: all 19,437 records (`original` + `nemotron`) from Raunak's release, all splits.
- Copied `origin/data:{left,right}_neutral_dedup.jsonl` to `data/led_eval/`. **Stale, see below.**
- `qa_eval/qa_eval.py` updated for the new checkpoints (not yet run):
  - `REPO` + `MODELS` subfolders, overridable with env `LED_REPO=<local dir>`;
  - `fmt` = `title </s> body`, `MAX_IN` 512, greedy `GEN`;
  - `plain()` (no `</s>`) for the texts the NLI/QuestEval scorers read;
  - a `nemotron` QuestEval system, read from `texts.jsonl` (`find("texts.jsonl")`);
  - `--reuse` copies only `gen_*.jsonl`.

  A local `--stage gen --limit 2` smoke test was started but never finished (the machine
  restarted), so generation is still untested.
- `led_eval/generate.py` (standalone generator, same settings) exists, but this chat didn't write it.
  Read it before relying on it.

### Changes after the first handoff (read before continuing)
- **User decision: no Baly.** Don't use the Baly classifiers (`models/baly`) or Baly data. The
  classifier stage is ChunkyBERT + No-Classifier only.
  - Both of those were *trained* on Baly's Article-Bias-Prediction corpus. I told the user I'd keep
    them for inference unless they say otherwise; there is no answer yet.
- **Raunak pushed after `2b00cbe`.** Remote `data` head is `15a0876` (2026-10-01):
  - `2b70898` tops `left_neutral_dedup.jsonl` up to 10,000 articles (+563, `single_left_extra2.jsonl`).
  - `15a0876` adds `dataset/side_neutral_full/prepared/{train,validation,test}/{baseline,triplets}.jsonl`
    (12,484 / 1,778 / 1,775). This is **the exact split the λ models trained on**.

  To do: `git fetch origin data` (it hangs on the local machine; it should work in the cloud).
  Then:
  1. re-copy both dedup files to `data/led_eval/`;
  2. re-check that the 2,000 `stories_10k` test articles are absent from `prepared/train` and
     `prepared/validation`;
  3. rerun `led_eval/build_texts.py`.
  - `data/neutral_led/` (local only) is an abandoned partial download of that newer release
    (`right_*.jsonl.part`). Ignore or delete it.
- **Kaggle plan** (the user wants as much as possible on Kaggle; keep this machine light):
  1. Upload a private dataset `led-eval-data` with:
     - `texts.jsonl`, `own_test.jsonl` and `diverse_summ.json`;
     - the code: `qa_eval.py`, a classify script, `chunkybert/model.py`, `replication/variants.py`.
  2. **Kernel B1:** `qa_eval.py --stage gen`, one λ per T4.
  3. **Kernel A** (can run in parallel with B1): ChunkyBERT on one GPU and No-Classifier on the
     other, over all distinct `h` in `texts.jsonl`. Write jsonl `{h, p}` incrementally. Attach
     weights through `dataset_sources` (`chunkybert-political-bias`) and `kernel_sources`
     (`chunkybert-best-variant`).
  4. **After B1:**
     - Kernel B2: `--stage nli`, then `--stage qe`, each in its own try/except. Run a `--limit 5`
       QuestEval smoke test first.
     - Kernel C: classifiers on the 4,000 LED rewrites.
     - Locally: Jev and VAD, both light.

  Logs are only downloadable after a kernel finishes.
- **Data provenance:**
  - `data/own_test.jsonl` is byte-identical to `origin/manas:dataset/llm_neutral_prepared/test/baseline.jsonl`.
  - `external/DiverseSumm` is `github.com/salesforce/DiverseSumm` @ `42f22e3`; the file used is
    `data/diverse_summ.json`.
- **Storage on a 10 GB cloud box:** the job needs about 250 MB of data and no model weights:
  - data: `stories_10k`, both dedup files, `texts.jsonl`, `own_test.jsonl`, `diverse_summ.json`;
  - weights: none, because Kaggle pulls LED from HF and the classifiers from Kaggle;
  - Python: no `torch` needed locally (`build_texts`, VAD, Jev and the analysis need only stdlib,
    pandas, sklearn and `typesafe_sdk`);
  - secrets: recreate `.env` (`JEV_API`) and `~/.kaggle/access_token`.

### Next (in order)
1. **`led_eval/progress.py`** (designed, not written). One watcher loop, every ~30 s:
   - counts lines in each stage's incremental jsonl output against its expected total, and polls
     `kaggle kernels status` (match `KernelWorkerStatus.COMPLETE|ERROR|CANCEL` exactly);
   - renders `led_eval/RESULTS.md`: an overall weighted bar plus a per-stage table at the top, then
     the per-stage result fragments `led_eval/results/<stage>.md`;
   - runs that stage's analysis once it reaches 100%.

   Use `fcntl` locks if several writers share state.
2. **LED generation** of the 2,000 test articles per model, appended to jsonl per batch. Time a small
   `--limit` run on the local RTX 3050 (4 GB) first; if the full run exceeds 1 h, do it on Kaggle.
3. **Classifiers, on Kaggle** (see the plan above), over every distinct `h`, resumable jsonl keyed
   by `h`. Reuse the notebooks' `score()` code verbatim:
   - ChunkyBERT (`MAX_LEN` 5000, `forward(ids, mask, tokenizer)`)
   - No-Classifier (`VariantClassifier(3, **TABLE10["No Classifier"])`)

   That is about 40.8k texts × 2 models. No Baly. Then run **VAD** locally, which takes seconds
   (reuse `vad()` from `evaluation/vad/vad_scores.py`).
4. **Jev** (import `Ledger`, `score_all` and `QUESTIONS` from `jev/jev_classify.py`, plus `QUESTIONS`
   and `flatten` from `jev/bias_types.py`; items need keys
   `id, story_id, split, slot, label, publisher, title=None, text`):
   - ideology `outlet_style`, body-only, on the full test panel of about 10k texts: ≈ $0.24;
   - bias types on the first 600 of `random.Random(7).sample(test_stories, 750)`, with
     originals/nemotron/λ0/λ035 body-only: about 4,800 texts, ≈ $0.30. Reuse the existing centre and
     LLM-summary rows from `jev/results/bias_types.jsonl`, noting that the centre article was scored
     with its title.

   Apply the dev-fitted prior correction as `jev/jev_eval.ipynb` does.
5. **Kaggle QA kernel(s):** generation on DiverseSumm (245 × 10 articles × 2 models), NLI coverage,
   then QuestEval on own_test for {reference, nemotron, λ0, λ035}. Upload own_test, DiverseSumm and
   the Nemotron targets as a private dataset, and see the Kaggle mechanics in `CLAUDE.md`.
6. **Analysis per stage:**
   - full probability vectors;
   - paired original → rewrite shift, with P(original side) down and P(Centre) up;
   - the share whose argmax moves to Centre;
   - "label still recovered" accuracy (lower is better);
   - bias-type paired win-rate vs the story's own original;
   - VAD deltas.

   Report λ0 vs λ0.35 vs Nemotron vs centre vs LLM summary. On the data panel, also report
   original vs Nemotron by split.

### Caveats to keep in the write-up
- The classifiers mostly learn outlet style (40–48% accuracy on the originals), so a rewrite that
  drops outlet style looks "more neutral" whether or not its slant changed. Lean on Jev and the
  bias types for slant.
- The 512-token input cap means the model sees only the opening of long DiverseSumm articles, which
  caps QA coverage.

### Environment gotchas
- `git fetch origin` hangs here: two attempts timed out at 100–120 s. `origin/data` is already
  current to `2b00cbe` (2026-09-29).
- `find /` is very slow on WSL. Scope searches.
- **Not in git:** these small files are untracked and must be committed for a cloud session to see
  them:
  - `led_eval/`, `qa_eval/`, `handoff.md`
  - `.gitignore`, `.env.example` (its values are empty)

  `data/`, `models/` and `external/` are gitignored. Their sources are listed above, in the
  `CLAUDE.md` Data table and in the other section below.
- `.env` (JEV_API) is not in git either; never print it.

## Chat: classifier replications (ChunkyBERT, Baly, POLITICS) + repo reorganisation (branch `harshit`), handed off 2026-10-01

Covers `chunkybert/`, `replication/`, `baly/`, `evaluation/`, the repo layout, `CLAUDE.md`, `CONTEXT.md`
and `TODO.md`. **All results and how they were obtained are already written up — read those, don't
re-derive:** `CONTEXT.md` (every result, one place), `replication/REPLICATION.md` (ChunkyBERT
Tables 8/10 log), `baly/REPLICATION.md` (Baly log). Rules for how the user wants work done are in
`CLAUDE.md`.

### Done (results in the files above)
- **Repo reorganised** into `chunkybert/ replication/ evaluation/ baly/` + gitignored
  `data/ models/ external/`; all paths fixed and verified; `CLAUDE.md`, `CONTEXT.md`, `TODO.md` written.
- **ChunkyBERT** (authors' code) reproduces: 86.6–86.9% val acc vs paper 86.22%.
- **ChunkyBERT Table 8/10 rebuilt** (no upstream code): GRU+attn reproduces, TCN doesn't; Table 10
  6/7 reproduce; the paper's "No Transformers = 41%" is actually a no-BERT model (we measured 82.3%
  for the described ablation, 37.2% with BERT removed).
- **Baly et al. 2020** (no upstream model code, `baly/baly.py`): majority, BERT random (82.2 F1 vs
  80.2), BERT media (37.1 vs 35.5) reproduce; adversarial adaptation does not (33.7 vs 43.9, single
  seed, very noisy split); outlet classifier acc 78.4 vs 80.1.
- **Scored `stories_10k`** (30k labelled source articles + 10k neutral summaries) with ChunkyBERT,
  the best variant and the 3 Baly models — notebooks `evaluation/{chunkybert_eval,best_variant_eval,baly_eval}.ipynb`,
  per-article probabilities in `evaluation/scores/`. All land at 36–44% acc; Centre collapses (The
  Hill = 61% of centre articles, 5–17% correct); everything leans Right.

### In progress: POLITICS (Liu et al. 2022, arXiv 2205.00619) — scope just agreed, nothing built yet
**User's decision:** reproduce **Table 3 only** (no pretraining — that's ~160–240 Kaggle GPU-h),
**ideology tasks only, 1 seed first**; if it's fast, run a 2nd seed. Compare the released POLITICS
checkpoint against RoBERTa-base (and BERT-base if cheap), measured vs paper.

Established facts:
- Checkpoint `launch/POLITICS` on HuggingFace is public, not gated (RoBERTa-base, CC BY-NC-SA).
- Repo cloned at `external/POLITICS`: pretraining code only. **No fine-tuning/eval code** ("stay
  tuned" since 2022) → rebuild from the paper, and say so.
- Fine-tuning setup (paper §5.4, Table A7): up to 10 epochs, early-stop patience 4 on val macro-F1,
  lr 1e-5 or 2e-5, linear decay with 6% warmup, AdamW wd 0.001 betas (0.9, 0.999), `[CLS]` → 2-layer
  FFNN (hidden 768, dropout 0.1); sliding window 512 / overlap 64 with mean pooling only for the
  user-level tasks. Metric: macro-F1.
- Paper ideology targets (macro-F1, mean of 5 runs; BERT / RoBERTa / POLITICS):
  YT-cmt 64.64/66.72/67.83 · CongS 65.88/67.25/70.86 · HP 48.42/60.43/70.25 ·
  AllS 60.88/74.75/74.93 · YT-user 65.24/67.98/78.73 · TW 44.20/48.90/48.92.
- **Data availability** (paper §8.2, App. D) — not yet verified by download:
  - CongS (Gentzkow et al., Stanford congress text): public. Filter speeches <80 words, label =
    speaker party. Paper: 7,000 train, random split — split not released, must be recreated.
  - HP (SemEval-2019 Hyperpartisan, webis/zenodo, may need access request): converted to 3-way by
    projecting *media-level* ideology onto articles; the paper does not list the publisher→ideology
    map — ambiguity to flag/ask about. Paper: 425 train.
  - AllS: on request from the authors only. YT and TW: authors cannot share. → likely only CongS +
    HP are runnable; tell the user before substituting anything.
- Contamination note for any later use on our data: POLITICS was pretrained on exactly the 11
  `stories_10k` publishers (2000–2021); `stories_10k` story IDs go up to 1,058,145, near BIGNEWSALIGN's
  1,060,512 clusters. Unverified without BIGNEWS (form-gated).

Next: (1) download/verify CongS and HP, ask about the HP media map; (2) one trainer (`politics/`,
mirror `baly/baly.py` style) for the three encoders; (3) time a local `--limit` run, then Kaggle
(ask before pushing); (4) `politics/REPLICATION.md` progress log, measured vs paper.

### Open decisions waiting on the user
- **Baly triplet-loss pre-training** (`baly/kaggle/tlp/`, ~1.7 h Kaggle, built + smoke-tested) — on hold.
- **Baly AA re-run with 2–3 seeds** to settle the non-reproduction (~3.5 h/seed) — offered, not approved.
- `harshitlalwani/chunkybert-political-bias` public visibility never confirmed.
- Nothing committed; `git fetch` hangs here (see above).

### What a cloud session needs (none of it is in git)
- **Kaggle** (CLI auth was `~/.kaggle/access_token`; re-authenticate in the cloud): datasets
  `harshitlalwani/{media-bias-csv, baly-article-bias, chunkybert-political-bias}`; finished kernels
  `chunkybert-full-train`, `chunkybert-table10-ablations`, `chunkybert-best-variant`, `baly-bert-main`.
  Weights can be re-downloaded from their outputs with
  `kaggle kernels output <ref> -p dir --file-pattern '.*\.pt$'`:
  `chunkybert_final_epoch.pt` → `models/chunkybert_5ep.pt`;
  `best_variant_no_classifier.pt` → `models/no_classifier_1ep.pt`;
  `results/{bert_media,bert_aa_media,bert_random}.pt` → `models/baly/`.
- **Data:** regenerate per the `CLAUDE.md` Data table; Baly's article table is
  `python baly/baly.py prep` (needs `external/Article-Bias-Prediction`). `stories_10k` needs `gh`
  auth to the private team repo.
- **External clones:** `dloiya/chunkybert`, `ramybaly/Article-Bias-Prediction`, `launchnlp/POLITICS`
  into `external/`.
