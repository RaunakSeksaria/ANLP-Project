# TODO

Status: `[ ]` open · `[~]` in progress · `[x]` done

## A. Reorganize the codebase
- [x] A1 New layout: `chunkybert/`, `replication/`, `evaluation/`, `baly/`, `data/`, `models/`, `external/`
- [x] A2 Move files; remove the byte-identical copies made only for Kaggle uploads
- [x] A3 Fix hard-coded paths in scripts and notebooks; check every referenced path resolves
- [x] A4 Update `.gitignore` (data, weights, external clones, Kaggle outputs)

## B. Write `CLAUDE.md`
- [x] B1 Layout, how to run things, where data and weights live, conventions and gotchas

## C. Write `CONTEXT.md`
- [x] C1 What the project is, everything done so far, every result, open issues

## D. Replicate Baly et al. 2020 ("We Can Detect Your Bias", arXiv 2010.05338)
Scope: BERT models only. No LSTM, no hyperparameter search (use the paper's reported best
settings), no Twitter-bio or Wikipedia media-level models.

- [x] D1 Shared BERT trainer: released splits, macro-F1 / accuracy / MAE, early stopping on val macro-F1
- [x] D2 Majority baseline (Table 3)
- [x] D3 BERT, media-based split (Table 3; Table 5 "None")
- [x] D4 BERT, random split (Table 3)
- [x] D5 BERT medium classifier, predict the outlet (Table 4)
- [x] D6 BERT + adversarial media adaptation, λ = 0.7 (Table 5)
- [ ] D7 BERT + triplet-loss pre-training (Table 5) — on hold, awaiting decision
- [x] D8 Local smoke tests of all five BERT models
- [~] D9 Full runs on Kaggle
- [~] D10 Results: measured vs paper, in `baly/REPLICATION.md`
- [x] D11 Score `stories_10k` with the three Baly ideology models (`evaluation/baly_eval.ipynb`)

## E. Jev zero-shot side experiment
- [x] E1 Compare 4 prompt designs × 2 state variants on 450 dev articles
- [x] E2 Full run with the best design (29,550 labelled + 8,001 neutral), $0.95 of $1
- [x] E3 Results notebook `jev/jev_eval.ipynb`, compared against the BERT models
- [x] E4 AllSides bias types on 3,000 texts (`jev/bias_types.ipynb`), $0.19
- [ ] E5 Omission bias: the three articles of a story compared in one request (not run)

## F. Held aside (parked 2026-10-25; nothing running)
- [ ] F1 Fold into `RESULTS.md`: new LED `first_token_pooling` results (classifiers, outlet leakage, VAD,
      Jev, bias types) and POLITICS-on-LED tables. Numbers are in the chat log and these files:
      `led_eval/kaggle/new_model/output/out/`, `politics/kaggle/{media,random_score}/output*/`,
      `jev/results/{led_outputs,bias_types_led}.jsonl` (`jev/bias_types_led_report.py` prints the tables)
- [ ] F2 Resume paused classifier queue on the Nemotron rewrite sets: `evaluation/long10k/run_all.sh`
      (resumable). Done: ChunkyBERT + Baly random on both sets. Left: Baly media (long 51% done),
      Baly AA, outlet classifier, No-Classifier, on both short and long sets. Then execute
      `evaluation/{short,long10k}_eval.ipynb`
- [ ] F3 POLITICS Table 3 replication (ideology tasks, 1 seed): Congress Speech + Hyperpartisan.
      Hyperpartisan by-article data downloaded (Zenodo 5776081); need media→ideology mapping decision
- [ ] F4 Jev: re-score short/long-set originals body-only for an exact paired comparison (~$1.25; spend at $2.84)
- [x] F5 QA / QuestEval content-preservation eval: ran on Kaggle 1–2 Oct; results and code in `qa_eval/` (README). Open: score the human centre article as a reference
- [ ] F6 Baly AA with 2-3 more seeds (~3.5 h Kaggle each); Baly triplet pre-training (D7)
- [ ] F7 Seeds for the LED comparison (first-token vs λ0 vs λ0.35 differences are within single-seed noise)

## Carried over (not started)
- [ ] Publisher-leakage classifier (`eval_metrics.md` item 2)
- [ ] Commit the work (everything is currently untracked on branch `harshit`)
- [ ] LED decoding: rerun E2-long generation with `no_repeat_ngram_size=3` / `repetition_penalty` (33-46% of LED rewrites loop; `quality_eval/README.md`), then re-score with a faster MiniCheck variant
