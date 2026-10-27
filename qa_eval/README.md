# QA content-preservation eval of the LED neutralisers

Do the LED rewrites keep the facts of the article they rewrite? Two QA-based checks, run on Kaggle
(2×T4) on 1–2 Oct 2026, on Raunak's `RaunakSeksaria/anlp-neutral-led` checkpoints
`side-neutral-full/lambda0` and `side-neutral-full/lambda035`.

- **QuestEval** (text2text): questions generated from one text are answered on the other.
  - **Precision**: questions from the rewrite, answered on the source. Measures faithfulness.
  - **Recall**: questions from the source, answered on the rewrite. Measures coverage.
  - **QuestEval** is their mean.

  The Nemotron rewrite, which is the LED training target (`target_text`), is scored the same way as a reference
  system. The human centre article was not scored.
- **DiverseSumm NLI coverage**: DiverseSumm (245 stories × 10 articles) ships reference answers, each tied to
  the article it came from. An answer counts as covered by a text if
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` gives P(entailment) > 0.5 for some 320-token chunk
  of that text. The original article is the control.

## Results

`python summarize.py` prints these tables from `results/`.

**QuestEval, `side_neutral_full` test split** (1,775 held-out articles; LED input 512 tokens, greedy, ≤ 512 new):

| System | QuestEval | Precision | Recall |
|---|---|---|---|
| Nemotron (target) | 0.670 | 0.704 | 0.636 |
| LED λ0 | 0.674 | 0.710 | 0.638 |
| LED λ0.35 | 0.673 | 0.709 | 0.637 |

**QuestEval, long1k sample** (300 articles = 150 events × left + right, `random.Random(7)`; LED input 1,024
tokens, ≤ 1,024 new; the models were trained at 512):

| System | QuestEval | Precision | Recall |
|---|---|---|---|
| Nemotron (target) | 0.620 | 0.656 | 0.585 |
| LED λ0 | 0.624 | 0.654 | 0.594 |
| LED λ0.35 | 0.623 | 0.653 | 0.592 |

**DiverseSumm NLI coverage** (6,559 answers, 245 stories, 1,743 articles; LED input 512 tokens):

| Text | Answers covered | Kept, of the answers the original covers |
|---|---|---|
| Original article (control) | 83.5% | — |
| LED λ0 | 62.5% | 70.8% |
| LED λ0.35 | 62.1% | 70.1% |

Reading:
- The LED rewrites keep as much of the source as their Nemotron target does: λ0 and λ0.35 are within
  0.005 QuestEval of Nemotron, and of each other. One seed per model, so these differences are noise.
- Rewriting drops facts: about 30% of the answers the original article supports are no longer supported by
  its rewrite.
- Long articles score lower for every system, the target included (0.62 vs 0.67).
- At 512 input tokens the LED sees only the opening of a long DiverseSumm article, which caps its coverage.

## Files

| Path | What |
|---|---|
| `qa_eval.py` | Stages `gen` (LED rewrites), `nli` (DiverseSumm coverage), `qe` (QuestEval). This is the version the kernels ran |
| `summarize.py` | Prints the tables above |
| `kaggle/<kernel>/` | The four Kaggle kernels as run, with their `kernel-metadata.json` |
| `inputs/led_long1k.jsonl` | 2,000 long1k articles (1,000 left + 1,000 right), `qa_eval.py` record format |
| `inputs/led_long1k_qe300.jsonl` | The 300-article QuestEval sample of it |
| `generations/own_test/gen_own_lambda{0,035}.jsonl` | LED rewrites of the 1,775 test articles (`id`, `output`) |
| `generations/long1k/gen_own_lambda{0,035}.jsonl` | LED rewrites of the 2,000 long1k articles (1,024-token input) |
| `generations/diversesumm/gen_diversesumm_lambda{0,035}.jsonl` | LED rewrites of every DiverseSumm article |
| `results/own_test/`, `results/long1k_qe300/` | Per-article QuestEval: `id`, `questeval`, `precision`, `recall` |
| `results/diversesumm/diversesumm_items.csv` | Per answer: story `eid`, question `q`, answer group `g`, article `aid`, P(entail) for original / λ0 / λ0.35 |

Inputs not in this folder:
- **Test split (`led_test.jsonl`)**: identical to `origin/data:dataset/side_neutral_full/prepared/test/baseline.jsonl`.
- **`diverse_summ.json`**: from https://github.com/salesforce/DiverseSumm (commit `42f22e3`), `data/diverse_summ.json`.

## Rerunning

`qa_eval.py` finds every input by file name anywhere under `/kaggle/input`, its own folder, or `.`. Each stage
skips outputs that already exist, so copy `generations/*` into `--out` (or pass `--reuse`) to skip generation.

```bash
pip install questeval --no-deps && pip install unidecode evaluate bert_score spacy transformers torch pandas
python -m spacy download en_core_web_sm
python qa_eval.py --stage gen --out out                                        # 512-token test + DiverseSumm
python qa_eval.py --stage nli --out out
python qa_eval.py --stage qe  --out out                                        # test split
python qa_eval.py --stage gen --own-file led_long1k.jsonl --sets own --max-in 1024 --max-new 1024 --out long
python qa_eval.py --stage qe  --own-file led_long1k_qe300.jsonl --out long
```

Measured on 2×T4: NLI 22 min; QuestEval 4.3–4.9 h per system on the 1,775 articles (one system per GPU, 8.7 h
for three), 64 min per system on the 300 long ones. The first long run lost a QuestEval worker to an out-of-memory
kill (exit -9) with both workers loaded; the rerun scored the remaining system alone.
