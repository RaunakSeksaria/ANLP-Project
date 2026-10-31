# cc2025 evaluation: eight LED neutralisers on 2025–26 articles

The eight LED models rewrite 2,160 news articles from Raunak's 2025–26 Common Crawl collection, and Harshit's
POLITICS-v2 classifier scores the originals and every rewrite. These articles are newer than all training data
(dates 2025-01-01 to 2026-10-03; none of them is in the training files).

The articles come from Raunak's `cc2025_dataset.zip`, which is **not** in this repository: the text belongs to the
publishers, so only code and summary numbers are here.

## Articles

Pool: `cc2025_strict_LR` ∪ `cc2025_mixed_LR` = 12,500 unique articles (left = Daily Kos, Salon; right = Fox,
Breitbart). `select_cc2025.py` keeps an article when
- `title </s> body` fits in the models' 512-token input (LED tokenizer), and
- the original has **P(own side) ≥ 0.75** under Raunak's stored POLITICS-v2 scores (Manas's threshold).

Result: **2,160 articles**: left 1,120 (Daily Kos 843, Salon 277), right 1,040 (Breitbart 747, Fox 293).
There are no Nemotron neutral targets for these articles, and no event links between left and right articles.

## Models

| Key | Model | HF repo / subfolder | Training data |
| --- | --- | --- | --- |
| `baseline_lambda0` | generation only (λ 0) | `RaunakSeksaria/anlp-neutral-led` / `side-neutral-full/lambda0` | side_neutral_full (12,484) |
| `ideology_standard_m04` | first token, standard triplet, margin 0.4 | `tchaitanya0109/anlp-neutral-led-ideology-first-token` / `standard-margin04-lambda035` | side_neutral (16,447) |
| `ideology_attraction_m02` | first token + attraction, margin 0.2 | same repo / `attraction-margin02-lambda035` | side_neutral (16,447) |
| `ideology_attraction_m05` | first token + attraction, margin 0.5 | same repo / `attraction-margin05-lambda035` | side_neutral (16,447) |
| `ideology_attraction_m03_b05_t03` | first token + attraction (β 0.5, τ 0.3), margin 0.3 | same repo / `attraction-margin03-beta05-tau03-lambda035` | side_neutral_full (12,484) |
| `ideology_attraction_m03_b06_t05` | first token + attraction (β 0.6, τ 0.5), margin 0.3 | same repo / `attraction-margin03-beta06-tau05-lambda035` | side_neutral_full (12,484) |
| `publisher_standard_m02` | first token, standard triplet, margin 0.2, same-publisher negatives | `tchaitanya0109/anlp-neutral-led-publisher-first-token` / `standard-lambda035` | side_neutral_publisher (16,447) |
| `publisher_attraction_m02` | first token + attraction, margin 0.2, same-publisher negatives | same repo / `attraction-lambda035` | side_neutral_publisher (16,447) |

All use λ 0.35 except the baseline; attraction models use β 0.3, τ 0.5 unless stated. One training seed per model.

## How it was run

```bash
python chaitanya/cc2025_eval/select_cc2025.py <cc2025 folder> <politics_v2>/calibration.json <LED tokenizer folder> \
    dataset/cc2025/cc2025_eval.jsonl
python chaitanya/cc2025_eval/generate_cc2025.py --model <key>       # once per model, one GPU (RTX 2080 Ti on Ada)
python chaitanya/cc2025_eval/score_cc2025_politics.py               # all eight models, one GPU
```

- **Generation** follows the training-time validation decoding: input `title </s> body`, max 512 input tokens,
  global attention on token 0, greedy (`num_beams=1`), `max_new_tokens=512`, fp32, one article at a time, with the
  project's `load_led_triplet` and `generate_with_cache`. About 1.5–2 s per article.
- **Scoring:** POLITICS v2, 4-seed calibrated ensemble (`seed42–45_best.pt`, T = 3.079; Kaggle dataset
  `harshitlalwani/politics-ideology-classifier`). Outlet names are masked with Raunak's `OUTLET_RE` plus `Salon`,
  and texts use 512-token windows with 64 overlap. P(own) is the calibrated probability of the article's own side.
  Our scores of the originals reproduce Raunak's stored scores: same top label for 100 %, max |Δ P(own)| 0.0016.

## Results

Lower P(own) = more neutral. Δ = paired difference to the baseline with a 95 % bootstrap CI over articles.

| Text scored | Mean P(own) | Median P(own) | Own side top | Centre top | Δ mean P(own) vs baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| Original article | 0.968 | 0.995 | 100.0 % | 0.0 % | — |
| Baseline λ0 | 0.708 | 0.845 | 76.9 % | 8.7 % | — |
| Ideology: standard, margin 0.4 | 0.697 | 0.840 | 76.2 % | 8.5 % | −0.010 [−0.019, −0.001] |
| **Ideology: attraction, margin 0.2** | **0.684** | **0.812** | **74.7 %** | 8.1 % | **−0.023 [−0.032, −0.014]** |
| Ideology: attraction, margin 0.5 | 0.696 | 0.831 | 76.4 % | 8.3 % | −0.011 [−0.020, −0.002] |
| Ideology: attraction, m 0.3, β 0.5, τ 0.3 | 0.704 | 0.848 | 76.6 % | 8.6 % | −0.003 [−0.012, +0.006] |
| Ideology: attraction, m 0.3, β 0.6, τ 0.5 | 0.705 | 0.851 | 76.9 % | 7.5 % | −0.002 [−0.011, +0.007] |
| Publisher: standard, margin 0.2 | 0.688 | 0.824 | 74.7 % | 8.6 % | −0.019 [−0.028, −0.010] |
| Publisher: attraction, margin 0.2 | 0.690 | 0.832 | 74.7 % | 8.8 % | −0.018 [−0.027, −0.008] |

Mean P(own) by outlet:

| Text scored | Daily Kos (843) | Salon (277) | Fox (293) | Breitbart (747) | Left (1,120) | Right (1,040) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original article | 0.969 | 0.943 | 0.950 | 0.982 | 0.962 | 0.973 |
| Baseline λ0 | 0.612 | 0.452 | 0.846 | 0.856 | 0.572 | 0.853 |
| Ideology: standard, margin 0.4 | 0.588 | 0.453 | 0.856 | 0.850 | 0.554 | 0.852 |
| Ideology: attraction, margin 0.2 | 0.566 | 0.442 | 0.843 | 0.846 | 0.535 | 0.845 |
| Ideology: attraction, margin 0.5 | 0.589 | 0.450 | 0.846 | 0.850 | 0.554 | 0.849 |
| Ideology: attraction, m 0.3, β 0.5, τ 0.3 | 0.601 | 0.462 | 0.852 | 0.852 | 0.567 | 0.852 |
| Ideology: attraction, m 0.3, β 0.6, τ 0.5 | 0.603 | 0.456 | 0.851 | 0.857 | 0.567 | 0.855 |
| Publisher: standard, margin 0.2 | 0.574 | 0.439 | 0.843 | 0.849 | 0.541 | 0.847 |
| Publisher: attraction, margin 0.2 | 0.579 | 0.436 | 0.846 | 0.848 | 0.544 | 0.847 |

- Every model moves the articles towards neutral (0.968 → about 0.70), but almost all of the drop is on the left
  articles (to about 0.54–0.57). Right articles only drop to about 0.85.
- The first-token triplet models with margin 0.2–0.5 (ideology and publisher) are slightly more neutral than the
  baseline (−0.010 to −0.023). The two margin-0.3 models trained on side_neutral_full are not different from it.
- Each of the four classifier seeds scored alone also puts ideology attraction m 0.2 first (model ranking vs the
  ensemble: Kendall τ 0.86–1.0).
- Caveats: one training seed per model; the classifier is the only neutrality measure; there is no reference
  rewrite to compare content against.

`results/politics_summary.json` has these numbers for every group (all, left, right, each outlet, and `strict` =
without Salon), plus median P(own), mean P(Centre) and mean margin.
