# Hard-test evaluation: eight LED neutralisers on Raunak's hard-test

The eight LED models rewrite the articles of Raunak's hard-test (`hard-test/` on the `data` branch @ `69ebfdf`:
validation and test articles that the POLITICS-v2 ensemble puts most confidently on their own side), and Harshit's
POLITICS-v2 classifier scores the originals, the Nemotron targets and every rewrite.

The rewrites are public: [tchaitanya0109/anlp-hard-test-generations](https://huggingface.co/datasets/tchaitanya0109/anlp-hard-test-generations)
(`generations/<model>.jsonl` per model, `merged.jsonl` with all texts per article).

## Articles

The **1,187** hard-test articles whose `title </s> body` fits in the models' 512-token input (LED tokenizer):
711 from the dedup release and 476 from the full1024 release; left 570, right 617. None of them is in a training
split.

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
python chaitanya/hard_test_eval/generate_hard_test.py --model <key>     # once per model, one GPU (RTX 2080 Ti on Ada)
python chaitanya/hard_test_eval/score_hard_test_politics.py             # all eight models, one GPU
```

- **Generation** follows the training-time validation decoding: input `title </s> body`, global attention on
  token 0, greedy (`num_beams=1`), `max_new_tokens=512`, fp32, one article at a time, with the project's
  `load_led_triplet` and `generate_with_cache`. About 1.25 s per article.
- **Checks:** the dedup articles give exactly the source text of the same articles in `dataset/side_neutral`; for
  the three models whose training-time `validation_predictions_best.jsonl` we have (publisher standard and the two
  margin-0.3 models), 24 hard-test articles that are also in those predictions were regenerated (`--check`) and
  came out identical (24/24 each).
- **Scoring:** POLITICS v2, 4-seed calibrated ensemble (`seed42–45_best.pt`, T = 3.079; Kaggle dataset
  `harshitlalwani/politics-ideology-classifier`), outlet names masked with Raunak's `OUTLET_RE`, 512-token windows
  with 64 overlap, `politics()` from `scripts/make_hard_subset.py`. Our scores reproduce the hard-test's stored
  scores: same top label for 100 % of originals and Nemotron targets (max |Δ P(own)| 0.0001 / 0.0023).

## Results

Lower P(own) = more neutral. Δ = paired difference to the baseline with a 95 % bootstrap CI over articles.

| Text scored | Mean P(own) | Median P(own) | Own side top | Centre top | Δ mean P(own) vs baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| Original article | 0.998 | 0.998 | 100.0 % | 0.0 % | — |
| Nemotron target (training target) | 0.691 | 0.864 | 74.2 % | 10.9 % | — |
| Baseline λ0 | 0.761 | 0.944 | 81.8 % | 9.7 % | — |
| Ideology: standard, margin 0.4 | 0.774 | 0.961 | 82.9 % | 8.4 % | +0.013 [+0.002, +0.024] |
| **Ideology: attraction, margin 0.2** | **0.743** | **0.924** | 80.1 % | 8.8 % | **−0.018 [−0.031, −0.006]** |
| Ideology: attraction, margin 0.5 | 0.774 | 0.959 | 82.6 % | 8.7 % | +0.013 [+0.002, +0.024] |
| Ideology: attraction, m 0.3, β 0.5, τ 0.3 | 0.763 | 0.954 | 81.8 % | 8.9 % | +0.002 [−0.009, +0.014] |
| Ideology: attraction, m 0.3, β 0.6, τ 0.5 | 0.767 | 0.961 | 82.1 % | 8.8 % | +0.006 [−0.006, +0.017] |
| Publisher: standard, margin 0.2 | 0.744 | 0.925 | **80.0 %** | 8.8 % | −0.017 [−0.029, −0.005] |
| Publisher: attraction, margin 0.2 | 0.749 | 0.927 | 81.0 % | 8.6 % | −0.012 [−0.024, −0.000] |

Mean P(own) by group:

| Text scored | dedup (711) | full1024 (476) | Left (570) | Right (617) |
| --- | ---: | ---: | ---: | ---: |
| Original article | 0.998 | 0.998 | 0.998 | 0.998 |
| Nemotron target | 0.702 | 0.674 | 0.636 | 0.742 |
| Baseline λ0 | 0.763 | 0.758 | 0.742 | 0.779 |
| Ideology: standard, margin 0.4 | 0.778 | 0.768 | 0.752 | 0.795 |
| Ideology: attraction, margin 0.2 | 0.739 | 0.749 | 0.695 | 0.788 |
| Ideology: attraction, margin 0.5 | 0.777 | 0.770 | 0.753 | 0.793 |
| Ideology: attraction, m 0.3, β 0.5, τ 0.3 | 0.762 | 0.765 | 0.748 | 0.778 |
| Ideology: attraction, m 0.3, β 0.6, τ 0.5 | 0.770 | 0.763 | 0.754 | 0.779 |
| Publisher: standard, margin 0.2 | 0.737 | 0.756 | 0.699 | 0.786 |
| Publisher: attraction, margin 0.2 | 0.743 | 0.759 | 0.707 | 0.789 |

- The originals are almost certainly own-side (0.998, by construction of the hard-test); every model moves them
  towards neutral (to about 0.74–0.77), but not as far as the Nemotron targets (0.691).
- The margin-0.2 models (ideology attraction and both publisher models) are slightly more neutral than the baseline
  (−0.012 to −0.018); margin 0.4 / 0.5 are slightly less neutral (+0.013); the two margin-0.3 models are not
  different from it.
- Most of the gain is on left articles (baseline 0.742 → 0.695 for ideology attraction m 0.2).
- The mean is much lower than the median because the scores are split: for ideology attraction m 0.2, 53 % of the
  rewrites have P(own) > 0.9 and 8 % have P(own) < 0.1.
- Each of the four classifier seeds scored alone also puts ideology attraction m 0.2 first (model ranking vs the
  ensemble: Kendall τ 0.86–1.0).
- Caveats: one training seed per model; the classifier is the only neutrality measure here.

`results/politics_summary.json` has these numbers for every group, plus median P(own), mean P(Centre) and mean
margin. `results/generation_summary.json` has per model the HF repo, weights SHA-256, EOS rate (99.5–100 %) and
mean output length (180–185 words).
