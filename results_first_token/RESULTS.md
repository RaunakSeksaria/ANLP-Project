# First-token embedding evaluation

The selected checkpoint is `outputs/side_neutral_first_token_lambda035_21275/best`
(epoch 3, standard triplet objective, λ=0.35). Original-article embeddings were
extracted for the same 11,855 metadata rows used by the saved masked-mean
evaluations. No neutral rewrites or generated text were embedded.

| Metric | Masked mean λ=0 | Masked mean λ=0.35 | First token λ=0.35 |
| --- | ---: | ---: | ---: |
| APD, mean squared L2 (2,000 fixed triplets) | 0.3030 | 0.3768 | **0.0799** |
| MSG, mean squared-distance separation gap (2,000 fixed triplets) | 0.0866 | 0.6536 | **0.6447** |
| Alignment, mean squared L2 (2,000 fixed positive pairs) | 0.3086 | 0.3764 | **0.0776** |
| Uniformity, log mean exp(-2 squared L2) (2,000 articles) | -0.7578 | -1.9659 | **-1.2633** |
| Ideology probe accuracy, stratified 5-fold | 0.7902 | 0.7570 | **0.6919** |
| Ideology probe accuracy, publisher-held-out 3-fold | 0.3228 | 0.3366 | **0.4241** |

First-token APD 95% bootstrap CI: [0.0762, 0.0837]. First-token MSG CI:
[0.6300, 0.6593]. The paired MSG change versus masked-mean λ=0.35 is
-0.0089, 95% CI [-0.0207, 0.0020]. First-token alignment CI:
[0.0741, 0.0813]; uniformity CI: [-1.2750, -1.2497].

The first-token stratified probe's macro-F1 is 0.6913 and ROC-AUC is 0.7562;
publisher-held-out macro-F1 is 0.4214 and ROC-AUC is 0.3945. The stratified
label-permutation check produced 0.5034 accuracy. Publisher-held-out accuracy
below 0.5 should not be interpreted as proof of neutrality; it can reflect
publisher/domain shift or inverted predictive patterns.

APD and MSG use the exact saved triplets in `results/sampled_triplets.csv` and
squared L2 distances on float64-renormalized copies. Alignment and uniformity
reuse the exact saved article/pair indices. The probe uses identical folds.
The stored first-token embeddings are float32, finite, and unit-normalized.
Detailed values and provenance are in this directory's JSON/CSV files and
`embedding_data_first_token/manifest.json`.

This is not a pure pooling-only causal comparison: the previous masked-mean
λ=0.35 checkpoint records 4,683 optimizer updates by best epoch 3, versus
6,168 for this two-GPU first-token run. The same evaluation articles and
sampling/folds make the reported metrics comparable descriptively, but training
exposure and distributed order also differ.
