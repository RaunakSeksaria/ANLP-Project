# Results compilation

This compiles the reported values from the saved evaluations. Values below are copied from the result files without rerunning the models. `λ = 0.1` has no saved embedding matrix or evaluation result. All evaluated matrices contain **11,855 × 768** original-article embeddings (5,695 left; 6,160 right). Neutral rewrites were not evaluated.

## Alignment and uniformity

Source: [alignment_uniformity.csv](alignment_uniformity.csv) ([JSON](alignment_uniformity.json)); settings and sampled IDs: [evaluation_manifest.json](evaluation_manifest.json). Scores use L2-normalized embeddings. Lower alignment means closer source-paired cross-ideology articles; lower uniformity means more spread-out embeddings. Intervals are 95% percentile bootstrap intervals.

| λ | Alignment | Alignment 95% CI | Positive pairs | Uniformity | Uniformity 95% CI | Articles | Unique distances |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | 0.3086159662748418 | [0.30291296594109945, 0.3145097035770174] | 2,000 | -0.7577847246948665 | [-0.7661890226633191, -0.7474506863554897] | 2,000 | 1,999,000 |
| 0.35 | 0.3763947787796613 | [0.37042135206500215, 0.3826333196117481] | 2,000 | -1.965877412968375 | [-1.9718485941119464, -1.9546552274973228] | 2,000 | 1,999,000 |

The 2,000 articles comprise 1,000 from each ideology. There are **8,349** distinct eligible cross-ideology pairs in the full metadata; 2,000 were sampled for alignment. Uniformity uses all unordered pairs among its 2,000 sampled articles. Seed: **42**. Bootstrap replicates: **500**, using article resampling for uniformity and pair resampling for alignment. Plots: [alignment](../plots/alignment_comparison.png), [uniformity](../plots/uniformity_comparison.png) (also available as PDFs in `plots/`).

## Ideology clustering

Source: [ideology_clustering.csv](ideology_clustering.csv) ([JSON](ideology_clustering.json)); settings and sample IDs: [ideology_clustering_manifest.json](ideology_clustering_manifest.json) and [ideology_sample_ids.csv](ideology_sample_ids.csv). Both models use the same **2,000** articles (1,000 left, 1,000 right). K-means uses **k = 2**, seed **42**, and **20** initializations. Silhouette uses true ideology labels with cosine distance; centroid distance is `1 − cosine(left centroid, right centroid)`.

| λ | ARI | NMI | Ideology silhouette | Centroid cosine distance | K-means inertia | Cluster sizes | PCA PC1 variance | PCA PC2 variance |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 0.004831848541140814 | 0.003854684159336266 | 0.010827168747488725 | 0.0028169710090937272 | 339.19675405176287 | [963, 1037] | 0.18807072607907488 | 0.08149627947391784 |
| 0.35 | 0.00042828815818096494 | 0.0007028281706775694 | 0.003384154184434414 | 0.004573029113875915 | 951.2325694579719 | [1238, 762] | 0.10848124514091162 | 0.10280020449708926 |

PCA figures: [λ = 0](../plots/ideology_pca_lambda_0.png), [λ = 0.35](../plots/ideology_pca_lambda_035.png) (PDFs also available). The PCA fit is separate for each model and is qualitative. These are single-seed clustering results without uncertainty intervals.

## Triplet separation

Source: [triplet_separation.csv](triplet_separation.csv) ([JSON](triplet_separation.json)); the **2,000** exact triplets are in [sampled_triplets.csv](sampled_triplets.csv), their **4,000** model-specific distance rows are in [triplet_per_sample.csv](triplet_per_sample.csv), and settings are in [triplet_eval_manifest.json](triplet_eval_manifest.json). The same triplets are used for both models. Each of the **6,000** selected articles appears once. Seed: **42**. There are **98,960,697** potential directed triplets under the saved pairing rules.

Distances are squared L2 on normalized embeddings: `d_pos = ||anchor − positive||²`, `d_neg = ||anchor − negative||²`, `gap = d_neg − d_pos`. Ordering success means `gap > 0`. The reported requested margin is **0.2** in squared-distance units; the actual training margin uses unsquared L2 with `p = 2` and `eps = 1e-6`.

| λ | Triplets | Mean d_pos | Mean d_neg | Mean gap | Median gap | Ordering success | Mean (gap − 0.2) | Squared margin satisfied | Training margin satisfied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2,000 | 0.30304828710081677 | 0.38969693852664095 | 0.0866486514258242 | 0.08916106957621456 | 72.95% | -0.11335134857417582 | 23.05% | 19.950000000000003% |
| 0.35 | 2,000 | 0.37681067869052287 | 1.0304064415300536 | 0.6535957628395306 | 0.6681405362075511 | 98.85000000000001% | 0.4535957628395306 | 94.6% | 88.25% |

| λ | Mean d_pos 95% CI | Mean d_neg 95% CI | Mean gap 95% CI |
| --- | --- | --- | --- |
| 0 | [0.29718871480223247, 0.3091688866560654] | [0.3843045548995828, 0.39501899412669417] | [0.07921765716459217, 0.09411718140447307] |
| 0.35 | [0.37065946470229266, 0.3835929309657327] | [1.020832847961367, 1.040350528375925] | [0.6412534200981027, 0.6656654122346903] |

**Paired gap change (λ 0.35 − λ 0): 0.5669471114137064**, 95% CI **[0.5560097835738623, 0.5775599010906546]**, across 2,000 triplets. Confidence intervals use **2,000** paired percentile bootstrap replicates, seed **43**. Plots: [positive and negative distances](../plots/triplet_distances.png), [separation gap](../plots/triplet_separation_gap.png) (PDFs also available).

## Embedding audit

Source: [embedding_audit.json](embedding_audit.json). Audit status: **PASS**; warnings: **0**. The ten-article re-extraction spot check also **PASS**ed on CUDA. All **11,855** original-body hashes were validated; duplicate original bodies: **0**. Both matrices have shape **[11855, 768]**, dtype **float32**, and **0** NaNs, positive/negative infinities, duplicate embedding rows, near-zero rows, or abnormal-norm rows.

| λ | Value min | Value max | Value mean | Value std | Norm min | Norm max | Norm mean | Norm std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.5684704184532166 | 0.5156863927841187 | 0.0017923125122352058 | 0.036039852237242834 | 0.9999998931234072 | 1.00000010386293 | 1.0000000000347433 | 3.4159726200015496e-08 |
| 0.35 | -0.5003805756568909 | 0.45649978518486023 | 0.002263042686679992 | 0.036013358243060165 | 0.9999998771954911 | 1.0000001136324563 | 1.0000000003115737 | 3.591969063830843e-08 |

Corresponding rows across models: mean cosine **0.8224386578158**, mean L2 distance **0.5913323884186105**, maximum absolute difference **0.30513789504766464**; the matrices are neither exactly nor nearly identical.

| λ | Spot-check mean cosine | Spot-check minimum cosine | Maximum absolute difference | Mean absolute difference |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.9999999999999865 | 0.9999999999999121 | 5.541369318962097e-08 | 2.0279870227568607e-09 |
| 0.35 | 0.9999999999999737 | 0.9999999999998572 | 6.798654794692993e-08 | 2.9836137116490135e-09 |

Spot-check thresholds: maximum absolute difference **2e-5**, minimum cosine **0.99999**. The sampled row indices and all per-row comparisons remain in [embedding_audit.json](embedding_audit.json). Both checkpoints record seed **42**, best epoch **3**, global optimizer step **4,683**, input/target limits **512**, and triplet margin **0.2**. Best validation generation loss: **0.42020329994566913** for λ 0; **0.4254892829750335** for λ 0.35.

## Source pairing ID audit

Source: [event_id_audit.json](event_id_audit.json); detailed manual reviews: [event_id_audit.md](event_id_audit.md). The `event_ids` values are raw source pairing IDs, not verified real-world event labels.

| Measure | Value |
| --- | ---: |
| Articles | 11,855 |
| Source pairing rows / distinct IDs | 10,000 |
| Article–ID memberships | 20,000 |
| Articles with multiple IDs | 4,009 |
| IDs per article: min / median / mean / max | 1 / 1 / 1.687051876845213 / 32 |
| Articles per ID: min / median / mean / max | 2 / 2 / 2 / 2 |
| IDs with both ideologies | 10,000 |
| Distinct left–right pairings | 8,349 |
| Exact-body/source-story memberships matching metadata | 20,000 / 20,000 |
| Title variants with identical body | 2 |

**IDs per article histogram (ID count → article count):** 1→7,846; 2→2,293; 3→811; 4→394; 5→208; 6→94; 7→72; 8→43; 9→26; 10→23; 11→15; 12→10; 13→3; 14→4; 15→2; 16→1; 17→2; 18→1; 19→2; 21→1; 23→1; 25→1; 26→1; 32→1.

**Source IDs per distinct article pairing (ID count → pairing count):** 1→7,110; 2→970; 3→190; 4→43; 5→20; 6→9; 7→4; 8→2; 10→1. Every source ID has exactly two articles (histogram: 2→10,000).

Of **20** seed-42 manual reviews drawn from the 2,000 saved triplets, **11** were judged the same specific story, **5** related developments, **3** different events, and **1** topic only or insufficient. These are qualitative judgments, not a precision estimate for the full sample. The reviewed triplet IDs and individual reasons are in [event_id_audit.md](event_id_audit.md).

## Interpretation and provenance

The λ 0.35 model has a larger mean triplet gap and lower uniformity, while its mean positive-pair squared distance is also larger. The ideology clustering scores are close to zero for both models. These numbers describe the saved original-article embeddings; they do not establish neutrality or generation quality.

The source ID audit found semantically noisy positives: some source-linked pairs are separate events. Consequently, “10,000 events” means **10,000 source pairing IDs**, and the triplet metrics should be interpreted against those IDs. The audit did not change any triplet scores.

All matrices rely on the extraction README's shared row-order guarantee. The arrays have no embedded row IDs; the ten-row re-extraction is sampled identity verification, not proof for every row. Bootstrap intervals describe sampling variation conditional on the saved data and do not account for training-run variation. Alignment and triplet intervals can also understate uncertainty when pairs share articles or source IDs.

The complete result values at individual sample, article, and pairing granularity remain in the linked CSV/JSON files above. Input file hashes, checkpoint mappings, package versions, and all sampled identifiers are retained in the linked manifests and audit JSON.
