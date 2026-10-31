
**E2 long, all splits** (articles also in Baly's corpus excluded)

| Texts | n | Own side top label % | Left vs Right % | Recall L / R % | Mean P(L/C/R) | Mean P(own side) | Left articles → L/C/R | Right articles → L/C/R | Moved to Centre | Moved to other side |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | 19,783 | 85.3 | 92.5 | 83.9 / 86.8 | 0.43/0.12/0.45 | 0.814 | 84/10/6% | 4/9/87% | — | — |
| Nemotron rewrite | 19,783 | 63.4 | 75.5 | 54.2 / 72.7 | 0.32/0.24/0.44 | 0.582 | 54/21/24% | 10/17/73% | 15.7% | 11.7% |

**E2 long, test split** (articles also in Baly's corpus excluded)

| Texts | n | Own side top label % | Left vs Right % | Recall L / R % | Mean P(L/C/R) | Mean P(own side) | Left articles → L/C/R | Right articles → L/C/R | Moved to Centre | Moved to other side |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | 2,005 | 84.1 | 92.7 | 81.6 / 86.6 | 0.42/0.13/0.45 | 0.805 | 82/12/6% | 3/10/87% | — | — |
| Nemotron rewrite | 2,005 | 62.0 | 73.9 | 52.4 / 71.6 | 0.31/0.24/0.45 | 0.576 | 52/21/26% | 11/18/72% | 14.7% | 12.7% |

**E2 long test split, LED quality-eval set** (1,864 articles not in LED training data)

| Texts | n | Own side top label % | Left vs Right % | Recall L / R % | Mean P(L/C/R) | Mean P(own side) | Left articles → L/C/R | Right articles → L/C/R | Moved to Centre | Moved to other side |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | 1,864 | 84.2 | 93.1 | 81.8 / 86.6 | 0.42/0.13/0.45 | 0.806 | 82/12/6% | 3/10/87% | — | — |
| Nemotron rewrite | 1,864 | 62.1 | 74.2 | 52.6 / 71.5 | 0.31/0.24/0.45 | 0.577 | 53/21/26% | 10/18/72% | 14.7% | 12.7% |
| LED lambda0 | 1,864 | 63.1 | 79.7 | 58.0 / 68.1 | 0.32/0.27/0.41 | 0.592 | 58/24/18% | 7/25/68% | 18.6% | 7.0% |
| LED lambda035 | 1,864 | 63.1 | 78.6 | 56.5 / 69.7 | 0.32/0.27/0.41 | 0.586 | 56/25/19% | 8/22/70% | 18.1% | 7.7% |
| LED first_token | 1,864 | 64.7 | 79.8 | 57.4 / 71.9 | 0.32/0.26/0.43 | 0.595 | 57/24/18% | 7/21/72% | 17.3% | 7.0% |
