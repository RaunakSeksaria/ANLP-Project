# Variant comparison: think_on_t1.0, think_on_t1.0_rep, think_off_t1.0, think_on_t0.0, think_on_t0.3, think_on_t0.6, think_on_t1.3, think_on_t1.0_p0.8, think_on_t1.0_p1.0, para_t1.0, n3_t1.0, n2light_t1.0, n2heavy_t1.0, n2min_t1.0

Paired on events every variant finished. Reference for differences: `think_on_t1.0`.

## left (n = 150 paired events)

| metric | `think_on_t1.0` | `think_on_t1.0_rep` | `think_off_t1.0` | `think_on_t0.0` | `think_on_t0.3` | `think_on_t0.6` | `think_on_t1.3` | `think_on_t1.0_p0.8` | `think_on_t1.0_p1.0` | `para_t1.0` | `n3_t1.0` | `n2light_t1.0` | `n2heavy_t1.0` | `n2min_t1.0` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| out tokens/article (API) | 2432.00 | 2256.00 | 696.00 | 2515.00 | 2425.50 | 2491.00 | 2458.00 | 2421.00 | 2507.50 | 3733.50 | 2582.50 | 2868.00 | 2340.50 | 4250.00 |
| wall min / 100 articles | 48.38 | 56.08 | 33.29 | 48.37 | 38.31 | 44.37 | 47.48 | 44.60 | 49.77 | 48.37 | 43.42 | 48.57 | 44.32 | 172.31 |
| similarity to input | 0.79 | 0.78 | 0.85 | 0.79 | 0.80 | 0.79 | 0.77 | 0.78 | 0.78 | 0.68 | 0.77 | 0.90 | 0.69 | 0.79 |
| input words changed % | 25.99 | 26.19 | 19.15 | 25.05 | 23.99 | 24.98 | 27.56 | 26.43 | 25.96 | 32.80 | 27.26 | 12.42 | 36.86 | 24.69 |
| output/input words | 0.90 | 0.91 | 0.94 | 0.89 | 0.90 | 0.89 | 0.87 | 0.90 | 0.91 | 0.97 | 0.89 | 0.97 | 0.84 | 0.93 |
| entities kept % | 94.47 | 94.02 | 94.88 | 94.43 | 94.85 | 94.41 | 94.05 | 94.52 | 93.94 | 95.95 | 94.28 | 96.81 | 92.56 | 94.49 |
| quotes verbatim % | 99.08 | 98.66 | 93.03 | 99.09 | 98.94 | 98.52 | 99.62 | 99.10 | 99.12 | 93.93 | 97.41 | 98.62 | 98.42 | 99.54 |
| quotes reworded+invented % | 0.92 | 1.34 | 6.97 | 0.91 | 1.06 | 1.48 | 0.38 | 0.90 | 0.88 | 6.07 | 2.59 | 1.38 | 1.58 | 0.46 |
| adds entity % | 26.67 | 30.67 | 24.00 | 24.67 | 26.67 | 30.67 | 27.33 | 25.33 | 28.00 | 56.00 | 24.00 | 18.00 | 40.00 | 34.67 |
| adds number % | 3.33 | 0.67 | 1.33 | 1.33 | 2.00 | 2.00 | 3.33 | 2.00 | 2.00 | 0.67 | 2.67 | 0.67 | 4.00 | 3.33 |
| adds date % | 4.00 | 2.00 | 3.33 | 2.67 | 4.00 | 2.00 | 2.67 | 3.33 | 3.33 | 0.67 | 2.67 | 0.67 | 4.67 | 4.67 |
| 'the article ...' % | 13.33 | 11.33 | 14.00 | 12.67 | 17.33 | 12.67 | 12.67 | 13.33 | 10.67 | 2.00 | 9.33 | 6.00 | 12.00 | 3.33 |
| loaded words /1k (in -> out) | 1.69 -> 0.77 | 1.69 -> 0.82 | 1.69 -> 1.20 | 1.69 -> 0.99 | 1.69 -> 0.86 | 1.69 -> 1.05 | 1.69 -> 0.90 | 1.69 -> 0.91 | 1.69 -> 0.85 | 1.69 -> 2.22 | 1.69 -> 0.81 | 1.69 -> 0.83 | 1.69 -> 0.74 | 1.69 -> 0.73 |
| output < 512 LED tokens % | 56.00 | 54.00 | 50.00 | 52.00 | 54.67 | 54.00 | 52.00 | 58.00 | 53.33 | 40.00 | 56.67 | 43.33 | 60.00 | 50.67 |
| POLITICS P(true side) orig -> out | 0.713 -> 0.372 | 0.713 -> 0.366 | 0.713 -> 0.405 | 0.713 -> 0.393 | 0.713 -> 0.378 | 0.713 -> 0.364 | 0.713 -> 0.370 | 0.713 -> 0.399 | 0.713 -> 0.382 | 0.713 -> 0.519 | 0.713 -> 0.392 | 0.713 -> 0.477 | 0.713 -> 0.310 | 0.713 -> 0.494 |
| POLITICS true side on top % orig -> out | 75.3 -> 36.7 | 75.3 -> 35.3 | 75.3 -> 40.0 | 75.3 -> 41.3 | 75.3 -> 38.0 | 75.3 -> 36.7 | 75.3 -> 36.0 | 75.3 -> 40.0 | 75.3 -> 39.3 | 75.3 -> 53.3 | 75.3 -> 38.0 | 75.3 -> 45.3 | 75.3 -> 28.0 | 75.3 -> 50.7 |
| POLITICS P(Centre) orig -> out | 0.171 -> 0.469 | 0.171 -> 0.472 | 0.171 -> 0.426 | 0.171 -> 0.460 | 0.171 -> 0.464 | 0.171 -> 0.478 | 0.171 -> 0.463 | 0.171 -> 0.420 | 0.171 -> 0.459 | 0.171 -> 0.225 | 0.171 -> 0.458 | 0.171 -> 0.392 | 0.171 -> 0.545 | 0.171 -> 0.389 |

**Paired differences vs `think_on_t1.0`** (mean, bootstrap 95% CI):

| variant | metric | diff | 95% CI |
| --- | --- | --- | --- |
| `think_on_t1.0_rep` | similarity to input | -0.005 | [-0.024, +0.017] |
| `think_on_t1.0_rep` | input words changed (pts) | +0.201 | [-2.091, +2.266] |
| `think_on_t1.0_rep` | entities kept (pts) | -0.454 | [-1.468, +0.578] |
| `think_on_t1.0_rep` | 'the article ...' (pts) | -2.000 | [-7.333, +4.000] |
| `think_on_t1.0_rep` | adds entity (pts) | +4.000 | [-2.667, +10.667] |
| `think_on_t1.0_rep` | adds date (pts) | -2.000 | [-4.667, +0.000] |
| `think_on_t1.0_rep` | POLITICS P(true side) on output | -0.007 | [-0.058, +0.048] |
| `think_on_t1.0_rep` | POLITICS P(Centre) on output | +0.003 | [-0.055, +0.057] |
| `think_off_t1.0` | similarity to input | +0.061 | [+0.038, +0.083] |
| `think_off_t1.0` | input words changed (pts) | -6.839 | [-9.463, -4.171] |
| `think_off_t1.0` | entities kept (pts) | +0.406 | [-0.762, +1.532] |
| `think_off_t1.0` | 'the article ...' (pts) | +0.667 | [-5.333, +6.667] |
| `think_off_t1.0` | adds entity (pts) | -2.667 | [-9.333, +4.000] |
| `think_off_t1.0` | adds date (pts) | -0.667 | [-3.333, +2.000] |
| `think_off_t1.0` | POLITICS P(true side) on output | +0.032 | [-0.022, +0.090] |
| `think_off_t1.0` | POLITICS P(Centre) on output | -0.043 | [-0.102, +0.016] |
| `think_on_t0.0` | similarity to input | +0.009 | [-0.012, +0.032] |
| `think_on_t0.0` | input words changed (pts) | -0.935 | [-3.551, +1.453] |
| `think_on_t0.0` | entities kept (pts) | -0.038 | [-0.878, +0.896] |
| `think_on_t0.0` | 'the article ...' (pts) | -0.667 | [-5.333, +4.667] |
| `think_on_t0.0` | adds entity (pts) | -2.000 | [-8.000, +3.333] |
| `think_on_t0.0` | adds date (pts) | -1.333 | [-4.667, +1.333] |
| `think_on_t0.0` | POLITICS P(true side) on output | +0.021 | [-0.025, +0.068] |
| `think_on_t0.0` | POLITICS P(Centre) on output | -0.009 | [-0.057, +0.038] |
| `think_on_t0.3` | similarity to input | +0.018 | [-0.003, +0.043] |
| `think_on_t0.3` | input words changed (pts) | -1.996 | [-4.582, +0.450] |
| `think_on_t0.3` | entities kept (pts) | +0.374 | [-0.674, +1.512] |
| `think_on_t0.3` | 'the article ...' (pts) | +4.000 | [-1.333, +10.000] |
| `think_on_t0.3` | adds entity (pts) | +0.000 | [-6.000, +6.000] |
| `think_on_t0.3` | adds date (pts) | +0.000 | [-2.000, +2.000] |
| `think_on_t0.3` | POLITICS P(true side) on output | +0.005 | [-0.043, +0.057] |
| `think_on_t0.3` | POLITICS P(Centre) on output | -0.005 | [-0.058, +0.046] |
| `think_on_t0.6` | similarity to input | +0.007 | [-0.014, +0.029] |
| `think_on_t0.6` | input words changed (pts) | -1.011 | [-3.534, +1.558] |
| `think_on_t0.6` | entities kept (pts) | -0.061 | [-1.079, +0.977] |
| `think_on_t0.6` | 'the article ...' (pts) | -0.667 | [-6.667, +5.333] |
| `think_on_t0.6` | adds entity (pts) | +4.000 | [-2.667, +10.000] |
| `think_on_t0.6` | adds date (pts) | -2.000 | [-5.333, +0.667] |
| `think_on_t0.6` | POLITICS P(true side) on output | -0.008 | [-0.057, +0.039] |
| `think_on_t0.6` | POLITICS P(Centre) on output | +0.009 | [-0.043, +0.061] |
| `think_on_t1.3` | similarity to input | -0.010 | [-0.031, +0.011] |
| `think_on_t1.3` | input words changed (pts) | +1.576 | [-0.851, +3.984] |
| `think_on_t1.3` | entities kept (pts) | -0.419 | [-1.395, +0.545] |
| `think_on_t1.3` | 'the article ...' (pts) | -0.667 | [-6.667, +5.333] |
| `think_on_t1.3` | adds entity (pts) | +0.667 | [-5.333, +6.667] |
| `think_on_t1.3` | adds date (pts) | -1.333 | [-4.000, +1.333] |
| `think_on_t1.3` | POLITICS P(true side) on output | -0.002 | [-0.053, +0.048] |
| `think_on_t1.3` | POLITICS P(Centre) on output | -0.005 | [-0.060, +0.050] |
| `think_on_t1.0_p0.8` | similarity to input | -0.004 | [-0.027, +0.019] |
| `think_on_t1.0_p0.8` | input words changed (pts) | +0.448 | [-2.198, +3.012] |
| `think_on_t1.0_p0.8` | entities kept (pts) | +0.045 | [-0.897, +1.058] |
| `think_on_t1.0_p0.8` | 'the article ...' (pts) | +0.000 | [-6.667, +6.667] |
| `think_on_t1.0_p0.8` | adds entity (pts) | -1.333 | [-8.000, +4.667] |
| `think_on_t1.0_p0.8` | adds date (pts) | -0.667 | [-3.333, +1.333] |
| `think_on_t1.0_p0.8` | POLITICS P(true side) on output | +0.026 | [-0.025, +0.077] |
| `think_on_t1.0_p0.8` | POLITICS P(Centre) on output | -0.049 | [-0.102, +0.005] |
| `think_on_t1.0_p1.0` | similarity to input | -0.003 | [-0.026, +0.021] |
| `think_on_t1.0_p1.0` | input words changed (pts) | -0.024 | [-2.656, +2.653] |
| `think_on_t1.0_p1.0` | entities kept (pts) | -0.527 | [-1.598, +0.629] |
| `think_on_t1.0_p1.0` | 'the article ...' (pts) | -2.667 | [-8.000, +2.667] |
| `think_on_t1.0_p1.0` | adds entity (pts) | +1.333 | [-5.333, +8.667] |
| `think_on_t1.0_p1.0` | adds date (pts) | -0.667 | [-4.000, +2.667] |
| `think_on_t1.0_p1.0` | POLITICS P(true side) on output | +0.009 | [-0.036, +0.055] |
| `think_on_t1.0_p1.0` | POLITICS P(Centre) on output | -0.010 | [-0.065, +0.043] |
| `para_t1.0` | similarity to input | -0.103 | [-0.129, -0.077] |
| `para_t1.0` | input words changed (pts) | +6.814 | [+3.848, +9.793] |
| `para_t1.0` | entities kept (pts) | +1.479 | [+0.223, +2.850] |
| `para_t1.0` | 'the article ...' (pts) | -11.333 | [-17.333, -6.000] |
| `para_t1.0` | adds entity (pts) | +29.333 | [+20.667, +38.000] |
| `para_t1.0` | adds date (pts) | -3.333 | [-6.667, -0.667] |
| `para_t1.0` | POLITICS P(true side) on output | +0.147 | [+0.069, +0.220] |
| `para_t1.0` | POLITICS P(Centre) on output | -0.244 | [-0.308, -0.178] |
| `n3_t1.0` | similarity to input | -0.010 | [-0.032, +0.011] |
| `n3_t1.0` | input words changed (pts) | +1.278 | [-1.042, +3.573] |
| `n3_t1.0` | entities kept (pts) | -0.190 | [-1.161, +0.877] |
| `n3_t1.0` | 'the article ...' (pts) | -4.000 | [-10.000, +2.000] |
| `n3_t1.0` | adds entity (pts) | -2.667 | [-9.333, +3.333] |
| `n3_t1.0` | adds date (pts) | -1.333 | [-3.333, +0.000] |
| `n3_t1.0` | POLITICS P(true side) on output | +0.019 | [-0.029, +0.069] |
| `n3_t1.0` | POLITICS P(Centre) on output | -0.011 | [-0.059, +0.037] |
| `n2light_t1.0` | similarity to input | +0.113 | [+0.091, +0.135] |
| `n2light_t1.0` | input words changed (pts) | -13.568 | [-16.058, -11.083] |
| `n2light_t1.0` | entities kept (pts) | +2.341 | [+1.350, +3.414] |
| `n2light_t1.0` | 'the article ...' (pts) | -7.333 | [-12.667, -2.000] |
| `n2light_t1.0` | adds entity (pts) | -8.667 | [-14.667, -2.667] |
| `n2light_t1.0` | adds date (pts) | -3.333 | [-7.333, +0.000] |
| `n2light_t1.0` | POLITICS P(true side) on output | +0.105 | [+0.046, +0.160] |
| `n2light_t1.0` | POLITICS P(Centre) on output | -0.077 | [-0.135, -0.021] |
| `n2heavy_t1.0` | similarity to input | -0.100 | [-0.127, -0.074] |
| `n2heavy_t1.0` | input words changed (pts) | +10.868 | [+8.088, +13.759] |
| `n2heavy_t1.0` | entities kept (pts) | -1.911 | [-2.969, -0.884] |
| `n2heavy_t1.0` | 'the article ...' (pts) | -1.333 | [-8.000, +4.667] |
| `n2heavy_t1.0` | adds entity (pts) | +13.333 | [+6.667, +20.017] |
| `n2heavy_t1.0` | adds date (pts) | +0.667 | [-2.000, +4.000] |
| `n2heavy_t1.0` | POLITICS P(true side) on output | -0.062 | [-0.118, -0.008] |
| `n2heavy_t1.0` | POLITICS P(Centre) on output | +0.076 | [+0.019, +0.134] |
| `n2min_t1.0` | similarity to input | +0.008 | [-0.016, +0.033] |
| `n2min_t1.0` | input words changed (pts) | -1.299 | [-4.060, +1.408] |
| `n2min_t1.0` | entities kept (pts) | +0.024 | [-1.020, +1.131] |
| `n2min_t1.0` | 'the article ...' (pts) | -10.000 | [-15.350, -4.667] |
| `n2min_t1.0` | adds entity (pts) | +8.000 | [+0.667, +15.333] |
| `n2min_t1.0` | adds date (pts) | +0.667 | [-2.000, +3.333] |
| `n2min_t1.0` | POLITICS P(true side) on output | +0.122 | [+0.064, +0.186] |
| `n2min_t1.0` | POLITICS P(Centre) on output | -0.080 | [-0.147, -0.016] |

## right (n = 150 paired events)

| metric | `think_on_t1.0` | `think_on_t1.0_rep` | `think_off_t1.0` | `think_on_t0.0` | `think_on_t0.3` | `think_on_t0.6` | `think_on_t1.3` | `think_on_t1.0_p0.8` | `think_on_t1.0_p1.0` | `para_t1.0` | `n3_t1.0` | `n2light_t1.0` | `n2heavy_t1.0` | `n2min_t1.0` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| out tokens/article (API) | 2400.50 | 2379.00 | 683.50 | 2564.50 | 2675.00 | 2408.50 | 2445.50 | 2386.50 | 2289.50 | 3849.50 | 2627.50 | 2919.00 | 2433.50 | 4942.50 |
| wall min / 100 articles | 47.17 | 49.39 | 23.77 | 44.98 | 52.20 | 48.60 | 51.29 | 38.20 | 42.23 | 58.70 | 46.82 | 41.21 | 48.09 | 166.41 |
| similarity to input | 0.80 | 0.80 | 0.85 | 0.80 | 0.80 | 0.79 | 0.78 | 0.80 | 0.79 | 0.69 | 0.79 | 0.92 | 0.71 | 0.84 |
| input words changed % | 24.84 | 24.84 | 18.50 | 24.44 | 24.51 | 25.67 | 26.41 | 24.78 | 25.35 | 32.24 | 26.03 | 10.91 | 34.01 | 18.97 |
| output/input words | 0.89 | 0.89 | 0.93 | 0.89 | 0.90 | 0.89 | 0.89 | 0.89 | 0.90 | 0.97 | 0.89 | 0.96 | 0.86 | 0.94 |
| entities kept % | 92.46 | 91.26 | 92.84 | 92.68 | 92.20 | 91.45 | 91.81 | 91.84 | 91.83 | 94.24 | 91.31 | 94.12 | 90.65 | 94.37 |
| quotes verbatim % | 98.85 | 99.08 | 98.49 | 98.81 | 97.81 | 98.22 | 98.84 | 98.80 | 94.15 | 91.50 | 98.73 | 97.65 | 96.43 | 98.90 |
| quotes reworded+invented % | 1.15 | 0.92 | 1.51 | 1.19 | 2.19 | 1.78 | 1.16 | 1.20 | 5.85 | 8.50 | 1.27 | 2.35 | 3.57 | 1.10 |
| adds entity % | 22.67 | 28.00 | 20.00 | 26.00 | 24.67 | 28.67 | 26.00 | 24.00 | 26.00 | 59.33 | 27.33 | 13.33 | 34.00 | 26.67 |
| adds number % | 2.67 | 2.67 | 2.67 | 4.67 | 4.67 | 2.67 | 4.00 | 4.67 | 2.67 | 0.00 | 3.33 | 2.00 | 7.33 | 2.67 |
| adds date % | 0.00 | 1.33 | 0.67 | 2.67 | 2.67 | 0.67 | 2.67 | 2.00 | 1.33 | 0.00 | 1.33 | 0.67 | 2.67 | 0.67 |
| 'the article ...' % | 7.33 | 6.00 | 6.67 | 4.67 | 6.00 | 6.00 | 8.00 | 5.33 | 7.33 | 1.33 | 6.67 | 3.33 | 8.00 | 1.33 |
| loaded words /1k (in -> out) | 1.27 -> 0.57 | 1.27 -> 0.67 | 1.27 -> 0.89 | 1.27 -> 0.62 | 1.27 -> 0.53 | 1.27 -> 0.59 | 1.27 -> 0.80 | 1.27 -> 0.57 | 1.27 -> 0.50 | 1.27 -> 1.99 | 1.27 -> 0.60 | 1.27 -> 0.62 | 1.27 -> 0.61 | 1.27 -> 0.58 |
| output < 512 LED tokens % | 51.33 | 55.33 | 48.67 | 53.33 | 52.67 | 54.67 | 50.67 | 52.67 | 50.00 | 39.33 | 53.33 | 42.00 | 55.33 | 44.00 |
| POLITICS P(true side) orig -> out | 0.723 -> 0.453 | 0.723 -> 0.455 | 0.723 -> 0.486 | 0.723 -> 0.483 | 0.723 -> 0.443 | 0.723 -> 0.444 | 0.723 -> 0.464 | 0.723 -> 0.459 | 0.723 -> 0.453 | 0.723 -> 0.658 | 0.723 -> 0.468 | 0.723 -> 0.521 | 0.723 -> 0.429 | 0.723 -> 0.534 |
| POLITICS true side on top % orig -> out | 74.0 -> 44.7 | 74.0 -> 48.7 | 74.0 -> 46.0 | 74.0 -> 48.7 | 74.0 -> 41.3 | 74.0 -> 41.3 | 74.0 -> 47.3 | 74.0 -> 46.0 | 74.0 -> 48.0 | 74.0 -> 70.0 | 74.0 -> 48.0 | 74.0 -> 52.7 | 74.0 -> 44.7 | 74.0 -> 54.0 |
| POLITICS P(Centre) orig -> out | 0.211 -> 0.439 | 0.211 -> 0.449 | 0.211 -> 0.398 | 0.211 -> 0.401 | 0.211 -> 0.443 | 0.211 -> 0.465 | 0.211 -> 0.436 | 0.211 -> 0.446 | 0.211 -> 0.427 | 0.211 -> 0.196 | 0.211 -> 0.444 | 0.211 -> 0.370 | 0.211 -> 0.496 | 0.211 -> 0.344 |

**Paired differences vs `think_on_t1.0`** (mean, bootstrap 95% CI):

| variant | metric | diff | 95% CI |
| --- | --- | --- | --- |
| `think_on_t1.0_rep` | similarity to input | +0.001 | [-0.020, +0.021] |
| `think_on_t1.0_rep` | input words changed (pts) | +0.002 | [-2.234, +2.348] |
| `think_on_t1.0_rep` | entities kept (pts) | -1.208 | [-2.374, -0.169] |
| `think_on_t1.0_rep` | 'the article ...' (pts) | -1.333 | [-6.667, +4.000] |
| `think_on_t1.0_rep` | adds entity (pts) | +5.333 | [-0.667, +12.000] |
| `think_on_t1.0_rep` | adds date (pts) | +1.333 | [+0.000, +3.333] |
| `think_on_t1.0_rep` | POLITICS P(true side) on output | +0.002 | [-0.059, +0.066] |
| `think_on_t1.0_rep` | POLITICS P(Centre) on output | +0.010 | [-0.046, +0.064] |
| `think_off_t1.0` | similarity to input | +0.058 | [+0.035, +0.080] |
| `think_off_t1.0` | input words changed (pts) | -6.342 | [-8.780, -3.836] |
| `think_off_t1.0` | entities kept (pts) | +0.374 | [-0.721, +1.449] |
| `think_off_t1.0` | 'the article ...' (pts) | -0.667 | [-6.017, +5.333] |
| `think_off_t1.0` | adds entity (pts) | -2.667 | [-8.000, +2.667] |
| `think_off_t1.0` | adds date (pts) | +0.667 | [+0.000, +2.000] |
| `think_off_t1.0` | POLITICS P(true side) on output | +0.033 | [-0.027, +0.093] |
| `think_off_t1.0` | POLITICS P(Centre) on output | -0.041 | [-0.102, +0.018] |
| `think_on_t0.0` | similarity to input | +0.004 | [-0.018, +0.027] |
| `think_on_t0.0` | input words changed (pts) | -0.395 | [-2.911, +2.027] |
| `think_on_t0.0` | entities kept (pts) | +0.217 | [-0.678, +1.079] |
| `think_on_t0.0` | 'the article ...' (pts) | -2.667 | [-7.333, +2.000] |
| `think_on_t0.0` | adds entity (pts) | +3.333 | [-4.667, +10.000] |
| `think_on_t0.0` | adds date (pts) | +2.667 | [+0.667, +5.333] |
| `think_on_t0.0` | POLITICS P(true side) on output | +0.030 | [-0.028, +0.090] |
| `think_on_t0.0` | POLITICS P(Centre) on output | -0.039 | [-0.092, +0.012] |
| `think_on_t0.3` | similarity to input | +0.003 | [-0.017, +0.023] |
| `think_on_t0.3` | input words changed (pts) | -0.326 | [-2.582, +2.053] |
| `think_on_t0.3` | entities kept (pts) | -0.268 | [-1.226, +0.642] |
| `think_on_t0.3` | 'the article ...' (pts) | -1.333 | [-6.667, +4.000] |
| `think_on_t0.3` | adds entity (pts) | +2.000 | [-4.000, +7.333] |
| `think_on_t0.3` | adds date (pts) | +2.667 | [+0.667, +5.333] |
| `think_on_t0.3` | POLITICS P(true side) on output | -0.010 | [-0.068, +0.047] |
| `think_on_t0.3` | POLITICS P(Centre) on output | +0.004 | [-0.050, +0.062] |
| `think_on_t0.6` | similarity to input | -0.007 | [-0.029, +0.015] |
| `think_on_t0.6` | input words changed (pts) | +0.834 | [-1.606, +3.294] |
| `think_on_t0.6` | entities kept (pts) | -1.010 | [-2.145, -0.040] |
| `think_on_t0.6` | 'the article ...' (pts) | -1.333 | [-7.333, +4.000] |
| `think_on_t0.6` | adds entity (pts) | +6.000 | [-0.667, +12.000] |
| `think_on_t0.6` | adds date (pts) | +0.667 | [+0.000, +2.000] |
| `think_on_t0.6` | POLITICS P(true side) on output | -0.009 | [-0.063, +0.046] |
| `think_on_t0.6` | POLITICS P(Centre) on output | +0.025 | [-0.027, +0.079] |
| `think_on_t1.3` | similarity to input | -0.015 | [-0.037, +0.005] |
| `think_on_t1.3` | input words changed (pts) | +1.572 | [-0.736, +4.016] |
| `think_on_t1.3` | entities kept (pts) | -0.656 | [-1.777, +0.370] |
| `think_on_t1.3` | 'the article ...' (pts) | +0.667 | [-5.333, +6.000] |
| `think_on_t1.3` | adds entity (pts) | +3.333 | [-2.667, +9.333] |
| `think_on_t1.3` | adds date (pts) | +2.667 | [+0.667, +5.333] |
| `think_on_t1.3` | POLITICS P(true side) on output | +0.011 | [-0.049, +0.075] |
| `think_on_t1.3` | POLITICS P(Centre) on output | -0.003 | [-0.060, +0.052] |
| `think_on_t1.0_p0.8` | similarity to input | +0.001 | [-0.020, +0.021] |
| `think_on_t1.0_p0.8` | input words changed (pts) | -0.064 | [-2.315, +2.380] |
| `think_on_t1.0_p0.8` | entities kept (pts) | -0.626 | [-1.743, +0.364] |
| `think_on_t1.0_p0.8` | 'the article ...' (pts) | -2.000 | [-6.667, +2.667] |
| `think_on_t1.0_p0.8` | adds entity (pts) | +1.333 | [-4.667, +7.333] |
| `think_on_t1.0_p0.8` | adds date (pts) | +2.000 | [+0.000, +4.667] |
| `think_on_t1.0_p0.8` | POLITICS P(true side) on output | +0.006 | [-0.052, +0.061] |
| `think_on_t1.0_p0.8` | POLITICS P(Centre) on output | +0.006 | [-0.046, +0.060] |
| `think_on_t1.0_p1.0` | similarity to input | -0.009 | [-0.034, +0.015] |
| `think_on_t1.0_p1.0` | input words changed (pts) | +0.510 | [-2.115, +3.368] |
| `think_on_t1.0_p1.0` | entities kept (pts) | -0.638 | [-1.737, +0.321] |
| `think_on_t1.0_p1.0` | 'the article ...' (pts) | +0.000 | [-5.333, +6.000] |
| `think_on_t1.0_p1.0` | adds entity (pts) | +3.333 | [-2.667, +8.667] |
| `think_on_t1.0_p1.0` | adds date (pts) | +1.333 | [+0.000, +3.333] |
| `think_on_t1.0_p1.0` | POLITICS P(true side) on output | -0.000 | [-0.061, +0.063] |
| `think_on_t1.0_p1.0` | POLITICS P(Centre) on output | -0.013 | [-0.067, +0.038] |
| `para_t1.0` | similarity to input | -0.110 | [-0.136, -0.084] |
| `para_t1.0` | input words changed (pts) | +7.402 | [+4.424, +10.344] |
| `para_t1.0` | entities kept (pts) | +1.777 | [+0.496, +2.942] |
| `para_t1.0` | 'the article ...' (pts) | -6.000 | [-10.667, -2.000] |
| `para_t1.0` | adds entity (pts) | +36.667 | [+28.000, +45.333] |
| `para_t1.0` | adds date (pts) | +0.000 | [+0.000, +0.000] |
| `para_t1.0` | POLITICS P(true side) on output | +0.205 | [+0.130, +0.281] |
| `para_t1.0` | POLITICS P(Centre) on output | -0.243 | [-0.310, -0.181] |
| `n3_t1.0` | similarity to input | -0.010 | [-0.030, +0.012] |
| `n3_t1.0` | input words changed (pts) | +1.190 | [-1.380, +3.619] |
| `n3_t1.0` | entities kept (pts) | -1.158 | [-2.397, -0.048] |
| `n3_t1.0` | 'the article ...' (pts) | -0.667 | [-6.000, +4.667] |
| `n3_t1.0` | adds entity (pts) | +4.667 | [-2.000, +11.333] |
| `n3_t1.0` | adds date (pts) | +1.333 | [+0.000, +3.333] |
| `n3_t1.0` | POLITICS P(true side) on output | +0.015 | [-0.040, +0.071] |
| `n3_t1.0` | POLITICS P(Centre) on output | +0.005 | [-0.048, +0.059] |
| `n2light_t1.0` | similarity to input | +0.121 | [+0.100, +0.141] |
| `n2light_t1.0` | input words changed (pts) | -13.934 | [-16.347, -11.471] |
| `n2light_t1.0` | entities kept (pts) | +1.660 | [+0.507, +2.692] |
| `n2light_t1.0` | 'the article ...' (pts) | -4.000 | [-8.667, +0.000] |
| `n2light_t1.0` | adds entity (pts) | -9.333 | [-15.333, -4.000] |
| `n2light_t1.0` | adds date (pts) | +0.667 | [+0.000, +2.000] |
| `n2light_t1.0` | POLITICS P(true side) on output | +0.068 | [+0.010, +0.131] |
| `n2light_t1.0` | POLITICS P(Centre) on output | -0.069 | [-0.128, -0.014] |
| `n2heavy_t1.0` | similarity to input | -0.084 | [-0.107, -0.061] |
| `n2heavy_t1.0` | input words changed (pts) | +9.175 | [+6.796, +11.667] |
| `n2heavy_t1.0` | entities kept (pts) | -1.811 | [-3.037, -0.722] |
| `n2heavy_t1.0` | 'the article ...' (pts) | +0.667 | [-4.667, +5.333] |
| `n2heavy_t1.0` | adds entity (pts) | +11.333 | [+4.667, +18.000] |
| `n2heavy_t1.0` | adds date (pts) | +2.667 | [+0.667, +5.333] |
| `n2heavy_t1.0` | POLITICS P(true side) on output | -0.024 | [-0.095, +0.045] |
| `n2heavy_t1.0` | POLITICS P(Centre) on output | +0.056 | [-0.015, +0.125] |
| `n2min_t1.0` | similarity to input | +0.049 | [+0.027, +0.069] |
| `n2min_t1.0` | input words changed (pts) | -5.870 | [-8.175, -3.415] |
| `n2min_t1.0` | entities kept (pts) | +1.907 | [+0.769, +2.986] |
| `n2min_t1.0` | 'the article ...' (pts) | -6.000 | [-10.667, -2.000] |
| `n2min_t1.0` | adds entity (pts) | +4.000 | [-2.667, +10.667] |
| `n2min_t1.0` | adds date (pts) | +0.667 | [+0.000, +2.000] |
| `n2min_t1.0` | POLITICS P(true side) on output | +0.081 | [+0.008, +0.151] |
| `n2min_t1.0` | POLITICS P(Centre) on output | -0.096 | [-0.159, -0.036] |
