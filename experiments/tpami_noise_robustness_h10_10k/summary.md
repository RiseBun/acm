# TPAMI perception-noise robustness probe

Evaluation-time input corruption on trained nuPlan Stage1 checkpoints; no retraining.

Setup: seeds [7, 42, 123], horizon=10, max_val_samples=10000.

| condition | corruption | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_object | clean | 8.863 +/- 0.370 | 3.553 +/- 0.118 | 1.922 +/- 41.820 | 0.676 +/- 0.095 | 0.673 +/- 0.062 | 0.276 |
| wm_object | loc1.5 | 8.864 +/- 0.369 | 3.554 +/- 0.117 | 1.932 +/- 41.850 | 0.675 +/- 0.095 | 0.673 +/- 0.062 | 0.276 |
| wm_object | miss0.2 | 8.880 +/- 0.391 | 3.560 +/- 0.126 | 2.116 +/- 41.940 | 0.640 +/- 0.087 | 0.644 +/- 0.059 | 0.280 |
| wm_object | relfp0.2 | 8.863 +/- 0.370 | 3.553 +/- 0.118 | 1.918 +/- 41.833 | 0.676 +/- 0.096 | 0.673 +/- 0.063 | 0.276 |
| wm_naive | clean | 8.912 +/- 0.245 | 3.562 +/- 0.062 | -5.998 +/- 11.006 | 0.415 +/- 0.521 | 0.450 +/- 0.239 | 0.102 |
| wm_naive | loc1.5 | 8.911 +/- 0.244 | 3.562 +/- 0.061 | -5.996 +/- 11.012 | 0.416 +/- 0.521 | 0.451 +/- 0.238 | 0.102 |
| wm_naive | miss0.2 | 8.919 +/- 0.253 | 3.564 +/- 0.064 | -6.012 +/- 11.016 | 0.407 +/- 0.525 | 0.444 +/- 0.245 | 0.102 |
| wm_naive | relfp0.2 | 8.921 +/- 0.259 | 3.565 +/- 0.066 | -6.012 +/- 10.996 | 0.416 +/- 0.521 | 0.453 +/- 0.236 | 0.102 |
| wm_decoupled_no_vis | clean | 6.628 +/- 0.110 | 2.115 +/- 0.083 | 33.756 +/- 7.213 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 0.111 |
| wm_decoupled_no_vis | loc1.5 | 6.628 +/- 0.110 | 2.115 +/- 0.083 | 33.739 +/- 7.226 | 0.262 +/- 0.046 | 0.280 +/- 0.032 | 0.111 |
| wm_decoupled_no_vis | miss0.2 | 6.633 +/- 0.116 | 2.117 +/- 0.084 | 30.567 +/- 5.604 | 0.249 +/- 0.034 | 0.263 +/- 0.024 | 0.089 |
| wm_decoupled_no_vis | relfp0.2 | 6.726 +/- 0.251 | 2.258 +/- 0.173 | 32.831 +/- 6.923 | 0.459 +/- 0.132 | 0.471 +/- 0.122 | 0.111 |

Reading:

- `locX` adds Gaussian localization noise with std `X` to dynamic and relation x/y features.
- `missR` randomly masks dynamic and relation tokens with probability `R`.
- `relfpR` turns a fraction `R` of existing relation tokens into high-risk false-positive relations.
- This probe measures evaluation-time robustness only; it is not a noisy-perception training result.
