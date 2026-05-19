# TPAMI perception-noise robustness probe

Evaluation-time input corruption on trained nuPlan Stage1 checkpoints; no retraining.

Setup: seeds [7], horizon=10, max_val_samples=500.

| condition | corruption | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_object | clean | 9.769 +/- 0.000 | 3.734 +/- 0.000 | 4.439 +/- 0.000 | 0.640 +/- 0.000 | 0.648 +/- 0.000 | 0.106 |
| wm_object | loc1.5 | 9.770 +/- 0.000 | 3.734 +/- 0.000 | 4.471 +/- 0.000 | 0.646 +/- 0.000 | 0.654 +/- 0.000 | 0.105 |
| wm_object | miss0.2 | 9.813 +/- 0.000 | 3.750 +/- 0.000 | 4.539 +/- 0.000 | 0.644 +/- 0.000 | 0.643 +/- 0.000 | 0.105 |
| wm_object | relfp0.2 | 9.769 +/- 0.000 | 3.734 +/- 0.000 | 4.476 +/- 0.000 | 0.640 +/- 0.000 | 0.648 +/- 0.000 | 0.105 |
| wm_naive | clean | 9.704 +/- 0.000 | 3.686 +/- 0.000 | 5.173 +/- 0.000 | 0.000 +/- 0.000 | 0.333 +/- 0.000 | 0.016 |
| wm_naive | loc1.5 | 9.704 +/- 0.000 | 3.687 +/- 0.000 | 5.238 +/- 0.000 | 0.000 +/- 0.000 | 0.333 +/- 0.000 | 0.016 |
| wm_naive | miss0.2 | 9.730 +/- 0.000 | 3.694 +/- 0.000 | 5.196 +/- 0.000 | 0.000 +/- 0.000 | 0.333 +/- 0.000 | 0.016 |
| wm_naive | relfp0.2 | 9.734 +/- 0.000 | 3.696 +/- 0.000 | 5.162 +/- 0.000 | 0.000 +/- 0.000 | 0.333 +/- 0.000 | 0.017 |
| wm_decoupled_no_vis | clean | 7.091 +/- 0.000 | 2.049 +/- 0.000 | 34.907 +/- 0.000 | 0.288 +/- 0.000 | 0.294 +/- 0.000 | 0.114 |
| wm_decoupled_no_vis | loc1.5 | 7.091 +/- 0.000 | 2.049 +/- 0.000 | 34.845 +/- 0.000 | 0.288 +/- 0.000 | 0.295 +/- 0.000 | 0.114 |
| wm_decoupled_no_vis | miss0.2 | 7.098 +/- 0.000 | 2.056 +/- 0.000 | 31.055 +/- 0.000 | 0.258 +/- 0.000 | 0.263 +/- 0.000 | 0.092 |
| wm_decoupled_no_vis | relfp0.2 | 7.337 +/- 0.000 | 2.397 +/- 0.000 | 33.631 +/- 0.000 | 0.582 +/- 0.000 | 0.574 +/- 0.000 | 0.114 |

Reading:

- `locX` adds Gaussian localization noise with std `X` to dynamic and relation x/y features.
- `missR` randomly masks dynamic and relation tokens with probability `R`.
- `relfpR` turns a fraction `R` of existing relation tokens into high-risk false-positive relations.
- This probe measures evaluation-time robustness only; it is not a noisy-perception training result.
