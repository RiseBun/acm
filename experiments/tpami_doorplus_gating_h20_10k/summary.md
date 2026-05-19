# DOOR+ post-hoc uncertainty gating

Evaluation-only oracle-confidence probe. `doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination.

Setup: seeds [7, 42, 123], horizon=20, max_val_samples=10000, fp_confidence=0.05.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| door | clean | 72.250 +/- 16.016 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| door | relfp0.2 | 71.261 +/- 15.668 | 0.458 +/- 0.125 | 0.469 +/- 0.116 | 6.727 +/- 0.250 | 0.150 +/- 0.074 | 0.561 |
| doorplus_posthoc | clean | 72.243 +/- 15.982 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| doorplus_posthoc | relfp0.2 | 72.284 +/- 16.046 | 0.242 +/- 0.046 | 0.261 +/- 0.032 | 6.627 +/- 0.110 | 0.111 +/- 0.018 | 0.558 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- This is a post-hoc oracle-confidence probe, not a detector-integrated or retrained DOOR+ result.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
