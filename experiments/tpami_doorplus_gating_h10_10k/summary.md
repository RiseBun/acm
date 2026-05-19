# DOOR+ post-hoc uncertainty gating

Evaluation-only oracle-confidence probe. `doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination.

Setup: seeds [7, 42, 123], horizon=10, max_val_samples=10000, fp_confidence=0.05.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| door | clean | 33.758 +/- 7.204 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| door | loc1.5 | 33.735 +/- 7.215 | 0.262 +/- 0.045 | 0.280 +/- 0.033 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| door | miss0.2 | 30.567 +/- 5.529 | 0.247 +/- 0.033 | 0.260 +/- 0.023 | 6.633 +/- 0.115 | 0.000 +/- 0.000 | 0.000 |
| door | relfp0.2 | 32.847 +/- 6.921 | 0.457 +/- 0.126 | 0.469 +/- 0.116 | 6.727 +/- 0.250 | 0.150 +/- 0.074 | 0.561 |
| doorplus_posthoc | clean | 33.749 +/- 7.212 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| doorplus_posthoc | loc1.5 | 33.737 +/- 7.213 | 0.263 +/- 0.045 | 0.280 +/- 0.032 | 6.628 +/- 0.110 | 0.000 +/- 0.000 | 0.000 |
| doorplus_posthoc | miss0.2 | 30.537 +/- 5.578 | 0.246 +/- 0.031 | 0.260 +/- 0.022 | 6.633 +/- 0.116 | 0.000 +/- 0.000 | 0.000 |
| doorplus_posthoc | relfp0.2 | 33.793 +/- 7.236 | 0.242 +/- 0.046 | 0.261 +/- 0.032 | 6.627 +/- 0.110 | 0.111 +/- 0.018 | 0.558 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- This is a post-hoc oracle-confidence probe, not a detector-integrated or retrained DOOR+ result.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
