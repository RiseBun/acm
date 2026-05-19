# DOOR+ post-hoc uncertainty gating

Evaluation-only oracle-confidence probe. `doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination.

Setup: seeds [7], horizon=10, max_val_samples=500, fp_confidence=0.05.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| door | clean | 34.837 +/- 0.000 | 0.288 +/- 0.000 | 0.294 +/- 0.000 | 7.091 +/- 0.000 | 0.000 +/- 0.000 | 0.000 |
| door | relfp0.2 | 33.592 +/- 0.000 | 0.584 +/- 0.000 | 0.579 +/- 0.000 | 7.421 +/- 0.000 | 0.188 +/- 0.000 | 0.570 |
| doorplus_posthoc | clean | 34.818 +/- 0.000 | 0.288 +/- 0.000 | 0.294 +/- 0.000 | 7.091 +/- 0.000 | 0.000 +/- 0.000 | 0.000 |
| doorplus_posthoc | relfp0.2 | 34.940 +/- 0.000 | 0.278 +/- 0.000 | 0.281 +/- 0.000 | 7.092 +/- 0.000 | 0.101 +/- 0.000 | 0.573 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- This is a post-hoc oracle-confidence probe, not a detector-integrated or retrained DOOR+ result.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
