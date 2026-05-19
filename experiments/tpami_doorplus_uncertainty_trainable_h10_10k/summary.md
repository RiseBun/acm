# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7, 42, 123], horizon=10, max_val_samples=10000, fp_confidence=0.05.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| doorplus_uncertainty | clean | 20.626 +/- 8.013 | 0.953 +/- 0.013 | 0.905 +/- 0.015 | 6.839 +/- 0.243 | 0.000 +/- 0.000 | 0.000 |
| doorplus_uncertainty | relfp0.2 | 20.669 +/- 8.100 | 0.967 +/- 0.008 | 0.916 +/- 0.014 | 6.815 +/- 0.226 | 0.007 +/- 0.000 | 0.556 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
