# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7, 42, 123], horizon=20, max_val_samples=2000, fp_confidence=0.05, confidence_mode=normal.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| doorplus_uncertainty | relfp0.2 | 46.652 +/- 20.307 | 0.968 +/- 0.009 | 0.918 +/- 0.014 | 6.977 +/- 0.280 | 0.008 +/- 0.002 | 0.558 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
