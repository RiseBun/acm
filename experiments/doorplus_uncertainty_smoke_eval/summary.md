# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7], horizon=2, max_val_samples=100, fp_confidence=0.05.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K |
|---|---|---:|---:|---:|---:|---:|---:|
| doorplus_uncertainty | clean | -0.634 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 9.398 +/- 0.000 | 0.000 +/- 0.000 | 0.000 |
| doorplus_uncertainty | relfp0.2 | -0.638 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 9.399 +/- 0.000 | 0.000 +/- 0.000 | 0.538 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
