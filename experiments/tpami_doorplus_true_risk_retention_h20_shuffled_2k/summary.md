# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7, 42, 123], horizon=20, max_val_samples=2000, fp_confidence=0.05, confidence_mode=shuffled.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| doorplus_uncertainty | relfp0.2 | 46.617 +/- 20.537 | 0.969 +/- 0.009 | 0.924 +/- 0.013 | 7.023 +/- 0.233 | 0.123 +/- 0.030 | 0.557 | 0.067 +/- 0.007 | 0.137 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `Selected-TrueRisk-Rel@K` is the selected non-FP high-risk relation count divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
