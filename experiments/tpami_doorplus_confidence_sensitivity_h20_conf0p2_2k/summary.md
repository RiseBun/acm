# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7, 42, 123], horizon=20, max_val_samples=2000, fp_confidence=0.2, confidence_mode=normal.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| doorplus_uncertainty | relfp0.2 | 46.730 +/- 20.326 | 0.958 +/- 0.016 | 0.909 +/- 0.019 | 6.984 +/- 0.270 | 0.008 +/- 0.002 | 0.558 | 0.083 +/- 0.013 | 0.139 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `Selected-TrueRisk-Rel@K` is the selected non-FP high-risk relation count divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
