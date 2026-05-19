# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7], horizon=10, max_val_samples=1000, fp_confidence=0.05, confidence_mode=constant.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| door | clean | -1.466 +/- 0.000 | 0.457 +/- 0.000 | 0.485 +/- 0.000 | 7.109 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 0.014 +/- 0.000 | 0.162 |
| door | relfp0.2 | -5.238 +/- 0.000 | 0.714 +/- 0.000 | 0.719 +/- 0.000 | 7.146 +/- 0.000 | 0.065 +/- 0.000 | 0.574 | 0.015 +/- 0.000 | 0.128 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `Selected-TrueRisk-Rel@K` is the selected non-FP high-risk relation count divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
