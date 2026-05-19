# DOOR+ uncertainty gating

`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. `doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.

Setup: seeds [7, 42, 123], horizon=10, max_val_samples=1000, fp_confidence=0.05, confidence_mode=proxy.

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| door | clean | 33.453 +/- 7.031 | 0.287 +/- 0.053 | 0.307 +/- 0.041 | 6.826 +/- 0.211 | 0.000 +/- 0.000 | 0.000 | 0.088 +/- 0.025 | 0.165 |
| door | relfp0.2 | 32.587 +/- 6.821 | 0.475 +/- 0.128 | 0.487 +/- 0.120 | 6.976 +/- 0.327 | 0.139 +/- 0.066 | 0.566 | 0.065 +/- 0.014 | 0.132 |
| doorplus_uncertainty | clean | 33.275 +/- 6.607 | 0.308 +/- 0.064 | 0.325 +/- 0.058 | 6.840 +/- 0.219 | 0.000 +/- 0.000 | 0.000 | 0.074 +/- 0.012 | 0.165 |
| doorplus_uncertainty | relfp0.2 | 31.727 +/- 6.546 | 0.511 +/- 0.328 | 0.526 +/- 0.317 | 6.838 +/- 0.216 | 0.052 +/- 0.004 | 0.561 | 0.060 +/- 0.011 | 0.132 |

Reading:

- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.
- `Selected-TrueRisk-Rel@K` is the selected non-FP high-risk relation count divided by K_rel=4.
- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.
- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.
