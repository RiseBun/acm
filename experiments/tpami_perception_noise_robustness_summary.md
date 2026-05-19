# TPAMI perception-noise robustness summary

This note summarizes the evaluation-only perception-noise probe for the TPAMI
extension of typed-budget object-relation world models.

## Source

- Script: `scripts/perception_noise_robustness.py`
- Main output: `experiments/tpami_noise_robustness_h10_10k/summary.md`
- Smoke output: `experiments/tpami_noise_robustness_smoke_500/summary.md`
- Checkpoints:
  - `experiments/nuplan_stage1_50k/seed*/wm_object/model.pt`
  - `experiments/nuplan_stage1_50k/seed*/wm_decoupled_no_vis/model.pt`
  - `experiments/nuplan_stage1_shared_relation_50k/seed*/wm_naive/model.pt`

## Setup

All experiments are evaluation-time input corruptions on nuPlan 50k validation
samples. No model is retrained. The formal run uses three seeds, 10k validation
samples per seed, and horizon 10 latent imagination.

Corruptions:

- `clean`: original token input.
- `loc1.5`: Gaussian localization noise with std 1.5 m on dynamic/relation x-y features.
- `miss0.2`: randomly masks 20% of dynamic and relation tokens.
- `relfp0.2`: turns 20% of existing relation tokens into high-risk false-positive relations.

## Main table

| condition | corruption | Return | CollRate | CollMean | Teacher MSE |
|---|---|---:|---:|---:|---:|
| wm_object | clean | 1.922 +/- 41.820 | 0.676 +/- 0.095 | 0.673 +/- 0.062 | 8.863 +/- 0.370 |
| wm_object | loc1.5 | 1.932 +/- 41.850 | 0.675 +/- 0.095 | 0.673 +/- 0.062 | 8.864 +/- 0.369 |
| wm_object | miss0.2 | 2.116 +/- 41.940 | 0.640 +/- 0.087 | 0.644 +/- 0.059 | 8.880 +/- 0.391 |
| wm_object | relfp0.2 | 1.918 +/- 41.833 | 0.676 +/- 0.096 | 0.673 +/- 0.063 | 8.863 +/- 0.370 |
| wm_naive | clean | -5.998 +/- 11.006 | 0.415 +/- 0.521 | 0.450 +/- 0.239 | 8.912 +/- 0.245 |
| wm_naive | loc1.5 | -5.996 +/- 11.012 | 0.416 +/- 0.521 | 0.451 +/- 0.238 | 8.911 +/- 0.244 |
| wm_naive | miss0.2 | -6.012 +/- 11.016 | 0.407 +/- 0.525 | 0.444 +/- 0.245 | 8.919 +/- 0.253 |
| wm_naive | relfp0.2 | -6.012 +/- 10.996 | 0.416 +/- 0.521 | 0.453 +/- 0.236 | 8.921 +/- 0.259 |
| wm_decoupled_no_vis | clean | 33.756 +/- 7.213 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 |
| wm_decoupled_no_vis | loc1.5 | 33.739 +/- 7.226 | 0.262 +/- 0.046 | 0.280 +/- 0.032 | 6.628 +/- 0.110 |
| wm_decoupled_no_vis | miss0.2 | 30.567 +/- 5.604 | 0.249 +/- 0.034 | 0.263 +/- 0.024 | 6.633 +/- 0.116 |
| wm_decoupled_no_vis | relfp0.2 | 32.831 +/- 6.923 | 0.459 +/- 0.132 | 0.471 +/- 0.122 | 6.726 +/- 0.251 |

## TPAMI reading

The typed-budget model remains the strongest condition under clean input,
localization noise, and random misses. Its return and teacher-action alignment
also remain much better than object-only and shared top-k.

The main vulnerability is false-positive relation risk. Under `relfp0.2`,
`wm_decoupled_no_vis` degrades from CollRate `0.260` to `0.459`. This is still
better than the object-only clean CollRate `0.676`, but it shows that typed
relation slots should not blindly trust relation risk/TTC features.

This gives a concrete TPAMI extension path: a DOOR+ model should add
uncertainty-aware relation gating or confidence-weighted relation scoring, while
preserving the typed dynamic budget that keeps entity retention stable.

The shared top-k baseline is high-variance and weak in return. Its noisy-input
rows are not a clean robustness win; they mostly indicate that the trained
shared-relation policy is already unstable/degenerate, so it should be used as a
mechanism baseline rather than as a strong noisy-perception competitor.

## Safe claim

Typed-budget abstraction improves the clean and noisy-input evaluation profile
on nuPlan 50k, especially under localization noise and random token misses. Its
largest remaining sensitivity is high-risk relation false positives, motivating
uncertainty-aware relation scoring as a TPAMI extension.
