# TPAMI long-horizon relation semantics ablation

This note consolidates the TPAMI-oriented relation feature ablations run on the
nuPlan 50k Stage1 checkpoints. All rows evaluate `wm_decoupled_no_vis` with
three seeds (`7, 42, 123`) and 10k validation samples per seed. The experiment is
evaluation-only: no model is retrained.

## Setup

- Checkpoints: `experiments/nuplan_stage1_50k/seed*/wm_decoupled_no_vis/model.pt`
- Script: `scripts/relation_feature_group_ablation.py`
- Dataset index: `experiments/nuplan_50k_balanced_paths_seed7.json`
- Ablations:
  - `none`: original relation features.
  - `no_ttc_risk`: zeros relation-token risk and TTC features.
  - `no_lane_priority`: zeros relation-token lane-conflict and priority features.
  - `no_relation_semantics`: zeros risk, TTC, lane-conflict, and priority.

## Cross-horizon summary

| Horizon | Ablation | Return | CollRate | CollMean | Stability |
|---:|---|---:|---:|---:|---:|
| 10 | none | 33.746 +/- 7.209 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 0.111 |
| 10 | no_ttc_risk | 27.475 +/- 5.710 | 0.895 +/- 0.126 | 0.891 +/- 0.126 | 0.111 |
| 10 | no_lane_priority | 33.755 +/- 7.239 | 0.257 +/- 0.052 | 0.274 +/- 0.040 | 0.111 |
| 20 | none | 72.241 +/- 16.020 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 0.056 |
| 20 | no_ttc_risk | 65.535 +/- 14.464 | 0.896 +/- 0.126 | 0.891 +/- 0.126 | 0.056 |
| 20 | no_lane_priority | 72.231 +/- 16.064 | 0.257 +/- 0.052 | 0.274 +/- 0.040 | 0.056 |
| 50 | none | 187.733 +/- 42.602 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 0.023 |
| 50 | no_ttc_risk | 179.790 +/- 40.690 | 0.896 +/- 0.126 | 0.891 +/- 0.126 | 0.023 |
| 50 | no_lane_priority | 187.724 +/- 42.610 | 0.257 +/- 0.052 | 0.274 +/- 0.040 | 0.023 |

## Reading

Across horizons 10, 20, and 50, removing TTC/risk relation features consistently
raises imagined collision rate from about `0.26` to about `0.90`. Removing
lane-conflict/priority features is nearly neutral in this probe. This supports a
mechanistic TPAMI claim: the useful relation pathway in the current nuPlan
`wm_decoupled_no_vis` policy is concentrated in risk/TTC semantics, and that
dependence persists under long latent rollouts.

These results should be described as an intervention-style sensitivity analysis,
not as causal proof or a newly trained model condition.

## Source summaries

- `experiments/tpami_relation_ablation_h10_10k/summary.md`
- `experiments/tpami_relation_ablation_h20_10k/summary.md`
- `experiments/tpami_relation_ablation_h50_10k/summary.md`
