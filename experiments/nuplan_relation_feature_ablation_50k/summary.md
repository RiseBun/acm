# nuPlan relation feature-group ablation

Evaluation-time ablation on trained Stage1 checkpoints; no retraining.

Setup: seeds [7, 42, 123], conditions ['wm_decoupled_no_vis'], horizon=5, val split from nuPlan 50k.

| condition | ablation | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_decoupled_no_vis | none | 6.628 ± 0.110 | 2.115 ± 0.083 | 14.510 ± 2.921 | 0.260 ± 0.045 | 0.277 ± 0.033 | 0.222 |
| wm_decoupled_no_vis | no_ttc_risk | 7.992 ± 0.805 | 3.154 ± 0.335 | 8.498 ± 1.397 | 0.895 ± 0.126 | 0.891 ± 0.126 | 0.222 |
| wm_decoupled_no_vis | no_lane_priority | 6.627 ± 0.111 | 2.113 ± 0.081 | 14.526 ± 2.931 | 0.257 ± 0.052 | 0.274 ± 0.040 | 0.222 |
| wm_decoupled_no_vis | no_relation_semantics | 7.989 ± 0.827 | 3.151 ± 0.351 | 8.724 ± 1.629 | 0.894 ± 0.126 | 0.888 ± 0.127 | 0.222 |

Reading:

- `no_ttc_risk` zeros relation-token risk and TTC features.
- `no_lane_priority` zeros relation-token lane-conflict and priority features.
- This is an interpretation probe, not a new trained model condition.
