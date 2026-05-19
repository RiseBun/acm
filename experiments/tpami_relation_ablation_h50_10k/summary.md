# nuPlan relation feature-group ablation

Evaluation-time ablation on trained Stage1 checkpoints; no retraining.

Setup: seeds [7, 42, 123], conditions ['wm_decoupled_no_vis'], horizon=50, val split from nuPlan 50k.

| condition | ablation | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_decoupled_no_vis | none | 6.628 ± 0.110 | 2.115 ± 0.083 | 187.733 ± 42.602 | 0.260 ± 0.045 | 0.277 ± 0.033 | 0.023 |
| wm_decoupled_no_vis | no_ttc_risk | 7.992 ± 0.805 | 3.154 ± 0.335 | 179.790 ± 40.690 | 0.896 ± 0.126 | 0.891 ± 0.126 | 0.023 |
| wm_decoupled_no_vis | no_lane_priority | 6.627 ± 0.111 | 2.113 ± 0.081 | 187.724 ± 42.610 | 0.257 ± 0.052 | 0.274 ± 0.040 | 0.023 |
| wm_decoupled_no_vis | no_relation_semantics | 7.989 ± 0.827 | 3.151 ± 0.351 | 179.973 ± 40.977 | 0.894 ± 0.125 | 0.888 ± 0.126 | 0.023 |

Reading:

- `no_ttc_risk` zeros relation-token risk and TTC features.
- `no_lane_priority` zeros relation-token lane-conflict and priority features.
- This is an interpretation probe, not a new trained model condition.
