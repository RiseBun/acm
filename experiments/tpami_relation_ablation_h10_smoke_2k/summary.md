# nuPlan relation feature-group ablation

Evaluation-time ablation on trained Stage1 checkpoints; no retraining.

Setup: seeds [7, 42, 123], conditions ['wm_decoupled_no_vis'], horizon=10, val split from nuPlan 50k.

| condition | ablation | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_decoupled_no_vis | none | 6.780 ± 0.183 | 2.130 ± 0.094 | 33.613 ± 7.054 | 0.264 ± 0.047 | 0.281 ± 0.035 | 0.111 |
| wm_decoupled_no_vis | no_ttc_risk | 8.163 ± 0.859 | 3.187 ± 0.332 | 27.354 ± 5.565 | 0.897 ± 0.129 | 0.893 ± 0.128 | 0.111 |
| wm_decoupled_no_vis | no_lane_priority | 6.780 ± 0.183 | 2.129 ± 0.092 | 33.637 ± 7.088 | 0.262 ± 0.053 | 0.278 ± 0.042 | 0.111 |
| wm_decoupled_no_vis | no_relation_semantics | 8.159 ± 0.884 | 3.183 ± 0.351 | 27.593 ± 5.832 | 0.895 ± 0.128 | 0.890 ± 0.127 | 0.111 |

Reading:

- `no_ttc_risk` zeros relation-token risk and TTC features.
- `no_lane_priority` zeros relation-token lane-conflict and priority features.
- This is an interpretation probe, not a new trained model condition.
