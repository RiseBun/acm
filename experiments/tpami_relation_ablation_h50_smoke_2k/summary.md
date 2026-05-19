# nuPlan relation feature-group ablation

Evaluation-time ablation on trained Stage1 checkpoints; no retraining.

Setup: seeds [7, 42, 123], conditions ['wm_decoupled_no_vis'], horizon=50, val split from nuPlan 50k.

| condition | ablation | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| wm_decoupled_no_vis | none | 6.780 ± 0.183 | 2.130 ± 0.094 | 187.554 ± 41.935 | 0.264 ± 0.047 | 0.281 ± 0.035 | 0.023 |
| wm_decoupled_no_vis | no_ttc_risk | 8.163 ± 0.859 | 3.187 ± 0.332 | 179.543 ± 40.010 | 0.898 ± 0.129 | 0.893 ± 0.128 | 0.023 |
| wm_decoupled_no_vis | no_lane_priority | 6.780 ± 0.183 | 2.129 ± 0.092 | 187.573 ± 41.988 | 0.262 ± 0.053 | 0.278 ± 0.042 | 0.023 |
| wm_decoupled_no_vis | no_relation_semantics | 8.159 ± 0.884 | 3.183 ± 0.351 | 179.790 ± 40.358 | 0.895 ± 0.128 | 0.890 ± 0.127 | 0.023 |

Reading:

- `no_ttc_risk` zeros relation-token risk and TTC features.
- `no_lane_priority` zeros relation-token lane-conflict and priority features.
- This is an interpretation probe, not a new trained model condition.
