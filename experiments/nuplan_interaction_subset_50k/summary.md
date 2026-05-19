# nuPlan Interaction-Conditioned Subset Analysis

Offline validation analysis using existing nuPlan 50k Stage-1 checkpoints.
This is not a closed-loop evaluation.

Subsets: low_ttc_proxy <= 5.0s, dense_agents >= 12, rare_agent_dense rare>=1 and dyn>=8.

## all_val

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 10000 | 8.863 ± 0.370 | 1.718 ± 17.895 | 0.610 ± 0.146 |
| wm_decoupled_no_vis | 10000 | 6.628 ± 0.110 | 14.512 ± 2.927 | 0.260 ± 0.045 |

## dense_agents

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 8690 | 8.692 ± 0.414 | 1.423 ± 17.975 | 0.637 ± 0.170 |
| wm_decoupled_no_vis | 8690 | 6.340 ± 0.170 | 14.311 ± 3.127 | 0.272 ± 0.049 |

## high_interaction_union

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 9594 | 8.836 ± 0.410 | 1.588 ± 17.940 | 0.625 ± 0.150 |
| wm_decoupled_no_vis | 9594 | 6.555 ± 0.154 | 14.418 ± 2.994 | 0.270 ± 0.047 |

## lane_conflict

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 6699 | 7.023 ± 0.393 | 0.943 ± 17.975 | 0.591 ± 0.176 |
| wm_decoupled_no_vis | 6699 | 4.225 ± 0.125 | 13.330 ± 3.134 | 0.205 ± 0.043 |

## low_ttc_proxy

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 4745 | 12.796 ± 0.425 | 2.650 ± 18.243 | 0.791 ± 0.084 |
| wm_decoupled_no_vis | 4745 | 11.534 ± 0.198 | 15.946 ± 2.851 | 0.541 ± 0.090 |

## rare_agent_dense

| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|---:|
| wm_object | 7280 | 8.729 ± 0.369 | 1.562 ± 17.985 | 0.626 ± 0.149 |
| wm_decoupled_no_vis | 7280 | 6.494 ± 0.137 | 14.496 ± 3.085 | 0.273 ± 0.045 |
