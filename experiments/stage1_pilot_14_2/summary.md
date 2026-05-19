# Stage 1 14+2 Ablation Summary

Seeds: [7, 42, 123]  Condition: `wm_decoupled_14_2`  Budget: top_k_dyn=14, top_k_rel=2

| condition | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_decoupled_14_2 | 2.473 ± 4.140 | 0.808 ± 0.196 | 0.774 ± 0.189 | 0.419 ± 0.264 | 0.125 ± 0.068 |

## Per-seed Raw Values

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 3.157 | 0.987 | 0.986 | 0.598 | 0.206 |
| 42 | -2.903 | 0.535 | 0.528 | 0.614 | 0.128 |
| 123 | 7.167 | 0.903 | 0.808 | 0.046 | 0.041 |

## Quick Read

- 14+2 does not close the Stage-1 gap to `wm_object`: collision remains high and return remains low.
- It is not a clear improvement over default 12+4; collision is worse on average, though seed 42 still has a moderate-collision regime.
