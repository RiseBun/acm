# Stage 1 Rel-to-Critic-Only Fusion Ablation Summary

Seeds: [7, 42, 123]  Condition: `wm_decoupled_rel_to_critic_only`  Fusion: actor=dyn, critic=dyn+rel

| condition | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_decoupled_rel_to_critic_only | 5.725 ± 7.248 | 0.729 ± 0.169 | 0.736 ± 0.163 | 0.453 ± 0.228 | 0.170 ± 0.036 |

## Per-seed Raw Values

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.274 | 0.683 | 0.685 | 0.139 | 0.179 |
| 42 | 15.922 | 0.549 | 0.567 | 0.676 | 0.122 |
| 123 | 1.526 | 0.956 | 0.957 | 0.544 | 0.208 |

## Quick Read

- Rel-to-critic-only improves collision rate over default 12+4 decoupled on average, but does not close the return gap to `wm_object`.
- The result is still high-variance: seed 42 is competitive, seed 123 remains a collapse-like high-collision run.
