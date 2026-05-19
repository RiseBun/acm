# nuPlan 50k Offline Planner-Like Sanity Check

This is an offline planner-like sanity check, not a substitute for external closed-loop evaluation.

Setup: nuPlan 50k val split, seeds [7, 42, 123], conditions ['wm_object', 'wm_decoupled_no_vis'], horizon=5, lazy loading with 4 DataLoader workers.

| condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Policy action L2 | Return ↑ | CollRate ↓ | CollMean ↓ | Stability | Progress proxy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_object | 8.673 ± 0.950 | 3.529 ± 0.209 | 3.005 | 1.532 ± 17.598 | 0.594 ± 0.114 | 0.613 ± 0.098 | 0.367 | 2.994 |
| wm_decoupled_no_vis | 6.190 ± 1.149 | 1.927 ± 0.406 | 1.354 | 13.912 ± 2.490 | 0.254 ± 0.044 | 0.272 ± 0.029 | 0.227 | 1.002 |

Per-seed quick view:

## wm_object

| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 8.340 | 1.376 | 0.621 | 0.636 | 0.210 | 3.000 |
| 42 | 7.934 | 19.208 | 0.691 | 0.697 | 0.655 | 2.983 |
| 123 | 9.744 | -15.987 | 0.469 | 0.506 | 0.236 | 3.000 |

## wm_decoupled_no_vis

| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 5.495 | 15.087 | 0.285 | 0.291 | 0.235 | -0.083 |
| 42 | 5.559 | 15.598 | 0.203 | 0.238 | 0.180 | 2.371 |
| 123 | 7.516 | 11.053 | 0.273 | 0.287 | 0.266 | 0.718 |

## Reading

- Use this as downstream offline evidence only; it does not include reactive agents or a simulator.
- The primary planner-like metric is teacher-derived action MSE; safety is still measured through imagined collision.
- A favorable result means the Stage-1 policy is not only better under latent return, but also more reasonable under offline planner-like probes.
