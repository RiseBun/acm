# nuPlan 50k Offline Planner-Like Sanity Check

这是一个更下游的 offline planner-like sanity check，不是正式 external closed-loop evaluation 的替代品。

设置：nuPlan 50k val split，seeds [7, 42, 123]，conditions ['wm_object', 'wm_decoupled_no_vis']，horizon=5，lazy loading + 32 DataLoader workers。

| condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Policy action L2 | Return ↑ | CollRate ↓ | CollMean ↓ | Stability | Progress proxy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_object | 8.863 ± 0.370 | 3.553 ± 0.118 | 3.010 | 1.722 ± 17.888 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 | 2.995 |
| wm_decoupled_no_vis | 6.628 ± 0.110 | 2.115 ± 0.083 | 1.512 | 14.512 ± 2.925 | 0.259 ± 0.045 | 0.277 ± 0.033 | 0.222 | 1.082 |

Per-seed raw view:

## wm_object

| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 9.269 | 1.835 | 0.655 | 0.662 | 0.214 | 3.000 |
| 42 | 8.543 | 19.553 | 0.728 | 0.695 | 0.652 | 2.985 |
| 123 | 8.776 | -16.223 | 0.447 | 0.485 | 0.236 | 3.000 |

## wm_decoupled_no_vis

| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6.742 | 15.842 | 0.306 | 0.312 | 0.225 | 0.062 |
| 42 | 6.521 | 16.536 | 0.216 | 0.246 | 0.176 | 2.444 |
| 123 | 6.621 | 11.158 | 0.256 | 0.273 | 0.266 | 0.740 |

## Reading

- 这个实验只能作为 downstream offline evidence；它没有 reactive agents，也没有 simulator。
- 主 planner-like 指标是 teacher-derived action MSE；安全性仍通过 imagined collision 近似观察。
- 如果结果支持某个条件，只能说明它在 offline planner-like probes 下更合理，不能声称已经通过正式闭环验证。
