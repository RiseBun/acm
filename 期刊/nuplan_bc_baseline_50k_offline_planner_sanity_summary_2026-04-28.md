# nuPlan 50k Offline Planner-Like Sanity Check

这是一个更下游的 offline planner-like sanity check，不是正式 external closed-loop evaluation 的替代品。

设置：nuPlan 50k val split，seeds [7, 42, 123]，conditions ['bc']，horizon=5，lazy loading + 8 DataLoader workers。

| condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Policy action L2 | Return ↑ | CollRate ↓ | CollMean ↓ | Stability | Progress proxy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | 8.824 ± 0.113 | 3.534 ± 0.017 | 3.000 | 0.433 ± 4.229 | 0.327 ± 0.566 | 0.479 ± 0.368 | 0.021 | 3.000 |

Per-seed raw view:

## bc

| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 8.953 | 5.308 | 0.980 | 0.902 | 0.006 | 3.000 |
| 42 | 8.744 | -2.255 | 0.000 | 0.299 | 0.055 | 3.000 |
| 123 | 8.776 | -1.754 | 0.000 | 0.237 | 0.003 | 3.000 |

## Reading

- 这个实验只能作为 downstream offline evidence；它没有 reactive agents，也没有 simulator。
- 主 planner-like 指标是 teacher-derived action MSE；安全性仍通过 imagined collision 近似观察。
- 如果结果支持某个条件，只能说明它在 offline planner-like probes 下更合理，不能声称已经通过正式闭环验证。
