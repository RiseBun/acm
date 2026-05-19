# Adaptive Split Learned Router

This is a deployable proxy router trained only on scene token statistics, not per-split outcome metrics at test time.

Common scenes: 140; train scenes: 98; test scenes: 42.

Learned rule: order=`dynamic_first`, risk_threshold=0.1500, dynamic_threshold=0.3709, risk_split=`12/4`, dynamic_split=`14/2`, default_split=`14/2`.

## Train

| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |
|---|---:|---:|---:|---:|---:|---|
| `fixed_14/2` | 0.0185 | 0.1968 | 1.0000 | 0.9137 | 0.2647 | 14/2:98 |
| `fixed_12/4` | 1.9677 | 0.5607 | 0.9709 | 0.8749 | 0.2595 | 12/4:98 |
| `fixed_10/6` | 9.2944 | 2.0781 | 0.8301 | 0.9189 | 0.2579 | 10/6:98 |
| `oracle_pareto` | 0.2068 | 0.2288 | 0.9969 | 0.9314 | 0.2633 | 10/6:4, 12/4:24, 14/2:70 |
| `learned_router` | 0.0945 | 0.2013 | 0.9989 | 0.9187 | 0.2630 | 12/4:11, 14/2:87 |

## Held-Out Test

| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |
|---|---:|---:|---:|---:|---:|---|
| `fixed_14/2` | 0.0145 | 0.1931 | 1.0000 | 0.9295 | 0.3423 | 14/2:42 |
| `fixed_12/4` | 1.8528 | 0.7164 | 0.9644 | 0.9077 | 0.3464 | 12/4:42 |
| `fixed_10/6` | 8.3767 | 2.2949 | 0.8106 | 0.9296 | 0.3461 | 10/6:42 |
| `oracle_pareto` | 0.1560 | 0.2070 | 1.0000 | 0.9417 | 0.3416 | 10/6:2, 12/4:9, 14/2:31 |
| `learned_router` | 0.1615 | 0.2201 | 0.9976 | 0.9334 | 0.3435 | 12/4:9, 14/2:33 |

Reading: `oracle_pareto` is an upper bound that chooses the best split per scene using outcome metrics. `learned_router` uses only deployable scene statistics (`risky_relation_density`, `dynamic_density`) and the learned thresholds above.
