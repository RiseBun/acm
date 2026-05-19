# Stage-0 Oracle / Adaptive Split Summary

Common scenes: 140

| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |
|---|---:|---:|---:|---:|---:|---|
| `fixed_14/2` | 0.5703 | 1.0510 | 0.8064 | 0.7962 | 0.2902 | - |
| `fixed_12/4` | 1.6818 | 1.1069 | 0.9306 | 0.6135 | 0.2917 | - |
| `fixed_10/6` | 9.1289 | 2.2822 | 0.7960 | 0.8038 | 0.2906 | - |
| `oracle_dyn_rare_intrec` | 1.2804 | 0.8616 | 0.9404 | 0.7106 | 0.2906 | 10/6:27, 12/4:63, 14/2:50 |
| `oracle_collision` | 4.8529 | 1.6637 | 0.8358 | 0.8645 | 0.2926 | 10/6:59, 12/4:9, 14/2:72 |
| `oracle_pareto` | 1.5541 | 0.8346 | 0.9157 | 0.7578 | 0.2905 | 10/6:30, 12/4:42, 14/2:68 |
| `adaptive_risk_density` | 4.5959 | 1.5966 | 0.8358 | 0.7726 | 0.2908 | 10/6:70, 12/4:35, 14/2:35 |

Oracle policies choose the split per scene and then aggregate the selected per-sample numerators/denominators. `adaptive_risk_density` uses only scene token statistics: high risky-relation density -> `10/6`, high dynamic density -> `14/2`, otherwise `12/4`.
