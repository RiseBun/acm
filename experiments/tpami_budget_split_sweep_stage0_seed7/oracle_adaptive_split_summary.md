# Stage-0 Oracle / Adaptive Split Summary

Common scenes: 140

| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |
|---|---:|---:|---:|---:|---:|---|
| `fixed_14/2` | 0.0173 | 0.1956 | 1.0000 | 0.9183 | 0.2880 | - |
| `fixed_12/4` | 1.9336 | 0.6104 | 0.9686 | 0.8846 | 0.2856 | - |
| `fixed_10/6` | 9.0221 | 2.1473 | 0.8232 | 0.9220 | 0.2844 | - |
| `oracle_dyn_rare_intrec` | 0.0803 | 0.1888 | 1.0000 | 0.9207 | 0.2870 | 10/6:4, 12/4:18, 14/2:118 |
| `oracle_collision` | 1.7938 | 0.5869 | 0.9791 | 0.9585 | 0.2858 | 10/6:33, 12/4:19, 14/2:88 |
| `oracle_pareto` | 0.1917 | 0.2218 | 0.9980 | 0.9344 | 0.2869 | 10/6:6, 12/4:33, 14/2:101 |
| `adaptive_risk_density` | 4.1922 | 1.2062 | 0.9095 | 0.9265 | 0.2830 | 10/6:70, 12/4:35, 14/2:35 |

Oracle policies choose the split per scene and then aggregate the selected per-sample numerators/denominators. `adaptive_risk_density` uses only scene token statistics: high risky-relation density -> `10/6`, high dynamic density -> `14/2`, otherwise `12/4`.
