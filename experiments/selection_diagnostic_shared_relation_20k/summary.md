# Selection Diagnostic Summary

| condition | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_naive | 7 | 4000 | 0.48057149 | 0.51942851 | 0.06813866 | 0.031203125 | 0.81236375 | 0.19634427 | 0.0021945832 |
| wm_naive | 42 | 4000 | 0.76918973 | 0.23081027 | 0.080979284 | 0.053703125 | 0.81157334 | 0.12740495 | 0.064308212 |
| wm_naive | 123 | 4000 | 0.39146155 | 0.60853845 | 0.49931852 | 0.07759375 | 0.81428394 | 0.18470555 | 0.11047333 |

Notes:

- `CDR = |S_dyn ∩ C_dyn| / |C_dyn|`; `MissRate = 1 - CDR`.
- Current adapters serialize ego-object relation features, but not endpoint ids. Endpoint `j` is inferred by nearest dynamic-token `(x,y)`; unmatched endpoints are `-1`.
- For ego-object relations, a selected relation is counted as wasted when its non-ego endpoint is not selected.
