# Selection Diagnostic Summary

| condition | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_object | 7 | 4 | 0.906 | 0.094 | nan | 0.000 | 0.923 | 1.000 | 0.000 |

Notes:

- `CDR = |S_dyn ∩ C_dyn| / |C_dyn|`; `MissRate = 1 - CDR`.
- Current adapters serialize ego-object relation features, but not endpoint ids. Endpoint `j` is inferred by nearest dynamic-token `(x,y)`; unmatched endpoints are `-1`.
- For ego-object relations, a selected relation is counted as wasted when its non-ego endpoint is not selected.
