# Selection Diagnostic Summary

| condition | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_naive | 7 | 10000 | 0.29778587 | 0.70221413 | 0.37700535 | 0.081914741 | 0.90198384 | -0.036858284 | -0.14121432 |
| wm_naive | 42 | 10000 | 0.81193698 | 0.18806302 | 0.20126227 | 0.38940392 | 0.9005325 | 0.34579027 | -0.091170348 |
| wm_naive | 123 | 10000 | 0.24746919 | 0.75253081 | 0.27373543 | 0.1600651 | 0.90172771 | -0.10502619 | -0.087554531 |

Notes:

- `CDR = |S_dyn ∩ C_dyn| / |C_dyn|`; `MissRate = 1 - CDR`.
- Current adapters serialize ego-object relation features, but not endpoint ids. Endpoint `j` is inferred by nearest dynamic-token `(x,y)`; unmatched endpoints are `-1`.
- For ego-object relations, a selected relation is counted as wasted when its non-ego endpoint is not selected.
