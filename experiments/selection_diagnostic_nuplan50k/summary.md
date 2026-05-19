# Selection Diagnostic Summary

| condition | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_object | 7 | 10000 | 0.89339475 | 0.10660525 |  | 0 | 0.90198384 | 0.15946339 | 0.098037021 |
| wm_object | 42 | 10000 | 0.92953246 | 0.070467537 |  | 0 | 0.9005325 | 0.26229476 | 0.004522663 |
| wm_object | 123 | 10000 | 0.53324524 | 0.46675476 |  | 0 | 0.90172771 | 0.26037438 | 0.23357046 |
| wm_decoupled_no_vis | 7 | 10000 | 0.9428942 | 0.057105804 | 0.041708644 | 0.26016218 | 0.90198384 | -0.17566495 | -0.07434338 |
| wm_decoupled_no_vis | 42 | 10000 | 0.94241826 | 0.057581735 | 0.068553269 | 0.26014547 | 0.9005325 | -0.001510592 | -0.11303113 |
| wm_decoupled_no_vis | 123 | 10000 | 0.93126851 | 0.068731493 | 0.022348905 | 0.25955311 | 0.90172771 | -0.062775546 | -0.15782737 |

Notes:

- `CDR = |S_dyn ∩ C_dyn| / |C_dyn|`; `MissRate = 1 - CDR`.
- Current adapters serialize ego-object relation features, but not endpoint ids. Endpoint `j` is inferred by nearest dynamic-token `(x,y)`; unmatched endpoints are `-1`.
- For ego-object relations, a selected relation is counted as wasted when its non-ego endpoint is not selected.
