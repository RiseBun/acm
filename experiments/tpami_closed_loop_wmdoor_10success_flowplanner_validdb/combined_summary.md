# TPAMI Closed-Loop WM/DOOR 10-Success Valid-DB Summary

Near-official nuPlan closed-loop sanity over 10 valid DB scenarios, WM-Object vs DOOR, seed7 K64 Stage-1 checkpoints.

Successful scenarios: 10.
Successful simulations: 20.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction | runs | runtime mean | duration mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.463 | 0.550 | 1.000 | 1.000 | 1.000 | 0.800 | 0.400 | 0.871 | 1.000 | 10 | 0.308 | 150.047 |
| doorrl_wm_object | 0.500 | 0.500 | 1.000 | 1.000 | 0.860 | 0.800 | 0.500 | 1.000 | 0.950 | 10 | 0.290 | 149.948 |

## Per-Attempt Final Scores

| attempt | planner | score | no at-fault collision | progress ratio | TTC |
|---|---|---:|---:|---:|---:|
| attempt01 | doorrl_wm_decoupled_no_vis | 0.326 | 0.500 | 1.000 | 0.000 |
| attempt01 | doorrl_wm_object | 0.000 | 0.000 | 0.340 | 0.000 |
| attempt02 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt02 | doorrl_wm_object | 0.000 | 0.000 | 0.265 | 0.000 |
| attempt03 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt03 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt04 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt04 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt05 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt05 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt06 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt06 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt07 | doorrl_wm_decoupled_no_vis | 0.312 | 1.000 | 1.000 | 0.000 |
| attempt07 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt08 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt08 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| attempt09 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt09 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| attempt10 | doorrl_wm_decoupled_no_vis | 0.992 | 1.000 | 1.000 | 1.000 |
| attempt10 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
