# TPAMI Closed-Loop WM/DOOR 22-Success Valid-DB Summary

Near-official nuPlan closed-loop sanity. This is an appendix-scale stability check, not a superiority claim.

Successful scenarios: 22.
Successful simulations: 44.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction | runs | runtime mean | duration mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.506 | 0.705 | 1.000 | 0.909 | 0.844 | 0.727 | 0.591 | 0.868 | 0.955 | 22 | 0.383 | 205.629 |
| doorrl_wm_object | 0.523 | 0.682 | 1.000 | 0.909 | 0.780 | 0.727 | 0.636 | 0.926 | 0.932 | 22 | 0.376 | 205.620 |

## Per-Attempt Final Scores

| attempt | planner | score | no at-fault collision | progress ratio | TTC |
|---|---|---:|---:|---:|---:|
| base_attempt01 | doorrl_wm_decoupled_no_vis | 0.326 | 0.500 | 1.000 | 0.000 |
| base_attempt01 | doorrl_wm_object | 0.000 | 0.000 | 0.340 | 0.000 |
| base_attempt02 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt02 | doorrl_wm_object | 0.000 | 0.000 | 0.265 | 0.000 |
| base_attempt03 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt03 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt04 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt04 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt05 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt05 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt06 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt06 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt07 | doorrl_wm_decoupled_no_vis | 0.312 | 1.000 | 1.000 | 0.000 |
| base_attempt07 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt08 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt08 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| base_attempt09 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt09 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| base_attempt10 | doorrl_wm_decoupled_no_vis | 0.992 | 1.000 | 1.000 | 1.000 |
| base_attempt10 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| extra_attempt11 | doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 0.139 | 1.000 |
| extra_attempt11 | doorrl_wm_object | 0.000 | 1.000 | 0.139 | 1.000 |
| extra_attempt12 | doorrl_wm_decoupled_no_vis | 0.830 | 1.000 | 0.993 | 1.000 |
| extra_attempt12 | doorrl_wm_object | 0.828 | 1.000 | 0.994 | 1.000 |
| extra_attempt13 | doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 0.172 | 0.000 |
| extra_attempt13 | doorrl_wm_object | 0.000 | 1.000 | 0.172 | 0.000 |
| extra_attempt15 | doorrl_wm_decoupled_no_vis | 0.419 | 1.000 | 0.767 | 1.000 |
| extra_attempt15 | doorrl_wm_object | 0.419 | 1.000 | 0.767 | 1.000 |
| extra_attempt16 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| extra_attempt16 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| extra_attempt17 | doorrl_wm_decoupled_no_vis | 0.987 | 1.000 | 1.000 | 1.000 |
| extra_attempt17 | doorrl_wm_object | 0.987 | 1.000 | 1.000 | 1.000 |
| extra_attempt19 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| extra_attempt19 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| extra_attempt20 | doorrl_wm_decoupled_no_vis | 0.868 | 1.000 | 1.000 | 1.000 |
| extra_attempt20 | doorrl_wm_object | 0.868 | 1.000 | 1.000 | 1.000 |
| extra_attempt21 | doorrl_wm_decoupled_no_vis | 0.770 | 1.000 | 0.263 | 1.000 |
| extra_attempt21 | doorrl_wm_object | 0.770 | 1.000 | 0.263 | 1.000 |
| extra_attempt22 | doorrl_wm_decoupled_no_vis | 1.000 | 1.000 | 1.000 | 1.000 |
| extra_attempt22 | doorrl_wm_object | 1.000 | 1.000 | 1.000 | 1.000 |
| extra_attempt23 | doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 1.000 | 0.000 |
| extra_attempt23 | doorrl_wm_object | 0.000 | 0.000 | 1.000 | 0.000 |
| extra_attempt25 | doorrl_wm_decoupled_no_vis | 0.634 | 1.000 | 0.230 | 1.000 |
| extra_attempt25 | doorrl_wm_object | 0.634 | 1.000 | 0.230 | 1.000 |
