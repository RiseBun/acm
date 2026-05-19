# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 0.200 | 0.200 | 0.267 | 0.000 | 1.000 | 1.000 | 0.600 |
| doorrl_wm_object | 0.000 | 1.000 | 0.200 | 0.000 | 0.033 | 0.000 | 0.600 | 1.000 | 0.400 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | nan | 481.6 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | nan | 477.4 |
| doorrl_wm_object | True | 336d83c2bf6e5efd | nan | 476.1 |
| doorrl_wm_decoupled_no_vis | True | 336d83c2bf6e5efd | nan | 477.4 |
| doorrl_wm_object | True | 693ca17ec8b251d7 | nan | 497.4 |
| doorrl_wm_decoupled_no_vis | True | 693ca17ec8b251d7 | nan | 479.4 |
| doorrl_wm_object | True | a040ab916275556d | nan | 475.9 |
| doorrl_wm_decoupled_no_vis | True | a040ab916275556d | nan | 482.5 |
| doorrl_wm_object | True | d178ef27e36c53b6 | 0.2671 | 470.1 |
| doorrl_wm_decoupled_no_vis | True | d178ef27e36c53b6 | 0.1024 | 458.0 |
