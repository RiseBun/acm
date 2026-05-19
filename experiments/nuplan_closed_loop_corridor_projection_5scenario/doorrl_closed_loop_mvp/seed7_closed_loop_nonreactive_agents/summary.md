# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.537 | 0.700 | 1.000 | 1.000 | 0.821 | 1.000 | 0.400 | 0.895 | 1.000 |
| doorrl_wm_object | 0.537 | 0.700 | 1.000 | 1.000 | 0.822 | 1.000 | 0.400 | 0.894 | 1.000 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | 0.1375 | 458.0 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | nan | 448.7 |
| doorrl_wm_object | True | 336d83c2bf6e5efd | nan | 474.1 |
| doorrl_wm_decoupled_no_vis | True | 336d83c2bf6e5efd | nan | 459.6 |
| doorrl_wm_object | True | 693ca17ec8b251d7 | nan | 475.4 |
| doorrl_wm_decoupled_no_vis | True | 693ca17ec8b251d7 | nan | 461.9 |
| doorrl_wm_object | True | a040ab916275556d | nan | 470.3 |
| doorrl_wm_decoupled_no_vis | True | a040ab916275556d | nan | 447.7 |
| doorrl_wm_object | True | d178ef27e36c53b6 | nan | 461.1 |
| doorrl_wm_decoupled_no_vis | True | d178ef27e36c53b6 | 0.1005 | 438.9 |
