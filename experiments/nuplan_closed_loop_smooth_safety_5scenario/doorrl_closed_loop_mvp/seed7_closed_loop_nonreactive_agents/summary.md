# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 1.000 | 0.000 | 0.164 | 1.000 | 1.000 | 1.000 | 0.000 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | nan | 1388.2 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | nan | 1428.0 |
| doorrl_wm_object | False | 336d83c2bf6e5efd | nan | 1524.0 |
| doorrl_wm_decoupled_no_vis | False | 336d83c2bf6e5efd | nan | 1523.2 |
| doorrl_wm_object | False | 693ca17ec8b251d7 | nan | 1524.3 |
| doorrl_wm_decoupled_no_vis | False | 693ca17ec8b251d7 | nan | 1515.0 |
| doorrl_wm_object | True | a040ab916275556d | 0.5589 | 1323.4 |
| doorrl_wm_decoupled_no_vis | True | a040ab916275556d | nan | 1376.1 |
| doorrl_wm_object | True | d178ef27e36c53b6 | nan | 1357.0 |
| doorrl_wm_decoupled_no_vis | True | d178ef27e36c53b6 | 0.4609 | 1353.3 |
