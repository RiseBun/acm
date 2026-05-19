# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 0.000 | 0.000 | 0.400 | 0.299 | 1.000 | 0.000 | 0.983 | 0.000 |
| doorrl_wm_object | 0.000 | 0.000 | 0.000 | 0.200 | 0.032 | 1.000 | 0.000 | 0.983 | 0.200 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | nan | 460.0 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | nan | 467.6 |
| doorrl_wm_object | True | 336d83c2bf6e5efd | nan | 460.3 |
| doorrl_wm_decoupled_no_vis | True | 336d83c2bf6e5efd | nan | 461.8 |
| doorrl_wm_object | True | 693ca17ec8b251d7 | 0.0588 | 450.8 |
| doorrl_wm_decoupled_no_vis | True | 693ca17ec8b251d7 | nan | 471.6 |
| doorrl_wm_object | True | a040ab916275556d | nan | 461.5 |
| doorrl_wm_decoupled_no_vis | True | a040ab916275556d | 0.0770 | 442.2 |
| doorrl_wm_object | True | d178ef27e36c53b6 | nan | 456.6 |
| doorrl_wm_decoupled_no_vis | True | d178ef27e36c53b6 | nan | 463.4 |
