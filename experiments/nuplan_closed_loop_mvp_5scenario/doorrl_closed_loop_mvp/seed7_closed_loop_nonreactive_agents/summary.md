# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 0.600 | 1.000 | 0.000 | 0.105 | 1.000 | 0.600 | 1.000 | 0.700 |
| doorrl_wm_object | 0.000 | 0.400 | 0.600 | 0.000 | 0.014 | 0.000 | 0.000 | 1.000 | 0.300 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | 0.1086 | 190.8 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | 0.0152 | 176.8 |
| doorrl_wm_object | True | 336d83c2bf6e5efd | 0.0147 | 180.0 |
| doorrl_wm_decoupled_no_vis | True | 336d83c2bf6e5efd | 0.0153 | 175.8 |
| doorrl_wm_object | True | 693ca17ec8b251d7 | 0.0148 | 178.0 |
| doorrl_wm_decoupled_no_vis | True | 693ca17ec8b251d7 | 0.0152 | 176.4 |
| doorrl_wm_object | True | a040ab916275556d | 0.0151 | 172.0 |
| doorrl_wm_decoupled_no_vis | True | a040ab916275556d | 0.0150 | 172.8 |
| doorrl_wm_object | True | d178ef27e36c53b6 | 0.0147 | 171.6 |
| doorrl_wm_decoupled_no_vis | True | d178ef27e36c53b6 | 0.0149 | 173.8 |
