# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.264 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.862 | 1.000 |
| doorrl_wm_object | 0.264 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.859 | 1.000 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | 0.3877 | 170.2 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | 0.3466 | 170.4 |
