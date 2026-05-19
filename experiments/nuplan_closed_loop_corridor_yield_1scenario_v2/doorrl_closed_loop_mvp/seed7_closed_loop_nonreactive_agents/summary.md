# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.825 | 0.000 |
| doorrl_wm_object | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.829 | 0.000 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | 0.4818 | 199.1 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | 0.5171 | 198.7 |
