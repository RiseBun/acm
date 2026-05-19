# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 1.000 | 0.000 | 1.000 | 0.620 | 1.000 | 1.000 | 1.000 | 0.000 |
| doorrl_wm_object | 0.000 | 1.000 | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 | 0.865 | 1.000 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | 0.4892 | 199.9 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | 0.5177 | 200.8 |
