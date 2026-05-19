# TPAMI Closed-Loop WM/DOOR 5x1 Valid-DB Summary

Near-official nuPlan closed-loop sanity. db2 was skipped after initialization stall; use as smoke sanity, not final large-scale claim.

Successful scenarios: 4 / 5 attempted valid DBs (db2 skipped).
Successful simulations: 8.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction | runs | runtime mean | duration mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.566 | 0.625 | 1.000 | 1.000 | 1.000 | 0.750 | 0.500 | 0.963 | 1.000 | 4 | 0.303 | 138.038 |
| doorrl_wm_object | 0.500 | 0.500 | 1.000 | 1.000 | 0.835 | 0.750 | 0.500 | 1.000 | 1.000 | 4 | 0.263 | 137.882 |
