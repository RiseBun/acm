# PlanTF external nuPlan-devkit baseline sanity

Setup: external PlanTF checkpoint through `nuplan-devkit` closed-loop nonreactive simulation, fixed 13 `one_of_each_scenario_type` tokens, `max_iterations=20`, 4 workers. Metrics were disabled and simulation-log serialization was set to `none` to avoid nuPlan DB/metric/LZMA overhead; this run is an integration sanity / appendix anchor, not a main official closed-loop benchmark.

| item | value |
|---|---:|
| scenarios | 13 |
| successful | 13 |
| failed | 0 |
| success rate | 1.000 |
| mean duration (s) | 397.5 |
| min duration (s) | 232.8 |
| max duration (s) | 530.9 |

Per-scenario runner report:

| succeeded | scenario token | log name | duration (s) |
|---|---|---|---:|
| True | `0194edaf63e75241` | `2021.05.25.14.24.08_veh-25_01129_01494` | 512.9 |
| True | `0ef918a26914564e` | `2021.05.25.12.30.39_veh-25_00321_01196` | 508.1 |
| True | `3474e3d6c7485935` | `2021.05.25.15.59.03_veh-30_00885_01251` | 516.9 |
| True | `5466833cffc653c5` | `2021.05.25.14.16.10_veh-35_01100_01664` | 530.9 |
| True | `7d094d765a8f5ffa` | `2021.05.25.12.40.06_veh-47_05213_05515` | 396.0 |
| True | `8de10fd86b825304` | `2021.05.25.12.30.39_veh-25_00321_01196` | 403.4 |
| True | `938047e260ec51a5` | `2021.06.03.17.06.58_veh-35_00016_00450` | 391.8 |
| True | `93c583b46398560e` | `2021.05.25.12.30.39_veh-25_03774_03886` | 386.5 |
| True | `af024db641ce5b02` | `2021.05.25.15.14.31_veh-47_01863_02344` | 300.6 |
| True | `ba57e98e0c1752bf` | `2021.05.25.12.30.39_veh-25_00321_01196` | 329.9 |
| True | `bb0531a7b6aa5d2e` | `2021.05.25.15.14.31_veh-47_01863_02344` | 307.9 |
| True | `d417ec1ee7295c5f` | `2021.05.25.12.40.06_veh-47_01110_01596` | 349.1 |
| True | `f3895453b6c35e51` | `2021.05.25.12.30.39_veh-25_00321_01196` | 232.8 |

Quick read:

- PlanTF now runs end-to-end in the nuPlan closed-loop wrapper: 13/13 selected scenarios completed without planner/runtime failure.
- The previous trajectory conversion failure was fixed by generating timestamps from the actual predicted trajectory length.
- Runtime is dominated by nuPlan DB/map/metric plumbing, not GPU inference. PlanTF forward calls in the logs are typically sub-second after feature construction.
- Keep this result as an external non-family wrapper sanity / appendix result. For main NeurIPS A1, prefer the lower-cost BC / planner-target imitation baseline recommended in `docs/1.md`.

## Official metric smoke

A separate 1-scenario official-metric smoke was run with `run_metric=true` and simulation-log serialization disabled:

| item | value |
|---|---:|
| scenario | `8de10fd86b825304` |
| scenario type | `traversing_traffic_light_intersection` |
| succeeded | true |
| duration (s) | 232.0 |
| final score | 0.0 |
| drivable area compliance | 0.0 |
| driving direction compliance | 0.0 |
| ego is comfortable | 0.0 |
| ego is making progress | 0.0 |
| no ego at-fault collisions | 0.0 |
| time to collision within bound | 0.0 |

Metric artifact:

- Aggregator parquet: `/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/planTF/experiments/exp/simulation/closed_loop_nonreactive_agents/plantf_baseline/single_right_turn_h20_20step_metric_nosimlog/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2026.04.28.03.12.18.parquet`
- Runner report: `/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/planTF/experiments/exp/simulation/closed_loop_nonreactive_agents/plantf_baseline/single_right_turn_h20_20step_metric_nosimlog/runner_report.parquet`

Reading: this confirms official metrics can be produced for PlanTF, but the single-scenario score is 0 and should be treated as a debug / appendix smoke only. The 13-scenario `one_of_each` run currently remains no-metric because the metric path was too slow for the payoff.

Artifacts:

- Runner report: `/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/planTF/experiments/exp/simulation/closed_loop_nonreactive_agents/plantf_baseline/one_of_each_h20_20step_tokens_nometric_nosimlog/runner_report.parquet`
- Log: `/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/planTF/experiments/plantf_baseline_smoke/logs/one_of_each_h20_20step_tokens_nometric_nosimlog.log`
