# Stage-0 Adaptive / Oracle Split Upper-Bound

Source runs:

- `tpami_budget_split_sweep_stage0_seed7`
- `tpami_budget_split_sweep_stage0_confirm_seeds`

Protocol: `object_relation_decoupled`, `K=16`, splits `14/2`, `12/4`, `10/6`, `8/8`, seeds `7/42/123`.

## Fixed Split Mean +- Std

| Split `K_dyn/K_rel` | Dyn Rollout MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Collision F1 ↑ | Action MSE ↓ |
|---|---:|---:|---:|---:|---:|
| `14/2` | `0.0887 +- 0.0835` | `0.3040 +- 0.1028` | `1.0000 +- 0.0000` | `0.9094 +- 0.0143` | `0.2726 +- 0.0159` |
| `12/4` | `1.9364 +- 0.3868` | `0.6017 +- 0.1824` | `0.9721 +- 0.0148` | `0.9208 +- 0.0384` | `0.2688 +- 0.0214` |
| `10/6` | `9.2797 +- 1.2573` | `2.0925 +- 0.1782` | `0.8531 +- 0.0650` | `0.9402 +- 0.0341` | `0.2735 +- 0.0253` |
| `8/8` | `22.4076 +- 1.7378` | `4.5180 +- 0.4700` | `0.6805 +- 0.1302` | `0.9033 +- 0.0406` | `0.2646 +- 0.0236` |

## Aggregate Oracle Upper-Bound

The aggregate best split is metric-dependent:

- Dynamic rollout / rare-agent preservation / interaction recall: `14/2` is best.
- Collision F1: `10/6` is best.
- Action MSE: differences are small; `8/8` is numerically best but does not preserve dynamic/rare agents.

This supports the paper claim that typed budgeting exposes an allocation axis rather than a single universally optimal split. A relation-heavy allocation improves collision classification, while a dynamic-heavy allocation is required to preserve compact world-model state quality.

## Limitation

The existing sweep saved only aggregate `table3_results.json`; it did not keep `model.pt` or per-scene/per-sample predictions. Therefore this file is an aggregate upper-bound, not a true per-scene oracle or adaptive routing result.

To obtain a true adaptive/oracle split table, rerun Stage-0 evaluation with per-sample metric dumps for each fixed split, then compute:

- oracle per scene: choose the split minimizing a composite score such as `RareADE + DynMSE - lambda * CollisionF1`;
- simple adaptive heuristic: choose higher `K_dyn` for dense dynamic scenes and higher `K_rel` for high relation-risk density scenes.
