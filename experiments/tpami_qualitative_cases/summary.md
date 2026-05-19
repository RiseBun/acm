# TPAMI Qualitative Case Studies

## Shared Top-K vs Typed DOOR

Source:

- Checkpoints: `experiments/table3_fair_fix2_seed7`
- Extraction: `experiments/tpami_qualitative_cases/slot_selections_seed7.pkl`
- Figures: `experiments/tpami_qualitative_cases/shared_vs_door`

Generated cases:

| case | sample idx | near agents | shared selected | DOOR selected | shared pred err | DOOR pred err | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 597 | 11 | 0 | 11 | 6.38 m | 0.37 m | shared top-K spends capacity away from nearby dynamic agents; typed budget keeps them |
| 1 | 599 | 11 | 0 | 11 | 6.78 m | 0.36 m | same mechanism on a neighboring scene frame |
| 2 | 600 | 11 | 0 | 11 | 6.78 m | 0.36 m | same mechanism on a neighboring scene frame |
| 3 | 601 | 11 | 0 | 11 | 6.86 m | 0.37 m | same mechanism on a neighboring scene frame |

Recommended paper usage: use case 0 as the main figure and keep the other
neighboring frames as appendix/backup. The main claim is qualitative, not a new
metric: a shared top-K bottleneck can spend all compact slots away from nearby
dynamic agents, while a typed budget preserves dynamic state under the same
16-slot capacity.

## DOOR+ FP Relation Suppression

Source:

- Aggregate result: `experiments/tpami_doorplus_proxy_confidence_seed7/combined_summary.md`
- Case figure: `experiments/tpami_qualitative_cases/doorplus_fp_relation/doorplus_fp_relation_case.{png,pdf}`
- Case metadata: `experiments/tpami_qualitative_cases/doorplus_fp_relation/case.json`

Aggregate result:

- Under `relfp0.2`, baseline DOOR selects `0.065` FP relations per relation
  budget, while DOOR+ proxy selects `0.040` and DOOR+ normal confidence selects
  `0.007`.
- True-risk relation density stays comparable (`0.128-0.130`), so the reduction
  is not simply dropping all relation tokens.

Generated case:

| scene | baseline FP rel selected | DOOR+ FP rel selected | baseline true-risk rel selected | DOOR+ true-risk rel selected | suppressed FP relation ids |
|---|---:|---:|---:|---:|---|
| `us-nv-las-vegas-strip_0a0a86cff5295b88` | 3 | 0 | 1 | 4 | `45,47,54` |

The selected false-positive relations have low proxy confidence
(`0.26-0.29`), while the retained true-risk relations have higher confidence
(`0.45-0.46`). This is the intended qualitative counterpart to the aggregate
DOOR+ table: the proxy does not blindly remove relation slots; it redirects the
relation budget away from unreliable false positives and toward consistent
high-risk relations.
