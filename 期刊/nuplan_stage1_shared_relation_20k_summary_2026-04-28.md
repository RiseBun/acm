# nuPlan Stage1 shared-relation 20k summary

Setup: wm_naive = object_relation shared top-k + WM, nuPlan 20k balanced NPZ, seeds 7/42/123, Stage0 warm-start from `experiments/nuplan_stage0_20k_seed7/object_relation`.

| condition | Return mean +/- seed-std | ImagColl mean +/- seed-std | CollMean mean +/- seed-std | Stability mean |
|---|---:|---:|---:|---:|
| wm_naive | -2.655 +/- 14.632 | 0.252 +/- 0.050 | 0.297 +/- 0.069 | 0.186 |

Per-seed latent return:

- wm_naive: s7=-15.597, s42=13.222, s123=-5.591

Per-seed imagined collision rate:

- wm_naive: s7=0.263, s42=0.198, s123=0.297

Quick read:

- Shared-relation `wm_naive` is runnable but high-variance and much weaker than the typed decoupled no-vis 20k main condition on latent return.
- The Stage0 shared-relation warm-start is also weak on representation metrics (DynRollout 239.0, IntRec@1m 0.132), matching the slot-mixing failure story.
