# nuPlan Stage1 20k summary

Setup: 20k balanced nuPlan NPZ subset, seeds 7/42/123, 10 epochs, batch 128, horizon 5, Stage0 warm-start from `experiments/nuplan_stage0_20k_seed7`, 32 tokenisation workers per seed.

| condition | Return mean +/- seed-std | ImagColl mean +/- seed-std | CollMean mean | Stability mean | StabilityGlobal mean |
|---|---:|---:|---:|---:|---:|
| wm_object | 4.738 +/- 13.946 | 0.373 +/- 0.083 | 0.395 | 0.426 | 0.047 |
| wm_decoupled | 13.477 +/- 4.086 | 0.488 +/- 0.217 | 0.509 | 0.546 | 0.095 |
| wm_decoupled_no_vis | 17.497 +/- 1.374 | 0.226 +/- 0.105 | 0.251 | 0.136 | 0.043 |

Per-seed latent return:

- wm_object: s7=20.581, s42=-0.694, s123=-5.674
- wm_decoupled: s7=15.707, s42=15.963, s123=8.760
- wm_decoupled_no_vis: s7=16.222, s42=17.319, s123=18.952

Per-seed imagined collision rate:

- wm_object: s7=0.277, s42=0.418, s123=0.424
- wm_decoupled: s7=0.294, s42=0.449, s123=0.722
- wm_decoupled_no_vis: s7=0.188, s42=0.346, s123=0.146

Quick read:

- The 5k nuPlan signal scales up: relation-aware decoupled variants are not a small-sample artifact on nuPlan.
- `wm_decoupled_no_vis` is the strongest 20k condition: highest Return, lowest collision, and lowest Return variance.
- `wm_decoupled(+vis)` improves Return over `wm_object`, but its collision metrics are worse and seed 123 is unstable.
- The cross-dataset visibility story is now stronger: visibility helps on nuScenes Stage1, but hurts or destabilises the enlarged nuPlan Stage1 setting.
