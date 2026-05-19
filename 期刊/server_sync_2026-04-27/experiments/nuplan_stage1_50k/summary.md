# nuPlan Stage1 50k summary

Setup: 50k balanced nuPlan NPZ subset, seeds 7/42/123, 10 epochs, batch 128, horizon 5, Stage0 warm-start from `experiments/nuplan_stage0_50k_seed7`, lazy loading with 32 DataLoader workers per seed.

| condition | Return mean +/- seed-std | ImagColl mean +/- seed-std | CollMean mean +/- seed-std | Stability mean | StabilityGlobal mean |
|---|---:|---:|---:|---:|---:|
| wm_object | 1.723 +/- 17.886 | 0.610 +/- 0.146 | 0.614 +/- 0.113 | 0.367 | 0.025 |
| wm_decoupled | -0.330 +/- 4.936 | 0.007 +/- 0.012 | 0.277 +/- 0.111 | 0.255 | 0.005 |
| wm_decoupled_no_vis | 14.511 +/- 2.925 | 0.259 +/- 0.045 | 0.277 +/- 0.033 | 0.222 | 0.043 |

Per-seed latent return:

- wm_object: s7=1.834, s42=19.554, s123=-16.218
- wm_decoupled: s7=-5.308, s42=-0.245, s123=4.562
- wm_decoupled_no_vis: s7=15.833, s42=16.542, s123=11.158

Per-seed imagined collision rate:

- wm_object: s7=0.655, s42=0.728, s123=0.447
- wm_decoupled: s7=0.021, s42=0.000, s123=0.000
- wm_decoupled_no_vis: s7=0.307, s42=0.216, s123=0.256

Quick read:

- The 20k conclusion survives the 50k scale-up: `wm_decoupled_no_vis` remains clearly stronger than `wm_object`.
- `wm_decoupled_no_vis` is positive on all three seeds and has much lower collision.
- The added `wm_decoupled(+vis)` arm nearly eliminates binary collision, but return is near zero / negative on average and training shows late sanity-loss blow-up. This is not the clean final winner.
- `wm_object` remains high-variance: seed42 is strong, seed123 collapses, and the mean collision rate is high.
- This is the strongest Stage-1 evidence so far that nuPlan favors the decoupled-no-visibility abstraction.
