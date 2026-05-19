# nuPlan Stage 1 5k Summary

Dataset: nuPlan preprocessed NPZ, 5000 samples per seed, 4000/1000 split. Stage0 warm-start: `experiments/nuplan_stage0_5k_seed7`. Stage1: 10 epochs, horizon=5, entropy_beta=0.003, action_clip=5.

| condition | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | -6.011 ± 3.101 | 0.348 ± 0.254 | 0.384 ± 0.150 | 0.505 ± 0.441 | 0.010 ± 0.006 |
| wm_decoupled | 9.383 ± 9.586 | 0.215 ± 0.095 | 0.281 ± 0.097 | 0.887 ± 0.011 | 0.067 ± 0.029 |
| wm_decoupled_no_vis | 12.907 ± 2.687 | 0.247 ± 0.029 | 0.325 ± 0.060 | 0.097 ± 0.062 | 0.037 ± 0.034 |
| wm_decoupled_rel_to_critic_only | 5.460 ± 2.258 | 0.231 ± 0.052 | 0.337 ± 0.025 | 0.848 ± 0.058 | 0.054 ± 0.023 |

## Per-seed Raw Values

### wm_object

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | -9.275 | 0.446 | 0.461 | 0.239 | 0.002 |
| 42 | -6.916 | 0.000 | 0.174 | 1.127 | 0.012 |
| 123 | -1.842 | 0.598 | 0.515 | 0.149 | 0.017 |

### wm_decoupled

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.961 | 0.250 | 0.352 | 0.902 | 0.083 |
| 42 | 4.394 | 0.309 | 0.347 | 0.876 | 0.025 |
| 123 | 22.795 | 0.085 | 0.144 | 0.884 | 0.092 |

### wm_decoupled_no_vis

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 16.640 | 0.261 | 0.372 | 0.036 | 0.019 |
| 42 | 10.426 | 0.273 | 0.363 | 0.073 | 0.009 |
| 123 | 11.656 | 0.207 | 0.241 | 0.183 | 0.085 |

### wm_decoupled_rel_to_critic_only

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 6.032 | 0.301 | 0.367 | 0.900 | 0.085 |
| 42 | 7.896 | 0.215 | 0.336 | 0.878 | 0.028 |
| 123 | 2.453 | 0.176 | 0.307 | 0.766 | 0.048 |

## Quick Read

- On this nuPlan 5k pilot, object-only is not the strongest Stage1 condition; it has negative mean return and higher collision than most decoupled variants.
- `wm_decoupled_no_vis` has the best mean return and low collision, which reverses the nuScenes visibility conclusion on this small nuPlan pilot.
- `wm_decoupled` has the best collision rate but high return variance, driven by a very strong seed 123.
- Treat this as a pilot: samples are 5k and Stage0 warm-start uses seed7 for all Stage1 seeds.
