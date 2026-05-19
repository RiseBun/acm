# nuPlan BC baseline 50k summary

Setup: external-style BC / planner-target imitation baseline using the existing `bc` condition in `run_stage1_table4.py`, `object_only` representation, nuPlan 50k balanced NPZ subset, seeds 7/42/123, 10 epochs, batch 128, horizon 5, Stage0 warm-start from `experiments/nuplan_stage0_50k_seed7/object_only`, lazy loading with 8 DataLoader workers per seed.

| condition | Return mean +/- seed-std | ImagColl mean +/- seed-std | CollMean mean +/- seed-std | Teacher action MSE +/- seed-std | Action delta L2 +/- seed-std | Stability mean | StabilityGlobal mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| bc | 0.433 +/- 4.230 | 0.327 +/- 0.566 | 0.479 +/- 0.368 | 8.824 +/- 0.113 | 3.534 +/- 0.017 | 0.021 | 0.006 |

Per-seed latent return:

- bc: s7=5.308, s42=-2.255, s123=-1.754

Per-seed imagined collision rate:

- bc: s7=0.980, s42=0.000, s123=0.000

Per-seed collision mean:

- bc: s7=0.902, s42=0.299, s123=0.237

Per-seed teacher action MSE:

- bc: s7=8.953, s42=8.744, s123=8.776

Per-seed teacher action delta L2:

- bc: s7=3.553, s42=3.523, s123=3.525

Quick read:

- The low-cost A1 BC / planner-target imitation baseline is runnable on the same nuPlan 50k Stage1 protocol and produces the same `stage1_metrics.json` format as the internal conditions.
- It is weaker and less stable than the main `wm_decoupled_no_vis` 50k condition by latent return and collision mean: BC return is 0.43 +/- 4.23 vs no-vis 14.51 +/- 2.93.
- Seed 7 is a high-collision outlier, while seeds 42/123 collapse to near-zero imagined binary collision but negative return. This is useful as a non-family anchor but not a competitive baseline.
- Offline planner sanity is complete: BC teacher action MSE is 8.824 +/- 0.113 and teacher action delta L2 is 3.534 +/- 0.017.

Artifacts:

- `seed7/bc/stage1_metrics.json`
- `seed42/bc/stage1_metrics.json`
- `seed123/bc/stage1_metrics.json`
- `logs/seed7_workers8.log`
- `logs/seed42_workers8.log`
- `logs/seed123_workers8.log`
- `offline_planner_sanity/summary.md`
- `offline_planner_sanity/summary.json`
