# Discarded vs Reliable Conclusions

_Last updated: 2026-04-27._

This note is the rebuttal-facing index for claims that are safe to use, claims
that should be treated as appendix/support, and conclusions that were discarded
after multi-seed or larger-scale checks.

## Reliable Claims

| Claim | Evidence | Use in paper |
|---|---|---|
| Decoupled typed-slot abstraction improves Stage-0 representation sufficiency on nuScenes. | `experiments/table3_fair_fix2_aggregate.json`, 3 seeds, dynamic-slot-filtered evaluator. | Main representation result. |
| nuScenes Stage-1 policy learning currently favors `wm_object`. | `experiments/stage1_pilot_x/X_summary.{md,json}`, 3 seeds, K=5. | Main Stage-1 nuScenes result. |
| nuPlan Stage-1 ranking reverses: `wm_decoupled_no_vis` beats `wm_object` on 20k/50k, and remains the cleanest 50k winner after adding the missing +visibility arm. | `experiments/nuplan_stage1_20k/summary.md`, `experiments/nuplan_stage1_50k/seed*/<condition>/stage1_metrics.json`. | Main cross-dataset result. |
| nuPlan interaction-heavy subsets preserve the no-vis advantage. | `experiments/nuplan_interaction_subset_50k/summary.json`; strongest on `lane_conflict`. | Explanation / figure support. |
| Cross-dataset eval shows relation-aware structure is dataset/training-regime conditional. | `experiments/cross_dataset_eval/cross_dataset_eval_all.json`; nuPlan-trained no-vis transfers better to nuScenes than nuPlan-trained object, while nuScenes-trained +vis is unstable on nuPlan. | Discussion support. |
| Dataset statistics explain why visibility behaves differently across datasets. | `experiments/dataset_token_stats/summary.md`; nuPlan dynamic visibility is almost constant. | Discussion support, not a performance metric. |

## Appendix / Support Only

| Result | Why it is useful | Why it is not a main claim |
|---|---|---|
| nuPlan 5k Stage-1 pilot. | Early evidence that nuPlan differs from nuScenes. | Superseded by 20k/50k. |
| nuPlan 50k `wm_decoupled(+vis)`. | Fills the missing visibility arm; shows visibility can suppress binary collision but does not recover return. | Return is near zero / negative on average and training shows late sanity-loss blow-up, so no-vis remains the main condition. |
| `wm_decoupled_14_2` nuScenes ablation. | Shows the nuScenes Stage-1 failure is not fixed by simply changing dyn/rel budget from 12/4 to 14/2. | Negative ablation; not a new method. |
| `wm_decoupled_rel_to_critic_only`. | Shows actor/critic fusion tweaks help only partially. | Still does not close the gap to `wm_object`. |
| nuPlan closed-loop corridor projection. | Proves the model can be wired into nuPlan devkit and gives an appendix sanity table. | Wrapper dominates and the small official subset shows little model separation. |
| P1 case-study figures. | Directly explain ranking reversal and failure modes. | Qualitative / explanatory; keep backed by aggregate tables. |

## Discarded Claims

| Discarded claim | Why discarded | Replacement wording |
|---|---|---|
| "`wm_decoupled` is the best Stage-1 policy on nuScenes." | Multi-seed X reverses the single-seed pilot: `wm_object` has higher return and lower collision. | "`wm_decoupled` improves representation quality but is not the most stable nuScenes Stage-1 policy under the current imagination setup." |
| "Removing visibility is generally harmful." | True on nuScenes Stage-1, false on nuPlan 20k/50k where `wm_decoupled_no_vis` is strongest. | "Visibility is dataset-conditional: stabilizing on nuScenes, largely uninformative or harmful on nuPlan." |
| "Stage-0 ranking predicts Stage-1 ranking." | nuScenes Stage 0 favors decoupled, but Stage 1 favors object-only; nuPlan Stage 1 favors no-vis. | "Representation sufficiency and policy-learning stability are related but non-equivalent." |
| "Closed-loop nuPlan results are ready as a main benchmark." | Official small-subset runs are dominated by the planner wrapper and do not separate the policies cleanly. | "Closed-loop is an appendix sanity / integration proof; main nuPlan evidence is latent imagination plus offline planner-like probes." |
| "Single-seed Stage-1 pilots are enough for conclusions." | Several single-seed positives vanished under 3-seed reruns. | "Use 3-seed means/stds for all Stage-1 claims; single-seed runs are smoke tests only." |

## Running Follow-Ups

The durable follow-up watchers are:

- `scripts/p1_mount_watch_job.sh`: one independent `nohup` watcher per job
  (`A`, `B`, `C`). Each watcher waits for the relevant mount and then runs the
  experiment.
- `scripts/p1_status.sh`: compact status report for mounts, GPU, watcher PIDs,
  and completed horizon sensitivity outputs.

The earlier umbrella runner `scripts/run_p1_full_gpu_experiments.sh` remains as
an orchestration reference, but long-lived waiting jobs should use the
per-job watcher above.

| Follow-up | Purpose | Status |
|---|---|---|
| nuPlan 50k `wm_decoupled(+vis)` 3 seeds | Fill the missing visibility arm at 50k. | Complete: Return -0.33 ± 4.94, CollRate 0.007 ± 0.012; no-vis remains cleaner by return. |
| Cross-dataset eval | Check whether ranking reversal follows architecture or dataset distribution. | Complete under `experiments/cross_dataset_eval/`. |
| Fig 4b lane-conflict imagination | Finish P1 case-study set. | Complete under `experiments/figures/case_studies/`; lane-conflict collision 0.246 vs 0.652. |
| nuScenes K=3/5/7 horizon sensitivity | Test whether decoupled drift compounds with rollout length. | Complete under `experiments/horizon_sensitivity_nuscenes/`; collision gap grows from +0.113 at K=3 to +0.306 at K=7. |
