# Paper Figures Index

All figures live under `experiments/figures/case_studies/{pdf,png}`. Each
PDF is paper-grade (vector, no rasterisation); the PNG mirrors are for
quick previews and inline embedding in slide decks / Notion. Figures are
deterministic given the seed-7 split, the fixed pickle, and the saved
checkpoints. Use the documented invocation under each figure to
reproduce.

| ID  | Title                                                          | Story (one line)                                                    | Status   | Paper §                                | Source script                                              |
|-----|----------------------------------------------------------------|---------------------------------------------------------------------|----------|----------------------------------------|------------------------------------------------------------|
| F1  | Naive shared top-k drops critical agents                       | Object+Relation-16 (naive) over-allocates to relations and misses near-field dynamic agents that decoupled keeps. | ready    | Method §3 / Stage 0 results §4         | `scripts/plot_paper_fig_slot_compare.py`                   |
| F2  | (combined with F1)                                             | Same scene, decoupled typed-budget keeps the missed agents.        | ready    | Method §3 / Stage 0 results §4         | `scripts/plot_paper_fig_slot_compare.py`                   |
| F3  | nuScenes Stage 1 imagination                                   | Object-only stays bounded; decoupled+visibility drifts and saturates over the 5-step rollout. | ready (5 622 val) | Stage 1 results §5 / discussion §12.1 | `scripts/plot_paper_fig_imagination_nuscenes.py`           |
| F4a | nuPlan Stage 1 — subset bar chart                              | Decoupled (no vis) wins on every val subset; gap concentrates on `lane_conflict` and `low_ttc_proxy`. | ready (3 seeds × 6 699 lc) | Cross-dataset analysis §11.x / §12.1   | `scripts/plot_paper_fig_nuplan_subsets.py`                 |
| F4b | nuPlan Stage 1 — per-step on `lane_conflict`                   | Per-rollout-step traces show object-only's collision saturating while decoupled stays bounded on lane_conflict samples. | **pending** — `e2e-nuplan` FUSE mount was disconnected (`Transport endpoint is not connected`) when the figure was due. The script is fully written and tested; rerun the command below once the mount is restored. | Cross-dataset analysis §11.x / §12.1   | `scripts/plot_paper_fig_imagination_nuplan.py`             |

## Reproduce

All commands assume the working directory is the repo root and that the
seed-7 checkpoints / pickle are present (see Stage 0 / Stage 1 docs).

```bash
# F1 + F2 (combined): the four ranked cases, ordered by
# (#dec_sel - #naive_sel) on the seed-7 nuScenes val set, with an
# honesty constraint that decoupled's near-field next-token error
# is at least 1.2x lower than naive's. The four cases produced are
#   case00_idx80  -> naive 0/7,  err 7.46m -> 0.26m  (29× ratio)  [paper main]
#   case01_idx88  -> naive 1/6,  err 3.46m -> 0.20m  (18× ratio)  [paper alt]
#   case02_idx81  -> naive 1/8,  err 5.60m -> 0.25m  (23× ratio)
#   case03_idx85  -> naive 3/8,  err 2.70m -> 0.22m  (12× ratio)
PYTHONPATH=src python scripts/plot_paper_fig_slot_compare.py \
    --sample-indices 80 88 81 85

# Or auto-rank by the differentiation score (returns the same set).
PYTHONPATH=src python scripts/plot_paper_fig_slot_compare.py --top-k 4

# F3: aggregate + 3 picked samples on the full seed-7 nuScenes val
# (5,622 samples, 26 s with the local token cache).
PYTHONPATH=src python scripts/plot_paper_fig_imagination_nuscenes.py \
    --num-scenes 700 --batch-size 64

# F4a: bar chart from the existing 3-seed nuPlan-50k subset summary.
python scripts/plot_paper_fig_nuplan_subsets.py

# F4b: per-step traces on lane_conflict subset (requires the
# /mnt/datasets/e2e-nuplan FUSE mount to be live).
PYTHONPATH=src python scripts/plot_paper_fig_imagination_nuplan.py \
    --max-val-samples 5000
```

For the paper, the recommended top-line picks are:

* `pdf/fig1_main_naive_drops_decoupled_keeps.pdf` (= case00_idx80, 29× ratio, naive selected zero of 7 near-field pedestrians)
* `pdf/fig1_alt_naive_drops_decoupled_keeps.pdf`  (= case01_idx88, 18× ratio, cleaner mid-density crosswalk scene)
* `pdf/fig3_imagination_nuscenes.pdf`
* `pdf/fig4a_nuplan_subset_bars.pdf`

## Underlying numerics (audit trail)

The two `imagine_trajectory`-based figures also dump their per-sample
per-step tensors as `.npz` next to the PDFs, for reviewer audit:

* `experiments/figures/case_studies/fig3_imagination_nuscenes_metrics.npz`
* `experiments/figures/case_studies/fig3_imagination_nuscenes_summary.json`
* `experiments/figures/case_studies/fig4b_imagination_nuplan_metrics.npz` (after F4b)
* `experiments/figures/case_studies/fig4b_imagination_nuplan_summary.json` (after F4b)

The per-rollout-step structure inside the npz is

    {cond}__coll          [N, K]  sigmoid(collision_t)
    {cond}__action_norm   [N, K]  ||a_mean_t||_2
    {cond}__ego_cos_step  [N, K]  1 - cos(e_t, e_{t+1})
    {cond}__ret_cumsum    [N, K]  cumulative reward through step k
    {cond}__return_sum    [N]
    {cond}__coll_max      [N]
    picks_idx             [3]     selected sample indices used in row 2
    picks_label           [3]     labels for each pick

For F4b the npz also includes

    lane_conflict_mask    [N]     boolean mask used to subset row 1

## Selection criteria summary

| Figure | Criterion                                                                               |
|--------|------------------------------------------------------------------------------------------|
| F1+F2  | `#dec_near_sel − #naive_near_sel >= 2` AND `naive_err >= 1.2 × dec_err`, ranked desc.    |
| F3     | aggregate over the full seed-7 val (5,622 samples). Picks: `decoupled-collapse`, `object-advantage`, `typical`. |
| F4a    | aggregate from `experiments/nuplan_interaction_subset_50k/summary.json` (3 seeds × 6,699 lane_conflict samples each). |
| F4b    | aggregate over `lane_conflict` subset of the seed-7 nuPlan val. Picks: `decoupled-wins`, `object-collides`, `typical`. |

## Notes

1. **`Object+Relation-16 (naive)` slot count** — F1 / F2 quote
   `selected K / N near-field`, where `K` is the number of `selected_indices`
   in the abstraction whose corresponding token is a dynamic GT agent
   inside the 18 m near-field circle. The remaining (16 − K) selected
   slots are RELATION / MAP / outside-the-circle dynamic tokens, which
   is exactly the failure mode the figure visualises.

2. **`pred err` numerics** — averaged distance between each near-field
   GT agent's next-frame position and the closest predicted next-token
   among the variant's selected slots. Lower is better.

3. **Why both with and without visibility for F3?** The
   `wm_decoupled` condition is the default Stage 1 decoupled run, which
   uses the visibility multiplication on the dyn path. F3 highlights
   that visibility weighting (a Stage 0 stabiliser) becomes a
   destabiliser under K-step imagination on nuScenes; this is the
   ranking reversal we explain in §12.1. The variant `wm_decoupled_no_vis`
   is included only as an extra trace if the user passes
   `--conditions wm_object wm_decoupled wm_decoupled_no_vis`.

4. **Cross-dataset story.** F3 (nuScenes) and F4a/F4b (nuPlan) together
   show the most surprising finding: the same decoupled abstraction
   that saturates on nuScenes Stage 1 dominates on nuPlan, where the
   simpler object-only collapses on `low_ttc_proxy` (collision rate
   0.79 vs decoupled's 0.54). This is the "ranking reversal" that
   §12.1 dissects.
