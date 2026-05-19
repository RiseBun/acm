# NeurIPS Closure Artifacts

This folder contains paper-ready artifacts generated from existing experiments. No model was trained or evaluated by this script.

## Generated Figures

- `figures/fig_selection_diagnostics_bars.{png,pdf}`
- `figures/fig_missrate_vs_interaction_recall.{png,pdf}`
- `figures/fig_relation_semantics_ablation.{png,pdf}`
- `figures/fig_budget_sensitivity_12_4_vs_10_6.{png,pdf}`

## Selection Diagnostic Takeaways

| condition | CDR mean | MissRate mean | WastedRel mean | ROI mean | MissRate~IntRec rho |
|---|---:|---:|---:|---:|---:|
| wm_naive | 0.547 | 0.453 | 0.216 | 0.054 | -0.680 |
| wm_object | 0.785 | 0.215 | N/A | 0.000 | -0.519 |
| wm_decoupled_no_vis | 0.939 | 0.061 | 0.044 | 0.260 | -0.404 |

## Relation Semantic Ablation

- Baseline collision rate: 0.260
- Removing TTC/risk collision rate: 0.895
- Removing lane/priority collision rate: 0.257

## External-Style BC Anchor

- BC latent return: 0.433 +/- 4.230
- BC teacher action MSE: 8.824 +/- 0.113

## Remaining Optional Experiment

- `wm_naive` nuPlan 50k Stage1 training remains optional. Current Stage0 50k artifacts do not include an `object_relation/model.pt` warm-start, so this job would likely train from scratch or require first producing the Stage0 warm-start.
