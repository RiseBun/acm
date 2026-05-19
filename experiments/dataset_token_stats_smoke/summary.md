# nuScenes vs nuPlan Token/Agent Statistics

Purpose: explain why representation quality, Stage-1 policy learning, and downstream planner-like behavior need not rank variants identically.

| Statistic | nuScenes 700 scenes | nuPlan 50k NPZ | Reading |
|---|---:|---:|---|
| Dynamic tokens / sample | 7.754 ± 4.042 / p90 13.000 | 12.293 ± 2.244 / p90 13.000 | planning density / interaction budget |
| Rare tokens / sample | 3.262 ± 3.438 / p90 9.000 | 3.977 ± 3.576 / p90 9.000 | pedestrian/cyclist pressure |
| Relation tokens / sample | 10.314 ± 3.180 / p90 12.000 | 11.293 ± 2.244 / p90 12.000 | relation-context availability |
| Dynamic visibility | 0.738 ± 0.280 / p90 1.000 | 1.000 ± 0.000 / p90 1.000 | whether visibility weighting has signal |
| Relation TTC | 14.078 ± 7.775 / p90 20.000 | 16.801 ± 6.106 / p90 20.000 | risk/interaction feature scale |
| Teacher action L2 | 0.485 ± 0.343 / p90 0.957 | 3.399 ± 4.300 / p90 10.235 | action-label scale and policy target |
| Ego next displacement | 0.000 ± 0.000 / p90 0.000 | 0.340 ± 0.430 / p90 1.023 | short-horizon motion scale |
| Dynamic next displacement | 0.000 ± 0.000 / p90 0.000 | 0.028 ± 0.154 / p90 0.000 | world-model target scale |

## Takeaways

- Use these statistics as explanatory evidence, not as performance metrics.
- The main question is whether nuPlan has denser or cleaner relation/action structure, which can make decoupled relation-aware abstraction more useful for policy learning.
- Visibility should be interpreted as a dataset-specific inductive bias: if its distribution carries little contrast or interacts poorly with action labels, no-vis can be better.
