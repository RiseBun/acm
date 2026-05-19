# nuScenes vs nuPlan Token/Agent Statistics

Purpose: explain why representation quality, Stage-1 policy learning, and downstream planner-like behavior need not rank variants identically.

| Statistic | nuScenes 700 scenes | nuPlan 50k NPZ | Reading |
|---|---:|---:|---|
| Dynamic tokens / sample | 9.715 ± 3.847 / p90 13.000 | 12.164 ± 2.457 / p90 13.000 | planning density / interaction budget |
| Rare tokens / sample | 2.439 ± 2.906 / p90 7.000 | 3.934 ± 3.490 / p90 9.000 | pedestrian/cyclist pressure |
| Relation tokens / sample | 11.318 ± 2.118 / p90 12.000 | 11.164 ± 2.457 / p90 12.000 | relation-context availability |
| Dynamic visibility | 0.746 ± 0.265 / p90 1.000 | 1.000 ± 0.000 / p90 1.000 | whether visibility weighting has signal |
| Relation TTC | 15.363 ± 7.210 / p90 20.000 | 16.727 ± 6.124 / p90 20.000 | risk/interaction feature scale |
| Teacher action L2 | 0.539 ± 0.632 / p90 1.183 | 3.420 ± 4.178 / p90 10.275 | action-label scale and policy target |
| Ego next displacement | 0.000 ± 0.000 / p90 0.000 | 0.342 ± 0.418 / p90 1.027 | short-horizon motion scale |
| Dynamic next displacement | 0.000 ± 0.000 / p90 0.000 | 0.028 ± 0.152 / p90 0.000 | world-model target scale |

## Takeaways

- Use these statistics as explanatory evidence, not as performance metrics.
- The main question is whether nuPlan has denser or cleaner relation/action structure, which can make decoupled relation-aware abstraction more useful for policy learning.
- Visibility should be interpreted as a dataset-specific inductive bias: if its distribution carries little contrast or interacts poorly with action labels, no-vis can be better.
