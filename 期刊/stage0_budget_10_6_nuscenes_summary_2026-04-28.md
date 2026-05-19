# Stage0 typed-budget sensitivity 10/6 summary

Setup: nuScenes 700 scenes, `object_relation_decoupled_visibility`, `top_k_dyn=10`, `top_k_rel=6`, seeds 7/42/2026, 15 epochs, batch 32. This is an appendix sensitivity point for the 12/4 typed budget.

| budget | Dyn Rollout MSE ↓ | Action MSE ↓ | Collision F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|
| 10/6 | 8.761 +/- 1.067 | 0.286 +/- 0.009 | 0.894 +/- 0.007 | 1.574 +/- 0.485 | 0.884 +/- 0.043 |
| 12/4 main | 1.876 +/- 0.227 | 0.284 +/- 0.023 | 0.926 +/- 0.029 | 0.520 +/- 0.049 | 0.979 +/- 0.008 |

Per-seed raw values:

| seed | Dyn Rollout MSE | Action MSE | Collision F1 | Rare ADE | IntRec@1m |
|---:|---:|---:|---:|---:|---:|
| 7 | 8.810 | 0.291 | 0.896 | 2.045 | 0.834 |
| 42 | 9.803 | 0.275 | 0.887 | 1.602 | 0.913 |
| 2026 | 7.671 | 0.292 | 0.900 | 1.077 | 0.904 |

Quick read:

- The relation-heavy 10/6 budget is runnable and keeps high interaction recall (0.884 +/- 0.043), but it is weaker than the 12/4 main setting on dynamic rollout and rare-agent ADE.
- This supports treating 12/4 as a robust default rather than an arbitrary one-off budget choice.
- Keep this in appendix; it does not change the main Stage0/Stage1 story.
