# DOOR Stage-1 Capacity Pareto Summary (H=10 Cost)

Source: Stage-1 nuPlan 5k, horizon=10, seeds 7/42/123; cost benchmark uses synthetic batch size 128 and H=10 rollout to match the policy study.

| K | Split | Return | CollRate | CollMean | H=10 rollout ms | Peak H=10 rollout MB | Cost reading |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 14/2 | -11.96 +/- 7.67 | 0.815 +/- 0.050 | 0.811 +/- 0.055 | 55.6 | 110.9 | too constrained |
| 32 | 24/8 | +1.17 +/- 3.36 | 0.864 +/- 0.036 | 0.861 +/- 0.032 | 56.4 | 166.8 | compact trade-off |
| 64 | 52/12 | +14.88 +/- 0.78 | 0.548 +/- 0.007 | 0.556 +/- 0.010 | 55.4 | 288.3 | stronger but costly |

## Paper Reading

- K=16 is too constrained: typed allocation helps representation, but downstream policy remains capacity-limited.
- K=32 is the compact trade-off: return becomes positive relative to K=16, but collision stability has not reached K=64.
- K=64 is stronger downstream, but H=10 rollout memory rises to 2.6x K=16, supporting the Pareto/cost argument rather than a claim that K=32 is globally optimal.
