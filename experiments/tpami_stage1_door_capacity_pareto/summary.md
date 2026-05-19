# DOOR Stage-1 Capacity Pareto Summary

Source: Stage-1 nuPlan 5k, horizon=10, seeds 7/42/123; latency benchmark uses synthetic batch H=20 rollout for repeated-rollout cost.

| K | Split | Return | CollRate | CollMean | H=20 rollout ms | Peak rollout MB | Cost reading |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 14/2 | -11.96 +/- 7.67 | 0.815 +/- 0.050 | 0.811 +/- 0.055 | 109.8 | 121.9 | too constrained |
| 32 | 24/8 | +1.17 +/- 3.36 | 0.864 +/- 0.036 | 0.861 +/- 0.032 | 107.9 | 188.0 | compact trade-off |
| 64 | 52/12 | +14.88 +/- 0.78 | 0.548 +/- 0.007 | 0.556 +/- 0.010 | 109.2 | 329.5 | stronger but costly |

## Paper Reading

- K=16 is too constrained: typed allocation helps the representation story, but downstream policy remains capacity-limited.
- K=32 is the compact trade-off: return becomes positive relative to K=16, but collision stability has not reached K=64.
- K=64 is stronger downstream, but rollout memory rises substantially, supporting the Pareto/cost argument rather than a claim that K=32 is globally optimal.

## Suggested Main-Text Claim

Capacity matters twice: it first determines representation sufficiency and then constrains downstream imagination-policy utility. DOOR improves compact-budget object-relation allocation, while larger K further improves policy metrics at higher repeated-rollout cost.
