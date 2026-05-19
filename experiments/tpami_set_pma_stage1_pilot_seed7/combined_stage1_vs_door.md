# Set-PMA Stage-1 Pilot vs DOOR

nuPlan 5k, seed7, horizon=10, 8 epochs. Set-PMA is warm-started from the corresponding Stage-0 K checkpoint; DOOR rows reuse existing clean K-scaling results.

| K | model | return ↑ | collision rate ↓ | collision mean ↓ | rollout stability ↓ | global stability ↓ | n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | DOOR | 4.542 | 0.825 | 0.825 | 0.002 | 0.008 | 1000 |
| 32 | Set-PMA | 4.032 | 0.000 | 0.074 | 0.000 | 0.000 | 1000 |
| 64 | DOOR | 15.781 | 0.549 | 0.562 | 0.000 | 0.007 | 1000 |
| 64 | Set-PMA | -15.059 | 0.295 | 0.375 | 0.000 | 0.000 | 1000 |
