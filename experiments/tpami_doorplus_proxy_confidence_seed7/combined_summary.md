# DOOR+ Proxy Confidence Seed7 Summary

Evaluation on K64 Stage-1 DOOR checkpoint, seed7, max_val_samples=1000, corruptions clean/relfp0.2.

| family | confidence | relfp return delta | clean CollRate | relfp CollRate | relfp CollMean | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| door | baseline | -3.772 | 0.457 | 0.714 | 0.719 | 0.065 | 0.574 | 0.015 | 0.128 |
| doorplus | normal | 0.118 | 0.488 | 0.488 | 0.526 | 0.007 | 0.569 | 0.018 | 0.130 |
| doorplus | proxy | -1.320 | 0.457 | 0.495 | 0.527 | 0.040 | 0.569 | 0.042 | 0.130 |
| doorplus | constant | -5.610 | 0.486 | 0.805 | 0.809 | 0.058 | 0.569 | 0.015 | 0.130 |
| doorplus | shuffled | -4.597 | 0.487 | 0.797 | 0.800 | 0.069 | 0.559 | 0.022 | 0.130 |
| doorplus | inverted | -6.031 | 0.426 | 0.803 | 0.795 | 0.545 | 0.569 | 0.006 | 0.130 |
| doorplus | proxy_inverted | -5.256 | 0.419 | 0.737 | 0.729 | 0.406 | 0.569 | 0.015 | 0.130 |
