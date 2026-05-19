# DOOR+ Proxy Confidence Multiseed Summary

Evaluation on Stage-1 DOOR checkpoints, seeds `7/42/123`, max_val_samples=1000, corruptions `clean/relfp0.2`, confidence_mode=`proxy`.

| method | relfp return delta | clean CollRate | relfp CollRate | relfp CollRate increase | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DOOR baseline | -0.866 | 0.287 ± 0.053 | 0.475 ± 0.128 | 0.188 | 0.139 ± 0.066 | 0.566 | 0.065 ± 0.014 | 0.132 |
| DOOR+ proxy | -1.548 | 0.308 ± 0.064 | 0.511 ± 0.328 | 0.203 | 0.052 ± 0.004 | 0.561 | 0.060 ± 0.011 | 0.132 |

Key readings:

- DOOR+ proxy reduces selected FP relations by `62.7%` relative to baseline DOOR under `relfp0.2` (`0.139` -> `0.052`).
- Clean collision rate remains comparable (`0.287` baseline vs `0.308` DOOR+ proxy).
- Collision-rate increase under relation false positives is similar in mean (`0.188` baseline vs `0.203` DOOR+), with high across-seed variance; the robust claim should focus on relation-selection reliability rather than closed-loop superiority.
- Selected true-risk relation retention is close (`0.065` baseline vs `0.060` DOOR+), so FP suppression is not achieved by dropping all relation tokens.

Recommended paper usage: report this as a multiseed reliability diagnostic for DOOR+ proxy confidence, not as a policy-performance win.
