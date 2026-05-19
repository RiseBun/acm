# TPAMI DOOR+ uncertainty-aware relation gating

This note records the first DOOR+ probe. It is an evaluation-only, post-hoc
uncertainty-gating experiment: no model is retrained. `doorplus_posthoc` reuses
the trained DOOR checkpoint (`wm_decoupled_no_vis`) and applies an oracle
low-confidence gate to artificial false-positive relation tokens before
selection and latent imagination.

## Source

- Script: `scripts/doorplus_uncertainty_gating.py`
- Main h10 output: `experiments/tpami_doorplus_gating_h10_10k/summary.md`
- Smoke output: `experiments/tpami_doorplus_gating_smoke_500/summary.md`
- Checkpoints: `experiments/nuplan_stage1_50k/seed*/wm_decoupled_no_vis/model.pt`

## Setup

- Dataset: nuPlan 50k validation split.
- Seeds: 7, 42, 123.
- Samples: 10k validation samples per seed.
- Horizon: 10.
- Corruptions: `clean`, `loc1.5`, `miss0.2`, `relfp0.2`.
- DOOR+ gate: false-positive relation confidence `0.05`, safe TTC `10.0`.

## Main h10 result

| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K |
|---|---|---:|---:|---:|---:|---:|
| DOOR | clean | 33.758 +/- 7.204 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 |
| DOOR | loc1.5 | 33.735 +/- 7.215 | 0.262 +/- 0.045 | 0.280 +/- 0.033 | 6.628 +/- 0.110 | 0.000 |
| DOOR | miss0.2 | 30.567 +/- 5.529 | 0.247 +/- 0.033 | 0.260 +/- 0.023 | 6.633 +/- 0.115 | 0.000 |
| DOOR | relfp0.2 | 32.847 +/- 6.921 | 0.457 +/- 0.126 | 0.469 +/- 0.116 | 6.727 +/- 0.250 | 0.150 +/- 0.074 |
| DOOR+ posthoc | clean | 33.749 +/- 7.212 | 0.260 +/- 0.045 | 0.277 +/- 0.033 | 6.628 +/- 0.110 | 0.000 |
| DOOR+ posthoc | loc1.5 | 33.737 +/- 7.213 | 0.263 +/- 0.045 | 0.280 +/- 0.032 | 6.628 +/- 0.110 | 0.000 |
| DOOR+ posthoc | miss0.2 | 30.537 +/- 5.578 | 0.246 +/- 0.031 | 0.260 +/- 0.022 | 6.633 +/- 0.116 | 0.000 |
| DOOR+ posthoc | relfp0.2 | 33.793 +/- 7.236 | 0.242 +/- 0.046 | 0.261 +/- 0.032 | 6.627 +/- 0.110 | 0.111 +/- 0.018 |

## Reading

The post-hoc DOOR+ gate leaves clean, localization-noise, and missing-token
metrics essentially unchanged. Under high-risk false-positive relation noise,
it reduces CollRate from `0.457` to `0.242` and CollMean from `0.469` to
`0.261`, while also lowering Selected-FP-Rel@K from `0.150` to `0.111`.

## Long-horizon confirmation

The same clean-vs-false-positive probe was repeated at horizons 20 and 50.

| Horizon | condition | clean CollRate | relfp0.2 CollRate | relfp0.2 CollMean | Selected-FP-Rel@K |
|---:|---|---:|---:|---:|---:|
| 10 | DOOR | 0.260 +/- 0.045 | 0.457 +/- 0.126 | 0.469 +/- 0.116 | 0.150 +/- 0.074 |
| 10 | DOOR+ posthoc | 0.260 +/- 0.045 | 0.242 +/- 0.046 | 0.261 +/- 0.032 | 0.111 +/- 0.018 |
| 20 | DOOR | 0.260 +/- 0.045 | 0.458 +/- 0.125 | 0.469 +/- 0.116 | 0.150 +/- 0.074 |
| 20 | DOOR+ posthoc | 0.260 +/- 0.045 | 0.242 +/- 0.046 | 0.261 +/- 0.032 | 0.111 +/- 0.018 |
| 50 | DOOR | 0.260 +/- 0.045 | 0.457 +/- 0.126 | 0.469 +/- 0.116 | 0.150 +/- 0.074 |
| 50 | DOOR+ posthoc | 0.260 +/- 0.045 | 0.243 +/- 0.046 | 0.261 +/- 0.032 | 0.111 +/- 0.018 |

Source summaries:

- `experiments/tpami_doorplus_gating_h10_10k/summary.md`
- `experiments/tpami_doorplus_gating_h20_10k/summary.md`
- `experiments/tpami_doorplus_gating_h50_10k/summary.md`

This supports the TPAMI three-stage story:

1. Typed budget solves inter-type competition between dynamic and relation
   tokens.
2. Relation utility is concentrated in risk/TTC semantics.
3. High-risk relation false positives are the remaining failure mode, and
   uncertainty-aware relation gating mitigates it without sacrificing the clean
   typed-budget behavior.

This is not yet a fully trained DOOR+ model. The safe wording is:

> A post-hoc uncertainty gate shows that suppressing low-confidence high-risk
> relation tokens can recover the false-positive relation failure mode while
> preserving clean-input performance.

The h20/h50 checks confirm that the improvement is not a short-horizon artifact.
The next step is to train a detector-confidence or learned-confidence version,
so the main paper can move from post-hoc oracle gating to a deployable DOOR+
model.
