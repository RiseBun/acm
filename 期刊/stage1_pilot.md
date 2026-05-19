## Stage 1 — Imagination RL Pilot (nuScenes)

_Last updated: 2026-04-27 — added FlowPlanner `comfort_423v1` comfort-metric pilot summary._

> Canonical status: this file is now a nuScenes Stage-1 historical/pilot note.
> Use `docs/experiment_report.md` as the canonical cross-dataset experiment
> report, and `docs/discarded_vs_reliable.md` as the compact reliability /
> discarded-conclusion index. The main paper figures are indexed in
> `docs/figures.md`.

---

### 0. TL;DR (current state, 3 seeds plus one targeted ablation)

After the NaN/Huber/cosine-stability fixes (the **A+B patch**), the v3 hparams
(`entropy_beta=0.003`, `action_clip=5`) were verified across **3 seeds**
(7, 42, 123) on nuScenes 700 scenes, K=5 imagination, 10 epochs. After X,
we also ran the targeted typed-budget ablation `wm_decoupled_14_2`
(`top_k_dyn=14`, `top_k_rel=2`) to test whether the default 12+4 relation
budget was the destabiliser.

| condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2)\* |
|---|---:|---:|---:|---:|---:|
| **wm_object** (object-only)        | **31.79 ± 19.70** | **0.597 ± 0.048** | 0.610 ± 0.048 | 0.258 ± 0.058 | 0.170 ± 0.034 |
| wm_decoupled_rel_to_critic_only    | 5.72 ± 7.25       | 0.729 ± 0.169     | 0.736 ± 0.163 | 0.453 ± 0.228 | 0.170 ± 0.036 |
| wm_decoupled (decoupled + vis)     | 4.34 ± 13.66      | 0.695 ± 0.283     | 0.676 ± 0.243 | 0.636 ± 0.108 | 0.056 ± 0.001 |
| wm_decoupled_14_2 (decoupled + vis) | 2.47 ± 4.14       | 0.808 ± 0.196     | 0.774 ± 0.189 | 0.419 ± 0.264 | 0.125 ± 0.068 |
| wm_decoupled_no_vis (decoupled)    | 0.34 ± 15.97      | 0.820 ± 0.260     | 0.814 ± 0.167 | 0.223 ± 0.033 | 0.043 ± 0.015 |

Reading the table:

1. **`wm_object` is the strongest and most stable Stage-1 baseline** under the current setup. All 3 seeds give positive return (21.8 / 54.5 / 19.1), CollRate std is only 0.05.
2. **`wm_decoupled (+vis)` is high-variance**: 1/3 seeds is competitive (return 19.9, coll 0.37), 2/3 collapse (return ≈ −5, coll ≈ 0.85). CollRate std is **0.28** — almost the entire spread of the metric.
3. **`wm_decoupled_no_vis` is uniformly worse**, all 3 seeds. Visibility weighting is **necessary** for the decoupled variant to even be competitive on a single seed.
4. **`wm_decoupled_14_2` does not recover Stage-1 learning.** Reducing the rel budget from 4 to 2 makes CollRate worse on average than default 12+4 and still leaves a large gap to `wm_object`.
5. **`wm_decoupled_rel_to_critic_only` helps only partially.** It improves Return over full decoupled and reduces variance versus the default 12+4 run, but CollRate remains high and the gap to `wm_object` is still large.

\* `Stab(L2)` (relative-L2 on `global_latent`) is *not* cross-variant comparable — kept only for backward compatibility. Use `Stab(ego-cos)` as a sanity check, not as a quality metric (see §5).

---

### 1. What still holds, what no longer holds

#### What still holds

- **Stage 0: decoupled wins on representation quality.** The Table-3
  results in `docs/stage0.md` (decoupled-vs-holistic, decoupled-vs-naïve)
  are unchanged. Decoupled (with or without visibility) gives higher
  next-token reconstruction quality, lower reward MSE, and the typed-budget
  argument for slot allocation.
- **Visibility weighting is necessary for `wm_decoupled` Stage-1 stability.**
  Removing it (the `wm_decoupled_no_vis` ablation, X seeds 7/42/123) makes
  the policy collapse on **all** seeds: CollRate ≥ 0.52 in the best case,
  and ≥ 0.97 in 2 of 3 seeds. The training log shows the same
  saturation pattern (`|a|max=5.0`, `log_std=+0.50`, `Cmax≥0.94` from
  epoch 3) that we used to see on un-fixed `wm_object`.
- **The A+B numerical fixes hold.** No NaN, no critic divergence, no WM
  corruption across all 9 X runs. Sanity loss stays < 1.0 for every seed
  and every condition for all 10 epochs.

#### What no longer holds

- **"Decoupled outperforms object-only in Stage 1."** This claim was
  based on a single seed (7) and a single hparams setting (v2). With v3
  hparams across 3 seeds, the ranking *reverses*: `wm_object` beats
  `wm_decoupled` on both Return (31.79 vs 4.34) and CollRate (0.597 vs
  0.695), and is dramatically more stable (Return std 19.7 vs 13.7,
  CollRate std 0.05 vs 0.28).
- **The early pilot's qualitative story was not robust.** Even the v3
  single-seed-7 numbers do not reproduce under multi-seed parallel
  execution: seed-7 Return for `wm_decoupled` was 19.9 in v3, but
  −1.45 when re-run under X. Single-seed Stage-1 numbers from this
  setup should be treated as noise floor, not signal.

#### Current best factual conclusion

- Under the current setup (K=5, GAE-λ actor-critic, Huber critic δ=10,
  detached WM, sanity loss 1.0, entropy β=0.003, action clip 5,
  warm-start from same-variant Stage-0 checkpoint), **the object-only
  abstraction yields the strongest and most stable Stage-1 policy**.
- The decoupled abstraction, despite winning Stage 0 cleanly, **does
  not currently transfer to a better Stage-1 policy**. It is
  high-variance across seeds, and the `wm_object` average dominates
  both default `wm_decoupled` and the 14+2 budget ablation on the
  primary metrics.

#### Likely gap (not yet measured, do not over-claim)

- **Representation quality ≠ policy-learning friendliness.** Whatever
  makes the decoupled latent a better predictor of next tokens does
  not (yet) make it a better state for an actor-critic to consume.
- **Visibility is not the simple explanation.** Removing visibility makes
  the decoupled variant worse, so the Stage-1 gap is not caused by
  visibility weighting alone.
- **Relation budget is not the simple explanation.** Reducing the rel
  budget from 4 to 2 (`wm_decoupled_14_2`) does not close the gap to
  `wm_object`; it worsens CollRate on average.
- **Remaining suspect 1: rel-branch fusion.** The decoupled-with-visibility
  variant fuses a visibility-weighted dyn latent with a rel latent before
  actor-critic learning. That fused state may be useful for representation
  prediction but unfriendly to the critic/value landscape or to stable
  policy gradients.
- **Fusion path matters, but critic-only is not sufficient.**
  `wm_decoupled_rel_to_critic_only` improves mean Return over full
  decoupled (5.72 vs 4.34) and has lower Return/CollRate variance, but
  it does not close the CollRate gap to `wm_object` (0.729 vs 0.597).
  This means naive actor fusion is not the only issue.
- **Remaining suspect 2: imagination-time selection drift.** The
  imagination rollout has no mechanism to keep the rel branch's slot
  selection consistent across steps, so relation context that helps static
  Stage-0 prediction may become unstable during multi-step policy learning.

These are still hypotheses, but the 14+2 result narrows the failure mode:
the Stage-1 problem is now more plausibly about **fusion / rollout mismatch**
than about a too-large relation budget.

---

### 2. Pipeline (unchanged from A+B patch)

| | |
|---|---|
| Dataset | nuScenes 700 scenes, scene-level 80/20 split, val = 5 622 samples |
| Imagination horizon K | 5 |
| Actor loss | -E[log π · stop_grad(adv)] − β·H(π), **β=0.003 (v3)** |
| Critic loss | **Huber(δ=10)** on stop_grad(GAE return) |
| Sanity loss | full Stage-0 losses on the real t=0 batch, weight=1.0 |
| GAE | γ=0.97, λ=0.95, discount-mask = sigmoid(continue) |
| Reward | `w_prog=1`, `w_coll=5`, `w_act=0.01`, clipped to [-5, 5] |
| Action head | `mean = 3·tanh(raw/3)`, `log_std ∈ [-2, 0.5]`, sample clipped to **[-5, 5] (v3)** |
| `detach_world_model` | True |
| Epochs / batch / lr | 10 / 128 / 4·base = 4e-3 |
| Seeds | **7, 42, 123** (X verification) |
| Warm-start | Stage 0 same-variant checkpoint (e.g. `experiments/table3_fair_fix2_seed7/<variant>/model.pt`) |

Runner: `scripts/run_stage1_pilot_x.sh` (X multi-seed, parallel),
`scripts/run_stage1_pilot_v3.sh` (single-seed v3 baseline),
`scripts/run_stage1_pilot_y.sh` (visibility ablation).

Output: `experiments/stage1_pilot_x/seed{7,42,123}/<cond>/stage1_metrics.json`,
aggregate `experiments/stage1_pilot_x/X_summary.{md,json}`.

---

### 3. X — per-seed raw values (re-grouped for readability)

```
                       seed 7         seed 42        seed 123        mean ± std
wm_object              R=21.79        R=54.48        R=19.10         R=31.79 ± 19.70
                       Coll=0.602     Coll=0.547     Coll=0.643      Coll=0.597 ± 0.048
                       Stab=0.193     Stab=0.302     Stab=0.279      Stab=0.258 ± 0.058

wm_decoupled (+vis)    R=−1.45        R=19.94        R=−5.48         R=4.34 ± 13.66
                       Coll=0.837     Coll=0.369     Coll=0.878      Coll=0.695 ± 0.283
                       Stab=0.727     Stab=0.517     Stab=0.666      Stab=0.636 ± 0.108

wm_decoupled_no_vis    R=−9.79        R=18.75        R=−7.94         R=0.34 ± 15.97
                       Coll=0.970     Coll=0.970     Coll=0.520      Coll=0.820 ± 0.260
                       Stab=0.187     Stab=0.253     Stab=0.228      Stab=0.223 ± 0.033
```

Key observations:

- `wm_object`: `Coll` band [0.547, 0.643], width 0.096. Tightest of all.
- `wm_decoupled (+vis)`: `Coll` band [0.369, 0.878], width 0.509. ~5× wider.
- `wm_decoupled_no_vis`: `Coll` band [0.520, 0.970], width 0.450. Same magnitude of variance, *uniformly worse mean*.

The single-seed-good case (`wm_decoupled @ seed 42`, R=19.94, Coll=0.369)
is genuinely better than `wm_object @ seed 42` (R=54.48 but Coll=0.547),
but only on collision and only on this one seed. There is currently no
seed-independent way to tell when this regime triggers.

---

### 4. History — A+B patch and the discarded v1 narrative

These notes are kept so future readers don't repeat the same mistakes,
not because the v1/v2 conclusions still hold.

**v1 (pre-fix), seed 7.** `wm_object` reported R=26.85 — exceeded the
reward-clip ceiling (5 × K=5 = 25). That was reward-hacking via an
about-to-diverge critic, **not** a real policy. The v1 "decoupled
beats object-only" reading was dependent on this artifact.

**v2 (after action `tanh` + `log_std` clamp + reward clip + detach WM
+ Huber critic δ=10), seed 7.** `wm_object` and `wm_decoupled` both
trained without NaN. Single-seed comparison gave `wm_decoupled` a
positive return and a 23-point lower collision rate. We took this as
"decoupled helps Stage 1." X showed this was seed-7-specific.

**v3 (v2 + entropy β 0.01→0.003 + action clip 8→5), seed 7.** Both
conditions reached R≈19.9, CollRate 0.37 vs 0.66. We took this as
"object-only catches up but decoupled still better on collision." X
showed this was also seed-7-specific.

**Y (decoupled without visibility), seed 7 (v3 hparams).** Hypothesised
"visibility weighting injects multiplicative latent noise in
imagination → hurts policy". Result: the policy *collapsed*
(R=1.79, Coll=0.99, `|a|max=5.0` and `log_std=+0.50` saturated from
epoch 3). Hypothesis refuted — visibility is a stabilizer, not a
noise source.

**X (3 seeds × 3 conditions, v3 hparams).** Confirmed Y on all seeds,
and reversed the v2/v3 single-seed claim. Current state of knowledge.

---

### 5. Stability metric (cosine, ego slot) — when to read it

`rollout_stability` is now `mean_t (1 − cos(e_t, e_{t+1}))` over the
ego slot (`selected_tokens[:, 0, :]`, guaranteed to be the ego token
under `force_ego=True` for every variant). It is scale-invariant
(visibility multiplication preserves direction) and bounded in [0, 2].

Why we no longer read it as a quality metric:

- A *low* cosine distance can mean either "the encoder is well-behaved"
  or "the policy is dead and doesn't move the state". `wm_decoupled_no_vis`
  has the *lowest* mean cosine (0.223) but is the *worst* policy.
- A *high* cosine distance can mean either "rollout is divergent" or
  "the policy meaningfully steers ego state through diverse latents".
  `wm_decoupled (+vis)` has the *highest* (0.636) and is the
  intermediate policy.

Reading rule: cosine stability is now used as a **divergence sentinel**
(catches NaN-like blow-ups, e.g. ≥ 1.5) and as a **liveness sentinel**
(near-zero with high CollRate ⇒ policy is collapsed). Return + CollRate
remain the primary quality metrics.

`Stab(L2)` (the legacy `rollout_stability_global`) is kept in the JSON
but should not appear in any cross-variant comparison — `global_latent`
is defined differently per variant, so the L2 number compares apples
to oranges.

---

### 6. Reproducing X

```
# 1. Build the on-disk token cache once (~3 min, 16 workers, 951 MB):
PYTHONPATH=src python scripts/build_token_cache.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 700 --workers 16

# 2. Launch X (9 jobs in parallel on a single GPU; ~10-15 min total):
bash scripts/run_stage1_pilot_x.sh

# 3. Aggregate and emit X_summary.{md,json}:
PYTHONPATH=src python scripts/aggregate_stage1_x.py
```

Each individual run also accepts `--token-cache-dir <dir>` directly via
`run_stage1_table4.py` (default `experiments/_token_cache`). Cache hit
loads the 28 096 pre-tokenised samples in ~6 s instead of the 18-min
single-threaded devkit pass.

---

### 7. 14+2 and fusion ablation results

The targeted typed-budget ablation is now complete. `wm_decoupled_14_2`
uses the same architecture as `wm_decoupled`
(`object_relation_decoupled_visibility`), but with
`top_k_dyn=14, top_k_rel=2` instead of the default `12+4`. It was run
on the same 3 seeds (7, 42, 123) under v3 hparams.

| condition | top_k_dyn | top_k_rel | role |
|---|---:|---:|---|
| wm_object | 16 | 0 | anchor (no rel branch at all) |
| wm_decoupled (12+4) | 12 | 4 | current default, X result |
| **wm_decoupled_14_2** | 14 | 2 | targeted budget test |

Result:

| condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_decoupled_14_2 | 2.47 ± 4.14 | 0.808 ± 0.196 | 0.774 ± 0.189 | 0.419 ± 0.264 | 0.125 ± 0.068 |

Per-seed raw values are in `experiments/stage1_pilot_14_2/summary.md`
and `experiments/stage1_pilot_14_2/summary.json`.

What we learn:

- Reducing the relation budget from 4 to 2 does **not** recover Stage-1
  policy learning. It does not close the gap to `wm_object`, and its
  CollRate is worse on average than default 12+4.
- The Stage-1 failure is therefore not explained solely by excessive
  relation-slot allocation. Together with the no-visibility result, this
  also rules out "visibility is the culprit" as the simple explanation.
- The remaining high-value question is whether relation information is
  useful but harmful when fused directly into the policy latent.

After 14+2, we ran the minimal fusion ablation:
`wm_decoupled_rel_to_critic_only`. It keeps the decoupled structure,
keeps relation slots in the world model and value input, but prevents
the relation branch from driving the actor policy mean directly:

| condition | dyn to actor | rel to actor | rel to critic | purpose |
|---|---:|---:|---:|---|
| wm_object | yes | no | no | stable anchor |
| wm_decoupled | yes | yes | yes | current high-variance full fusion |
| **wm_decoupled_rel_to_critic_only** | yes | no | yes | test whether rel helps value/risk but hurts control |

Result:

| condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_decoupled_rel_to_critic_only | 5.72 ± 7.25 | 0.729 ± 0.169 | 0.736 ± 0.163 | 0.453 ± 0.228 | 0.170 ± 0.036 |

Per-seed raw values are in
`experiments/stage1_pilot_rel_critic_only/summary.md` and
`experiments/stage1_pilot_rel_critic_only/summary.json`.

What we learn:

- Critic-only rel is **not a rescue**. It does not close the gap to
  `wm_object` on Return or CollRate.
- It is still informative: compared with full decoupled 12+4, it has a
  slightly higher mean Return (5.72 vs 4.34) and lower variance. That
  suggests the fusion path matters, but actor detachment alone is too
  weak to solve the Stage-1 mismatch.
- The next high-value direction is no longer budget sweep. It is either
  relation selection regularisation during imagination, or a cleaner
  separation between relation/risk estimation and the actor state.

---

### 8. nuPlan 5k cross-dataset pilot

We also repeated the Stage-1 core comparison on **nuPlan preprocessed NPZ**
as a fast cross-dataset pilot. This is **not** nuPlan closed-loop; it uses
the same offline Stage-1 imagination setup as nuScenes, with 5 000 NPZ
samples per seed, 4 000 / 1 000 train/val split, and nuPlan Stage-0
warm-start checkpoints from `experiments/nuplan_stage0_5k_seed7`.

| condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | -6.01 ± 3.10 | 0.348 ± 0.254 | 0.384 ± 0.150 | 0.505 ± 0.441 | 0.010 ± 0.006 |
| wm_decoupled | 9.38 ± 9.59 | **0.215 ± 0.095** | **0.281 ± 0.097** | 0.887 ± 0.011 | 0.067 ± 0.029 |
| **wm_decoupled_no_vis** | **12.91 ± 2.69** | 0.247 ± 0.029 | 0.325 ± 0.060 | 0.097 ± 0.062 | 0.037 ± 0.034 |
| wm_decoupled_rel_to_critic_only | 5.46 ± 2.26 | 0.231 ± 0.052 | 0.337 ± 0.025 | 0.848 ± 0.058 | 0.054 ± 0.023 |

Per-seed raw values are in `experiments/nuplan_stage1_5k/summary.md` and
`experiments/nuplan_stage1_5k/summary.json`.

What we learn:

- nuPlan **does change the Stage-1 ranking** in this pilot: the decoupled
  family beats `wm_object` on Return and CollRate.
- The strongest mean Return is `wm_decoupled_no_vis`, while the lowest
  CollRate is default `wm_decoupled`. This reverses the nuScenes
  conclusion that no-visibility is uniformly worse.
- Treat this as a pilot, not a paper table yet. It uses 5k samples and a
  seed-7 Stage-0 warm-start for all Stage-1 seeds. The result is strong
  enough to justify a larger nuPlan run, but not enough to replace the
  nuScenes conclusion.

---

### 9. nuPlan 20k scale-up

We then scaled the nuPlan pilot to a balanced **20 000-sample** subset,
with **3 seeds × 3 conditions**:

- `wm_object`
- `wm_decoupled`
- `wm_decoupled_no_vis`

Setup: 16 000 / 4 000 train/val split, 10 epochs, batch 128, horizon 5,
same Stage-1 hparams as the nuScenes X verification, Stage-0 warm-start
from `experiments/nuplan_stage0_20k_seed7`. The loader used the 20k index
`experiments/nuplan_20k_balanced_paths_seed7.json` and 32 nuPlan
tokenisation workers per seed.

| condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | 4.74 ± 13.95 | 0.373 ± 0.083 | 0.395 ± 0.057 | 0.426 ± 0.099 | 0.047 ± 0.009 |
| wm_decoupled (+vis) | 13.48 ± 4.09 | 0.488 ± 0.217 | 0.509 ± 0.193 | 0.546 ± 0.046 | 0.095 ± 0.049 |
| **wm_decoupled_no_vis** | **17.50 ± 1.37** | **0.226 ± 0.105** | **0.251 ± 0.101** | 0.136 ± 0.022 | 0.043 ± 0.020 |

Per-seed raw values are in `experiments/nuplan_stage1_20k/summary.md` and
`experiments/nuplan_stage1_20k/summary.json`.

What we learn:

- The 5k nuPlan signal **survives scale-up**. Decoupled is not winning
  only because the 5k pilot was small.
- `wm_decoupled_no_vis` is now the cleanest 20k winner: all three seeds
  are positive on Return (16.22 / 17.32 / 18.95), with the lowest mean
  collision rate.
- `wm_decoupled(+vis)` still improves Return over `wm_object`, but its
  collision metrics are worse and seed 123 is unstable. Visibility is no
  longer a universally helpful Stage-1 bias.
- The main paper-level insight is stronger than "decoupled is better":
  relation-aware abstraction helps Stage-1 policy learning **depending on
  the downstream data/planning regime**. On nuScenes, object-only is still
  the robust policy-learning baseline; on nuPlan 20k, decoupled-no-vis
  becomes the robust winner.

---

### 10. FlowPlanner `comfort_423v1` comfort-metric pilot

This note is separate from the Stage-1 imagination-RL tables above. It records
the FlowPlanner PnP comfort-metric pilot at:

`/mnt/volumes/cpfs/prediction/lipeinan/pnp_experiments/20260424_013330_comfort_423v1`

Setup captured by the saved config:

| item | value |
|---|---|
| Config | `projects/flow_planner/config/train_pnp_joint.py` |
| Epochs | 10 |
| Model | `FlowPlanner` PnP joint model, 2.29M trainable params |
| Optimizer / LR | AdamW, base LR `5e-4`, 5-epoch warmup |
| Batch | `batch_size_per_gpu=112`, `val_batch_size_per_gpu=112` |
| Saved data interval | `train_load_interval=1`, `val_load_interval=1` in `config.json` |
| Eval | every epoch, `PnPEvaluator`, `metric_key=plan/minADE` |
| Comfort path | model-score top-1 plan, 12-point joint output upsampled to 60 points at 0.1s |

Main result:

| Epoch | pred/minADE ↓ | pred/top1FDE ↓ | plan/minADE ↓ | plan/minFDE ↓ | plan/MR ↓ | accel ↑ | decel ↑ | jerk ↑ | PDMS comfort ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.117 | 4.475 | 0.672 | 1.416 | 0.196 | 0.981 | 0.969 | 0.194 | 71.45 |
| 2 | 1.034 | 4.146 | 0.613 | 1.301 | 0.172 | 0.989 | 0.958 | 0.383 | 77.65 |
| 3 | 0.999 | 4.001 | 0.585 | 1.240 | 0.157 | 0.991 | 0.988 | 0.541 | 83.99 |
| 4 | 0.976 | 3.922 | 0.569 | 1.213 | 0.156 | 0.993 | 0.999 | 0.606 | 86.58 |
| 5 | 0.954 | 3.837 | 0.565 | 1.203 | 0.149 | 0.994 | 0.999 | 0.661 | 88.49 |
| 6 | 0.941 | 3.857 | 0.564 | 1.199 | 0.154 | 0.997 | 1.000 | 0.732 | 90.96 |
| 7 | 0.931 | 3.823 | 0.564 | 1.188 | 0.150 | 0.999 | 1.000 | 0.749 | 91.56 |
| 8 | 0.923 | 3.778 | 0.544 | 1.154 | 0.142 | 0.999 | 0.999 | 0.789 | 92.89 |
| 9 | 0.913 | 3.707 | 0.541 | 1.145 | 0.133 | 0.998 | 1.000 | 0.785 | 92.75 |
| 10 | **0.907** | **3.690** | **0.538** | **1.140** | **0.138** | **0.999** | **0.999** | **0.810** | **93.60** |

What changed over the run:

- Planning quality improved steadily: `plan/minADE` went from **0.672 m → 0.538 m**,
  `plan/minFDE` from **1.416 m → 1.140 m**, and `plan/MR` from **0.196 → 0.138**.
- Prediction also improved, but more moderately: `pred/minADE` went from
  **1.117 m → 0.907 m**, and `pred/top1FDE` from **4.475 m → 3.690 m**.
- Comfort improved from **71.45 → 93.60 / 100**. Almost all of this gain came
  from jerk: `jerk_comfort` rose from **0.194 → 0.810**. Accel/decel ratios were
  already high at epoch 1 and saturated near 1.0 by epoch 6.

Important interpretation notes:

- The high `accel_comfort` / `decel_comfort` values do **not** mean every
  6-second trajectory is fully comfortable. They are frame-side ratios:
  `accel_comfort = compliant positive-acceleration frames / positive-acceleration frames`,
  and `decel_comfort = compliant negative-acceleration frames / negative-acceleration frames`.
  They are not binary per-sample comfort checks.
- `pdms_comfort` here is therefore best read as a **training trend / internal
  regression metric**, not as a strict deployment comfort score. A stricter
  sample-level or peak-violation metric would likely score lower.
- The saved code snapshot contains the important top-1 fix: comfort is computed
  on `plan_scores.argmax` rather than oracle best-ADE mode. This aligns the
  evaluator with the model-score top-1 path used at deployment.
- The score trajectory is internally consistent: after accel/decel saturate,
  the PDMS curve is driven almost entirely by jerk smoothing, which matches the
  intended role of the 60-point evaluator-aligned kinematic check.

Why the score looks high:

1. **The saved run is not a tiny 1/500 smoke in the final config.** The saved
   `config.json` records `train_load_interval=1` and `val_load_interval=1`.
   Therefore this run should be read as a much stronger training/evaluation run
   than the earlier smoke commands that intentionally sampled every 500th or
   1000th item.
2. **Accel/decel comfort is a frame-side ratio, not a trajectory-level veto.**
   A sample can contain a few uncomfortable bursts while still getting a high
   frame-ratio score if most positive-accel or negative-accel frames remain
   under the speed-banded threshold. This makes `accel_comfort` and
   `decel_comfort` optimistic compared with human visual inspection of a few
   bad-looking frames.
3. **The 12-point plan is evaluated after 12→60 cubic-spline upsampling.** This
   makes the acceleration curve much smoother than direct 12-point finite
   differencing. The remaining discomfort mainly appears as jerk at segment
   transitions, which is why `jerk_comfort` starts low and drives most of the
   PDMS improvement.
4. **The evaluator now uses model-score top-1, not oracle best-ADE.** The high
   score is therefore not caused by selecting the GT-closest mode. It reflects
   the model's chosen top-1 under this ratio-based comfort definition.

The strongest way to answer the concern is: the run is healthy and the trend is
real, but the absolute value is optimistic because the metric is a frame-ratio
comfort monitor. For release claims, report an additional stricter metric, such
as binary sample comfort (`all frames pass`) or peak violation, alongside this
PDMS ratio.

Practical conclusion:

`comfort_423v1` is a strong smoke / trend run for the revised comfort evaluator:
metrics are stable, planning accuracy improves, and the comfort trend is driven
by the expected bottleneck (`jerk_comfort`). However, because the comfort ratios
are frame-level and the run is a single configuration, this should not be cited
as an absolute "production comfort = 93.6" claim. Use it as evidence that the
revised metric wiring is healthy and that the model becomes smoother over
training; use a stricter binary/peak metric for release-level comfort claims.

---

### 11. File map

| File | Purpose |
|---|---|
| `src/doorrl/models/policy.py` | `ActorCriticHead` with `tanh` mean bound + `log_std` clamp |
| `src/doorrl/imagination/imagination.py` | K-step rollout; emits `ego_latents` for cosine stability |
| `src/doorrl/imagination/task_reward.py` | task reward, reward clip ±5 |
| `src/doorrl/training/losses_stage1.py` | actor / Huber-critic / sanity losses + diagnostics |
| `src/doorrl/training/trainer_stage1.py` | `ImaginationTrainer` (`detach_world_model=True` default) |
| `src/doorrl/evaluation/stage1_metrics.py` | latent return, imagined collision, cosine stability |
| `src/doorrl/data/real_dataset.py` | `NuScenesSceneDataset` with optional disk token cache |
| `run_stage1_table4.py` | per-condition runner, `--token-cache-dir`, `--entropy-beta`, `--action-sample-clip` |
| `scripts/build_token_cache.py` | 16-way parallel pre-tokeniser (run once, reused forever) |
| `scripts/run_stage1_pilot_v3.sh` | single-seed v3 baseline launcher |
| `scripts/run_stage1_pilot_y.sh` | decoupled-without-visibility ablation |
| `scripts/run_stage1_pilot_x.sh` | 3-seed × 3-condition X verification (parallel) |
| `scripts/aggregate_stage1_x.py` | X mean±std summary writer |
| `experiments/stage1_pilot_x/X_summary.{md,json}` | **primary X result table** |
| `experiments/stage1_pilot_14_2/summary.{md,json}` | 14+2 typed-budget ablation result |
| `experiments/stage1_pilot_rel_critic_only/summary.{md,json}` | rel-to-critic-only fusion ablation result |
| `experiments/nuplan_stage1_5k/summary.{md,json}` | nuPlan 5k Stage-1 cross-dataset pilot |
| `experiments/nuplan_stage1_20k/summary.{md,json}` | nuPlan 20k Stage-1 scale-up result |
| `experiments/stage1_pilot_x/seed*/<cond>/stage1_metrics.json` | per-(seed, cond) metrics |
| `experiments/stage1_pilot_x/logs/x_seed*_<cond>.log` | per-run training log |
| `experiments/stage1_pilot_y/seed7/wm_decoupled_no_vis/` | Y single-seed result (subsumed by X) |
| `experiments/stage1_pilot_v3/seed7/<cond>/` | v3 single-seed (kept for history; do **not** cite) |
| `experiments/stage1_pilot_ab/` | v2 A+B patch single-seed (historical) |
