# When Relations Compete with Objects: Budget-Aware Token Abstraction for Driving World Models

_NeurIPS-oriented draft v1, 2026-04-28._  
_Working title. This draft intentionally reframes DOOR-RL around limited-capacity object-relational abstraction rather than around a system name._

---

## Abstract

Object-relational world models provide a structured interface for autonomous driving, but they must often operate under a limited latent context budget. We show that this setting creates a previously under-examined failure mode: **type competition**. Object tokens are state-bearing entities, while relation tokens encode sparse but decision-dense cues such as time-to-collision and interaction risk. A unified top-K selector treats these heterogeneous tokens as exchangeable, causing relation tokens to displace critical dynamic agents or become ungrounded when their endpoint objects are absent from the latent state. We propose a **budget-aware, type-aware token abstraction** that allocates separate capacity to dynamic agents and relation edges under the same total world-model context budget. To expose the mechanism, we introduce selection diagnostics including **Critical Dynamic Retention** and **Relation Wasted Slot Rate**. Across nuScenes and nuPlan, our abstraction improves limited-budget representation sufficiency, reduces critical-agent misses and wasted relation slots, and yields more stable latent planning performance. Our results suggest that relation-aware driving world models require not only richer relation tokens, but also budget-aware mechanisms that keep those relations grounded in retained dynamic state.

---

## 1. Introduction

Structured world models are an increasingly attractive substrate for autonomous driving. Rather than compressing a scene into a single global latent vector, object-relational representations expose the entities and interactions that matter for decision making: the ego vehicle, nearby vehicles, pedestrians and cyclists, lane-level context, and interaction cues such as time-to-collision, conflict, and priority. Such structure is especially useful for planning, where rare but safety-critical entities and their relations often dominate the correct action.

Yet object-relational structure creates a practical bottleneck. A real driving scene can contain many object, map, signal, and relation tokens, while the downstream world model and policy typically consume only a limited context. This forces a token abstraction problem: which subset of tokens should be retained in the latent state? A common answer is to score all tokens and keep a unified top-K set. This design is simple, but it silently assumes that all token types are exchangeable under a shared budget.

We argue that this assumption is wrong for driving world models. Object tokens and relation tokens play different roles. Object tokens are **state-bearing**: they carry the physical states that must be predicted forward. Relation tokens are **decision-dense**: they may encode sparse but high-value signals such as low TTC, lane conflict, or interaction risk. Moreover, the value of a relation token depends on its endpoints. A selected relation whose associated dynamic agent is missing from the latent state is not a useful relational fact; it is an **orphan relation**. Under a fixed shared top-K budget, relation tokens and object tokens therefore do not merely add information to each other. They compete.

This paper studies the resulting failure mode, which we call **type competition**. In a limited-capacity object-relational world model, a unified selector can allocate slots to relation tokens while missing critical dynamic agents. The failure is subtle because the selected relations may appear semantically meaningful in isolation, yet the latent state lacks the entities needed to ground them. In prediction and planning, this manifests as poor dynamic-agent retention, degraded rare-agent forecasting, lower interaction recall, and unstable downstream latent policy learning.

We propose a simple but important alternative: **budget-aware type abstraction**. Instead of applying one top-K selector to the union of objects and relations, we allocate separate budgets to dynamic-agent tokens and relation tokens while keeping the total world-model context budget fixed. In our implementation, a 16-slot context is split into a dynamic path and a relation path, with independent selectors and type-aware supervision. Dynamic slots predict physical next-state quantities, while relation slots predict decision-relevant edge semantics.

To make the failure mode measurable, we introduce selection diagnostics. **Critical Dynamic Retention** directly measures whether a selector retains the agents that should not be forgotten: nearby agents, low-TTC agents, lane-conflicting agents, and rare agents near the ego. **Critical Agent Miss Rate** is its complement. **Relation Wasted Slot Rate** measures how often a selected relation is ungrounded because its endpoint dynamic agent is not retained. These diagnostics turn a qualitative motivation into testable mechanism evidence.

Empirically, we evaluate the abstraction in a staged pipeline. On nuScenes, under a fair 16-slot world-model context budget, naive shared object-relation selection catastrophically degrades dynamic rollout and interaction recall, while typed-budget abstraction recovers and surpasses object-only selection on dynamic-agent metrics. On nuPlan, the same abstraction yields more stable latent imagination planning at 50k scale. Selection diagnostics reveal that naive shared selection has low critical-agent retention and can produce many wasted relation slots, while typed-budget selection retains critical agents and keeps relation slots grounded. A relation feature ablation further shows that the useful relation semantics primarily come from TTC/risk features rather than from all relation attributes equally.

Our contributions are:

1. We identify **type competition** as a failure mode of limited-capacity object-relational world models for driving.
2. We propose a **decision-sufficient, budget-aware, type-aware token abstraction** that separates dynamic-agent and relation selection under a fixed total context budget.
3. We introduce **selection diagnostics** that quantify critical-agent misses and orphan relations, linking selector behavior to downstream interaction errors.
4. We provide prediction-to-planning evidence across nuScenes and nuPlan, including representation sufficiency, latent planning, relation semantic ablations, and external-style imitation baselines.

---

## 2. Related Work

### Object-Centric and Object-Relational World Models

Object-centric world models decompose a scene into entities, making dynamics, interaction, and planning more interpretable than holistic latent representations. In autonomous driving, such representations naturally align with tracked agents, lanes, and traffic signals. Relation-aware variants further introduce edges or pairwise tokens that encode interaction structure. Our work focuses on the selection problem that arises once object and relation tokens must share a limited latent context.

### Token Selection and Bottlenecked Scene Abstraction

Sparse attention, token pruning, and top-K selection are common mechanisms for reducing context length. However, most selection mechanisms treat tokens as comparable items competing for the same budget. We show that this can be harmful when token types have different semantic roles. Our method is not a larger model or a new perception frontend; it is a structural abstraction constraint for limited-budget world-model context.

### Decision-Sufficient Representation Learning

For planning, a good representation need not reconstruct every scene detail. It should preserve the variables needed to predict decision-relevant futures and select actions. We operationalize this idea with both prediction metrics and selection diagnostics: a representation should retain critical dynamic agents and relation semantics that influence downstream planning, while avoiding ungrounded relation slots.

### World Models for Autonomous Driving Planning

Driving world models are often evaluated through prediction errors, imitation losses, or closed-loop planning proxies. We use a staged evaluation: first testing representation sufficiency under a fair context budget, then evaluating latent imagination policy learning, offline planner-like alignment, relation semantic ablations, and cross-dataset behavior. This staging isolates the abstraction mechanism from unrelated simulator engineering.

---

## 3. Problem Formulation

Consider a driving scene represented as a set of typed tokens:

```text
X = O ∪ R ∪ M ∪ S
```

where `O` denotes dynamic object tokens, `R` relation tokens, `M` map tokens, and `S` traffic-signal or other auxiliary tokens. In this work, dynamic object tokens include ego, vehicle, pedestrian, and cyclist tokens. Relation tokens encode pairwise or ego-centric interaction cues such as time-to-collision, risk, lane conflict, and priority.

The world model consumes a limited context:

```text
Z = A(X),    |Z| <= K
```

where `A` is an abstraction module and `K` is a fixed context budget. The standard unified top-K abstraction scores all eligible tokens and selects the top K:

```text
S_shared = topK({ score(x) : x ∈ O ∪ R }, K).
```

This selector is budget-aware in the weak sense that it respects the total capacity, but it is not type-aware. Every relation token selected into `S_shared` can displace a dynamic object token, and vice versa.

We define **type competition** as the phenomenon where increasing the salience or density of one token type reduces the retention of decision-critical tokens of another type under a shared fixed budget. In object-relational driving representations, the harmful case is particularly clear: relation tokens can displace critical dynamic agents, and selected relations become less useful when their endpoint objects are not retained.

### Decision-Sufficient Abstraction

We call an abstraction **decision-sufficient** if it preserves the dynamic states and relation cues needed for task-relevant prediction and downstream policy learning. Decision sufficiency is not equivalent to reconstructing all tokens. Instead, it requires retaining the entities and interactions that matter for future collision, progress, and interaction behavior.

### Critical Dynamic Agents

For diagnostics, we define a set of critical dynamic agents at time `t`:

```text
C_dyn(t) = { i : d_i < 20m or TTC_i < 3s or lane_conflict_i = 1 or rare_agent_near_ego(i) }.
```

This set captures nearby agents, low-TTC agents, lane-conflicting agents, and rare vulnerable agents close to the ego vehicle.

---

## 4. Method

### 4.1 Object-Relational Tokenization

Each scene is tokenized into a fixed schema with dynamic, relation, map, signal, and padding tokens. Dynamic tokens store physical state, including ego-relative position and velocity. Relation tokens store decision-relevant edge features. The key raw dimensions used by the current implementation are:

| Raw dim | Meaning |
|---:|---|
| 0-3 | `(x, y, vx, vy)` |
| 7 | visibility |
| 8 | TTC |
| 9 | lane conflict |
| 10 | priority |
| 11 | distance |

The important distinction is semantic rather than purely dimensional: dynamic tokens carry state to be rolled forward, while relation tokens carry interaction cues whose value depends on the presence of their associated dynamic entities.

### 4.2 Unified Top-K Baseline

The naive object-relation baseline applies a single top-K selector to the union of dynamic and relation tokens. It is attractive because it lets the model decide whether object or relation tokens are more important. However, it also 爱reates a shared bottleneck. Under small K, a relation-heavy selection can starve the world model of dynamic-agent state.

### 4.3 Budget-Aware Type Abstraction

We replace the unified selector with two independent selectors:

```text
S_dyn = topK_dyn({ score_dyn(o) : o ∈ O }, K_dyn)
S_rel = topK_rel({ score_rel(r) : r ∈ R }, K_rel)
S     = S_dyn ∪ S_rel,
K_dyn + K_rel = K.
```

The total world-model context is unchanged. In the main experiments, `K = 16`, with `K_dyn = 12` and `K_rel = 4`. This is a typed capacity allocation, not an increase in model context.

The dynamic selector always preserves the ego token. The relation selector does not force-select ego, because relation tokens do not include ego as a token position in the relation-only set; forcing ego in that path would waste relation capacity. The selected dynamic and relation slots are concatenated and passed to the same world model used by all fair baselines.

### 4.4 Type-Aware Supervision

A second issue arises when different token types are trained with the same observation loss. Regressing all raw dimensions for relation tokens forces relation slots to predict physical state quantities that do not have a meaningful relation-token interpretation. We therefore use type-aware observation supervision:

| Selected slot type | Supervised target |
|---|---|
| Ego / vehicle / pedestrian / cyclist | `(x, y, vx, vy)` |
| Relation | `(TTC, lane_conflict, priority)` |
| Map / signal / padding | no next-state supervision |

This aligns supervision with token semantics and prevents relation slots from being optimized toward non-physical dynamic-state targets.

### 4.5 Latent Imagination Policy Learning

To evaluate whether representation improvements transfer downstream, we warm-start the world model and train an actor-critic policy through K-step latent imagination. The world model is detached during policy optimization, and the policy is evaluated with latent return, imagined collision, rollout stability, and teacher-action alignment. This stage is not the main methodological novelty; it tests whether the abstraction remains useful beyond one-step representation metrics.

---

## 5. Selection Diagnostics

We introduce diagnostics that expose how a selector spends its limited budget. Each evaluated sample logs, for every token:

```text
sample_id, dataset, scene_id, token_id, token_type, score, selected,
distance_to_ego, ttc_proxy, lane_conflict, visibility, is_rare_agent,
relation_endpoint_i, relation_endpoint_j
```

Current adapters serialize ego-object relation features but do not store endpoint identifiers directly. We recover the non-ego endpoint by nearest dynamic-token matching in ego-relative `(x, y)`. This approximation is sufficient for the current ego-centric relation schema; future tokenizers should store explicit endpoint ids.

### 5.1 Critical Dynamic Retention

Critical Dynamic Retention measures whether a selector keeps the dynamic agents that should not be forgotten:

```text
CDR = |S_dyn ∩ C_dyn| / |C_dyn|.
```

The complement is Critical Agent Miss Rate:

```text
MissRate = 1 - CDR.
```

Unlike Rare ADE, which measures prediction error after the world model has already operated on the selected context, CDR directly measures the selector's retention behavior.

### 5.2 Relation Wasted Slot Rate

Relation tokens are only useful when grounded in retained dynamic state. A selected relation `r_ij` is wasted if its associated dynamic endpoint is absent from the selected dynamic set. For the current ego-object relation schema, ego is always retained, so a selected relation is counted as wasted when the non-ego endpoint is missing.

```text
WastedRel = #{ r_ij ∈ S_rel : endpoint(r_ij) notin S_dyn } / |S_rel|.
```

This metric tests whether relation selection produces grounded interaction context or orphan relation slots.

### 5.3 Relation Over-Allocation and Density

We also log the selected relation ratio:

```text
ROI = |S_rel| / K
```

and scene relation density:

```text
RelDensity = |R| / (|O| + 1).
```

In the current results, relation density alone is not the strongest explanatory variable. The more robust mechanism is whether selected relations remain grounded by retained dynamic endpoints.

---

## 6. Experiments

### 6.1 Experimental Questions

We organize experiments around five questions:

1. Does unified object-relation selection fail under a fixed context budget?
2. Does budget-aware type abstraction improve representation sufficiency without increasing context size?
3. Does the selector retain critical dynamic agents and avoid orphan relations?
4. Do representation gains transfer to latent planning and planner-like alignment?
5. Which relation semantics are actually decision-relevant?

### 6.2 Datasets and Protocols

**nuScenes Stage 0.** We use 700 nuScenes scenes and evaluate representation sufficiency under a fair 16-slot world-model context budget. All variants use the same data, split, hyperparameters, and total context size.

**nuPlan Stage 1.** We use balanced nuPlan preprocessed subsets for latent imagination policy learning. The strongest scale-up uses 50k samples and three seeds.

**Selection diagnostics.** We evaluate selector behavior on nuPlan 50k for object-only and typed-budget conditions, and on nuPlan 20k for the naive shared-relation Stage 1 condition.

### 6.3 Baselines and Variants

| Method | Selector | Context budget | Purpose |
|---|---|---:|---|
| Holistic-full | no bottleneck | 97 | upper-bound reference |
| Holistic-16Slot | learned query compression | 16 | fair holistic bottleneck |
| Object-only | top-K over dynamic tokens | 16 | no relation baseline |
| Object+Relation naive | shared top-K over dynamic ∪ relation | 16 | type competition baseline |
| Object+Relation+Visibility | shared top-K with visibility weighting | 16 | shared-selector variant |
| Typed-budget decoupled | separate dynamic/relation top-K | 16 | ours |
| Typed-budget decoupled + visibility | decoupled with dynamic visibility weighting | 16 | ours variant |
| BC / planner-target imitation | supervised action baseline | 16 | external-style anchor |

---

## 7. Results

### 7.1 Fixed-Budget Representation Sufficiency

Under the fair 16-slot budget, naive shared object-relation selection fails catastrophically, while typed-budget abstraction recovers dynamic-agent prediction and interaction recall.

| Variant | Ctx | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|
| Holistic-16Slot | 16 | 2.11 ± 0.16 | 0.978 ± 0.011 | 1.42 ± 0.01 | 0.643 ± 0.015 |
| Object-only-16 | 16 | 3.74 ± 1.01 | 0.946 ± 0.004 | 1.10 ± 0.12 | 0.901 ± 0.034 |
| Object+Relation-16 naive | 16 | 40.28 ± 29.54 | 0.980 ± 0.013 | 7.51 ± 5.48 | 0.430 ± 0.407 |
| Obj+Rel+Vis-16 | 16 | 15.80 ± 9.93 | 0.933 ± 0.064 | 2.96 ± 1.64 | 0.728 ± 0.155 |
| Obj+Rel-Decoupled | 16 | 2.11 ± 0.19 | 0.929 ± 0.039 | 0.49 ± 0.18 | 0.984 ± 0.014 |
| Decoupled+Visibility | 16 | 1.88 ± 0.23 | 0.926 ± 0.029 | 0.52 ± 0.05 | 0.979 ± 0.008 |
| Holistic-full reference | 97 | 0.11 ± 0.12 | 0.988 ± 0.006 | 0.26 ± 0.02 | 1.000 ± 0.000 |

The failure of the naive shared selector is not a result of adding relation tokens per se. It is a budget allocation failure: relation and dynamic tokens compete for the same slots, and the world model loses state-bearing dynamic context.

### 7.2 Selection Diagnostics Reveal Critical-Agent Misses

Selection diagnostics show that the naive shared-relation condition misses many critical agents and sometimes produces wasted relation slots. Typed-budget selection maintains high critical-agent retention and low wasted-relation rates.

| Setting | Seed | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI |
|---|---:|---:|---:|---:|---:|
| wm_naive, nuPlan 20k | 7 | 0.481 | 0.519 | 0.068 | 0.031 |
| wm_naive, nuPlan 20k | 42 | 0.769 | 0.231 | 0.081 | 0.054 |
| wm_naive, nuPlan 20k | 123 | 0.391 | 0.609 | 0.499 | 0.078 |
| wm_object, nuPlan 50k | 7 | 0.893 | 0.107 | - | 0.000 |
| wm_object, nuPlan 50k | 42 | 0.930 | 0.070 | - | 0.000 |
| wm_object, nuPlan 50k | 123 | 0.533 | 0.467 | - | 0.000 |
| wm_decoupled_no_vis, nuPlan 50k | 7 | 0.943 | 0.057 | 0.042 | 0.260 |
| wm_decoupled_no_vis, nuPlan 50k | 42 | 0.942 | 0.058 | 0.069 | 0.260 |
| wm_decoupled_no_vis, nuPlan 50k | 123 | 0.931 | 0.069 | 0.022 | 0.260 |

The strongest downstream link is between MissRate and Interaction Recall. For `wm_naive`, the Spearman correlation between MissRate and Interaction Recall is approximately `-0.76`, `-0.68`, and `-0.60` across the three seeds. This supports the mechanism that missing critical dynamic agents directly harms interaction-sensitive prediction.

The relation-density chain is weaker: relation density alone does not reliably predict selected relation ratio or downstream error. We therefore interpret the main failure not as "more relation tokens are always bad", but as "selected relations must remain grounded by retained dynamic endpoints".

### 7.3 Latent Planning on nuPlan

On nuPlan 50k, typed-budget no-visibility abstraction yields the strongest and most stable latent planning performance among the evaluated world-model conditions.

| Condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|
| wm_object | 1.723 ± 17.886 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 |
| wm_decoupled | -0.330 ± 4.936 | 0.007 ± 0.012 | 0.277 ± 0.111 | 0.255 |
| wm_decoupled_no_vis | 14.511 ± 2.925 | 0.259 ± 0.045 | 0.277 ± 0.033 | 0.222 |

`wm_object` is high variance: one seed is strong, while another collapses. `wm_decoupled_no_vis` is positive on all three seeds and has lower mean collision than object-only. The visibility-weighted decoupled variant nearly eliminates binary imagined collision but sacrifices return, so it is not the final main condition.

### 7.4 Relation Semantics Matter: TTC/Risk Drives the Gain

We probe relation semantics by ablating relation feature groups at evaluation time on trained `wm_decoupled_no_vis` checkpoints.

| Ablation | Teacher MSE ↓ | Action ΔL2 ↓ | Return ↑ | CollRate ↓ | CollMean ↓ |
|---|---:|---:|---:|---:|---:|
| none | 6.628 ± 0.110 | 2.115 ± 0.083 | 14.510 ± 2.921 | 0.260 ± 0.045 | 0.277 ± 0.033 |
| no_ttc_risk | 7.992 ± 0.805 | 3.154 ± 0.335 | 8.498 ± 1.397 | 0.895 ± 0.126 | 0.891 ± 0.126 |
| no_lane_priority | 6.627 ± 0.111 | 2.113 ± 0.081 | 14.526 ± 2.931 | 0.257 ± 0.052 | 0.274 ± 0.040 |
| no_relation_semantics | 7.989 ± 0.827 | 3.151 ± 0.351 | 8.724 ± 1.629 | 0.894 ± 0.126 | 0.888 ± 0.127 |

Removing TTC/risk nearly triples collision metrics, while removing lane/priority has little effect under the current tokenizer. This suggests that the useful relation information is concentrated in risk/TTC semantics rather than distributed uniformly across all relation features.

### 7.5 External-Style Imitation Anchor

We include a low-cost BC / planner-target imitation baseline on the same nuPlan 50k protocol. It is not intended to beat the world-model method; its role is to ensure that the paper is not purely comparing internal variants. The BC baseline reaches return `0.433 ± 4.230` and teacher-action MSE `8.824 ± 0.113`, below `wm_decoupled_no_vis` in latent return and teacher-action alignment.

### 7.6 Cross-Dataset Behavior

The abstraction is not a universal win in every regime. On nuScenes Stage 1, object-only is currently more stable, while on nuPlan 50k the decoupled no-visibility abstraction is stronger. We interpret this as a regime-dependent finding: relation-aware abstraction helps when downstream planning data and token quality make relation semantics decision-relevant. This is consistent with interaction-conditioned analyses showing stronger gains in low-TTC, lane-conflict, dense-interaction, and rare-agent-dense subsets.

---

## 8. Discussion

### Type Competition, Not "Bad Relations"

The central result is not that relation tokens are harmful. Relation tokens can be highly useful, especially TTC/risk features. The problem is that relation tokens need to be grounded in retained dynamic state. A shared top-K selector can spend capacity on relation tokens while missing the objects that make those relations meaningful. Budget-aware type abstraction prevents this by reserving capacity for state-bearing agents and decision-dense relations separately.

### Why Not Learn the Budget?

A natural extension is to learn `K_dyn` and `K_rel` dynamically. We leave this for future work. The point of the present paper is to establish the failure mode and show that a simple structural prior fixes it under a strict fixed total budget. Adding learned budget allocation would introduce a second layer of optimization and make the mechanism harder to isolate.

### Why Relation Density Alone Is Insufficient

We originally hypothesized that relation density would directly predict relation over-allocation. The current diagnostics do not strongly support that chain. This negative result is useful: type competition is not merely a function of how many relation tokens exist. It depends on whether relation selection displaces critical dynamic state and whether selected relations remain grounded.

### What This Paper Does Not Claim

This work is not a full closed-loop autonomous driving leaderboard result. It studies limited-capacity object-relational abstraction and validates the mechanism through prediction, latent planning, offline planner-like alignment, and diagnostic metrics. Full reactive closed-loop evaluation is an important next step but not the central claim of this paper.

---

## 9. Limitations

First, current relation tokens are largely ego-object relations, and endpoint ids are recovered for diagnostics by nearest dynamic-token matching. Future tokenizers should store explicit relation endpoints. Second, the main typed budget uses a fixed `12/4` split. We include sensitivity evidence, but do not yet learn budgets dynamically. Third, latent planning metrics are not a substitute for full high-fidelity closed-loop evaluation. Finally, relation utility varies across datasets and planning regimes; this paper frames that variation as part of the phenomenon rather than hiding it.

---

## 10. Reproducibility and Artifact Map

Key implementation files:

| Component | Path |
|---|---|
| Model variants and typed-budget abstraction | `src/doorrl/models/doorrl_variant.py` |
| Top-K abstraction module | `src/doorrl/models/abstraction.py` |
| Type-aware observation losses | `src/doorrl/training/losses.py` |
| Stage0 metrics | `src/doorrl/evaluation/table3_metrics.py` |
| Stage1 metrics | `src/doorrl/evaluation/stage1_metrics.py` |
| Selection diagnostics | `scripts/selection_diagnostic.py` |

Key experiment outputs:

| Evidence | Path |
|---|---|
| Stage0 fair 16-slot aggregate | `experiments/table3_fair_fix2_aggregate.json` |
| nuPlan Stage1 50k | `experiments/nuplan_stage1_50k/summary.md` |
| Selection diagnostic, 50k | `experiments/selection_diagnostic_nuplan50k/summary.md` |
| Selection diagnostic, naive shared 20k | `experiments/selection_diagnostic_shared_relation_20k/summary.md` |
| Relation feature ablation | `experiments/nuplan_relation_feature_ablation_50k/summary.md` |
| BC baseline | `experiments/nuplan_bc_baseline_50k/summary.md` |
| Budget sensitivity | `experiments/stage0_budget_10_6_nuscenes/summary.md` |

---

## 11. Paper TODOs

### Main Text

- [ ] Replace this markdown with a NeurIPS LaTeX structure.
- [ ] Tighten the abstract to 150-200 words once final figures are fixed.
- [ ] Add citations and related-work paragraphs.
- [ ] Convert formulas into LaTeX.
- [ ] Replace "DOOR-RL" as the lead framing with the method name "budget-aware type abstraction"; keep DOOR-RL as implementation/system name if needed.

### Figures and Tables

- [ ] Figure 1: conceptual type competition diagram.
- [ ] Figure 2: CDR / MissRate / WastedRel diagnostic bars.
- [ ] Figure 3: MissRate vs Interaction Recall scatter or binned plot.
- [ ] Figure 4: relation semantic ablation.
- [ ] Table 1: Stage0 fixed-budget representation sufficiency.
- [ ] Table 2: Stage1 nuPlan 50k latent planning.
- [ ] Appendix table: BC baseline and budget sensitivity.

### Optional Experiments

- [ ] `wm_naive` 50k if compute is available.
- [ ] Paper-ready closed-loop smoke summary, clearly marked as sanity rather than SOTA.
- [ ] Explicit endpoint serialization in future tokenizer, if time permits.

