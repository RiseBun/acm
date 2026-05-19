# NeurIPS 重构方案：Budget-Aware Object-Relation Abstraction

_Created: 2026-04-28_  
_Goal: 将当前 DOOR-RL 从“一个 decoupled selector 工程消融”重构为一个可冲 NeurIPS 的有限容量 object-relational world model 表示学习问题。_

---

## 0. 新论文核心定位

当前论文不应再主打：

> 我们把 object token 和 relation token 分开 top-k，效果更好。

而应重构为：

> 在 object-relational world model 中，object tokens 和 relation tokens 具有不同的数量分布、语义密度和决策功能。当世界模型上下文容量有限时，统一 top-K selector 会把这些异质 token 当成可交换元素，产生系统性的 type competition：relation tokens 可能挤掉关键动态体，或者形成 endpoint dynamic agents 不在 latent state 中的 orphan relations。本文研究如何学习一个 decision-sufficient、budget-aware、type-aware 的 token abstraction，使有限容量 latent state 同时保留关键动态体与其决策相关关系，并在预测、潜在规划和跨数据集评估中稳定有效。

一句话版本：

> We identify type competition as a failure mode of limited-capacity object-relational world models, and propose a decision-sufficient, budget-aware, type-aware token abstraction that preserves critical agents while grounding relation tokens for robust prediction and planning.

---

## 1. 新问题定义

### 1.1 背景

自动驾驶 world model 常把场景表示为 object tokens、map tokens、relation tokens 等结构化 token。随着模型上下文容量受限，系统必须从大量候选 token 中选出一个小的 latent context，例如当前 DOOR-RL 中的 16-slot world-model context。

在这种设置下，object tokens 与 relation tokens 不是同质的：

- Object tokens 是 state-bearing entities，承载可预测的动态状态，如位置、速度、尺寸、类别。
- Relation tokens 是 decision-dense edges，承载 TTC、risk、lane conflict、priority 等稀疏但高价值的决策信号。
- Relation tokens 的价值依赖 endpoint objects 是否仍在 latent state 中；没有 endpoint dynamic agents 的 relation 很可能变成 orphan relation。
- Relation token 数量和密度随场景而变化，且与 object token 数量、风险分布、规划数据集风格相关。

### 1.2 失败模式

统一 top-K selector 默认将 object/relation token 视为可交换候选项，在固定 K 下竞争同一组 slot。这会导致：

1. **Critical dynamic miss**  
   关键车辆、行人、骑行者没有进入 latent state。

2. **Orphan relation**  
   relation token 被选中，但其 endpoint dynamic agent 没有被选中，使 relation 在 latent state 中失去 grounding。

3. **Type competition**  
   relation token 和 object token 争夺相同预算，导致“加入关系信息反而损害动态体表示”。

4. **Downstream instability**  
   表示层的 critical-agent miss 会传导到 Rare ADE、Interaction Recall、潜在想象规划、teacher-action alignment。

### 1.3 研究问题

建议主问题写成：

> How can we learn a decision-sufficient, budget-aware, and type-aware token abstraction for limited-capacity object-relational world models, such that the latent state preserves both critical state-bearing objects and decision-relevant relations for prediction, planning, and cross-dataset generalization?

---

## 2. 新贡献组织

### Contribution 1: Type Competition Diagnosis

提出并实证识别有限容量 object-relational world model 中的 type competition failure mode。

主张：

- Naive shared top-K 在 fixed context budget 下不是“relation token 无用”，而是“object 与 relation 的异质 token 被迫竞争同一预算”。
- 该失败表现为 critical dynamic agents 被 miss，以及 selected relation 与 retained dynamic state 脱钩。

支撑证据：

- Stage0 nuScenes: `object_relation` 在 16-slot 下 DynRoll / Rare ADE / IntRec 大幅崩坏。
- Selection diagnostic: `wm_naive` 20k 上 CDR 低、MissRate 高、WastedRel 在坏 seed 上高达约 0.499。

### Contribution 2: Budget-Aware Type Abstraction

提出 typed-budget decoupled abstraction：

- Dynamic path: top-K over ego / vehicle / pedestrian / cyclist。
- Relation path: top-K over relation tokens。
- Fixed total budget: `K_dyn + K_rel = K`，当前主设置为 `12 + 4 = 16`。
- Relation path 不强制选 ego，避免浪费 relation slot。
- 类型化 observation loss：dynamic slots 监督 `(x,y,vx,vy)`；relation slots 监督 `(TTC, lane_conflict, priority)`。

该方法不是简单调 K，而是一个结构归纳偏置：

> state-bearing objects and decision-dense relation edges require separate budget allocation under fixed context capacity.

### Contribution 3: Mechanistic Metrics

提出 selection diagnostic 指标，将动机变成硬证据：

- Critical Dynamic Retention, `CDR`
- Critical Agent Miss Rate, `MissRate = 1 - CDR`
- Relation Wasted Slot Rate, `WastedRel`
- Relation Over-allocation Index, `ROI`
- Relation density conditioned analysis, `RelDensity`

当前结果最稳的表述：

- `wm_naive`: CDR 低、MissRate 高、Rare ADE 高、Interaction Recall 低。
- `wm_naive seed123`: `WastedRel ~= 0.499`，说明 orphan relation 是真实 failure mode。
- `wm_decoupled_no_vis`: CDR 稳定在 `0.931-0.943`，WastedRel 低至 `0.022-0.069`。
- `MissRate` 与 `Interaction Recall` 强负相关，`wm_naive` 三个 seed 的 Spearman 约 `-0.76 / -0.68 / -0.60`。

注意：当前不应强推 `RelDensity -> selected relation ratio`，因为该相关性不强。更稳的结论是：

> Relation density alone is not the failure driver; the key failure is whether selected relation tokens remain grounded by retained dynamic endpoints.

### Contribution 4: Prediction-to-Planning Evidence

展示 type-aware abstraction 不只改善表示指标，还能延伸到规划相关指标：

- Stage0 representation sufficiency: nuScenes 700 scenes / 3 seeds。
- Stage1 latent imagination RL: nuPlan 50k / 3 seeds。
- Offline planner-like sanity: teacher action MSE / action delta。
- Interaction-conditioned subset analysis: relation-aware abstraction 更有利于低 TTC、dense interaction、lane conflict、rare-agent dense 等场景。
- Cross-dataset analysis: nuScenes 与 nuPlan ranking reversal 说明 relation abstraction 的收益依赖 downstream planning regime。

---

## 3. 新论文结构

### Abstract

核心要素：

1. object-relational world models need token selection under limited context;
2. object and relation tokens are heterogeneous;
3. unified top-K creates type competition;
4. propose budget-aware type abstraction;
5. introduce diagnostics proving critical-agent miss and orphan relations;
6. improves prediction and planning-related metrics across nuScenes/nuPlan.

建议摘要骨架：

> Object-relational world models provide structured scene representations for autonomous driving, but their context budget is often limited. We show that naively applying a unified top-K selector over objects and relations creates type competition: relation tokens may displace critical dynamic agents, while selected relations become ungrounded when their endpoint objects are absent. We propose a decision-sufficient, budget-aware, type-aware abstraction that allocates separate capacity to dynamic agents and relation edges under the same total context budget. We further introduce selection diagnostics including Critical Dynamic Retention and Relation Wasted Slot Rate to expose the mechanism. Across nuScenes and nuPlan, our abstraction improves representation sufficiency, retains critical agents, reduces orphan relations, and yields more stable latent planning performance.

### 1. Introduction

推荐叙事顺序：

1. 自动驾驶 world model 越来越依赖 object-relational tokenization。
2. 真实部署/latent planning 中 context budget 有限，必须选择 token。
3. 现有 top-K selector 通常把所有 token 混在一起，假设 token 可交换。
4. 但 object 与 relation 的语义角色不同：object 是 state-bearing，relation 是 decision-dense 且依赖 endpoint。
5. 我们发现 shared top-K 会产生 type competition。
6. 提出 typed-budget abstraction，并用新 diagnostics 证明机制。

Introduction 里不要过早说“12/4 最好”，而要说：

> The key design is to decouple token selection by semantic role while preserving the same total world-model context budget.

### 2. Related Work

建议分四类：

1. Object-centric world models for driving / robotics。
2. Relation-aware scene representation / graph neural planning。
3. Token selection / sparse attention / bottlenecked latent world models。
4. Decision-sufficient abstraction / representation learning for control。

需要避免把文章写成“又一个自动驾驶模型”。重点是 limited-capacity token abstraction。

### 3. Problem Formulation

建议正式定义：

- token set: `X = O ∪ R ∪ M`
- objects: `O = {o_i}`
- relations: `R = {r_ij}`
- fixed budget: select `S`, `|S| <= K`
- unified selector: `S = topK(score(x), x in O ∪ R)`
- type-aware selector: `S = S_dyn ∪ S_rel`, `|S_dyn| = K_dyn`, `|S_rel| = K_rel`, `K_dyn + K_rel = K`

定义 decision-sufficient abstraction：

> An abstraction is decision-sufficient if it preserves the subset of dynamic states and relation cues needed to predict task-relevant futures and support downstream policy learning.

定义 type competition：

> Type competition occurs when increasing the salience or density of one token type reduces the retention of decision-critical tokens of another type under a fixed shared budget.

### 4. Method

方法部分组织：

#### 4.1 Object-Relational Tokenization

说明 token 类型和关键 raw dims：

- dynamic: ego, vehicle, pedestrian, cyclist
- relation: TTC, risk, lane conflict, priority
- map/signal

#### 4.2 Unified Top-K Baseline and Its Failure

不要在方法里提前给结果，但要说明 shared top-K 的形式：

`S_shared = topK(score(x), K)`

它没有类型约束，因此可能让 relation 与 dynamic agent 竞争。

#### 4.3 Budget-Aware Type Abstraction

主方法：

`S_dyn = topK_dyn(score_dyn(o), K_dyn)`  
`S_rel = topK_rel(score_rel(r), K_rel)`  
`S = concat(S_dyn, S_rel)`

强调：

- same total K;
- separate selector heads;
- relation path `force_ego=False`;
- no extra world-model capacity;
- not a post-hoc filter but a structural prior.

#### 4.4 Type-Aware Supervision

说明不同 token 类型有不同预测目标：

- dynamic slots: `(x,y,vx,vy)`
- relation slots: `(TTC, lane_conflict, priority)`
- map/signal: no next-state supervision

这部分很关键，因为它让 relation token 的语义目标和 object token 分开。

#### 4.5 Latent Imagination Policy Learning

简述 Stage1：

- Stage0 warm-started world model;
- detach world model;
- K-step imagination;
- actor-critic over latent state;
- report latent return, imagined collision, teacher-action alignment。

不要把 RL 细节写得比 abstraction 更重；它是 downstream validation。

### 5. Selection Diagnostics

这是新版论文最重要的机制 section。

#### 5.1 Per-token selector log

每个 sample 保存：

- `sample_id`
- `dataset`
- `scene_id`
- `token_id`
- `token_type`
- `score`
- `selected`
- `distance_to_ego`
- `ttc_proxy`
- `lane_conflict`
- `visibility`
- `is_rare_agent`
- `relation_endpoint_i`
- `relation_endpoint_j`

#### 5.2 Critical Dynamic Retention

定义 critical dynamic agents：

```text
C_dyn(t) = { i : d_i < 20m or TTC_i < 3s or lane_conflict_i = 1 or rare-agent near ego }
```

指标：

```text
CDR = |S_dyn ∩ C_dyn| / |C_dyn|
MissRate = 1 - CDR
```

解释：

> CDR directly measures whether the selector retains the agents it should not forget.

#### 5.3 Relation Wasted Slot Rate

定义：

```text
WastedRel = #{ r_ij in S_rel : i notin S_dyn and j notin S_dyn } / |S_rel|
```

对于当前 ego-object relation tokenization，可写成：

> Since current adapters serialize ego-object relations, we count a selected relation as wasted when its non-ego endpoint is not selected.

#### 5.4 Relation Over-allocation Index

定义：

```text
ROI = |S_rel| / K
RelDensity = |R| / (|O| + 1)
```

当前结果注意事项：

- ROI 在 decoupled 中约为固定预算比例，主作用是说明预算稳定。
- `RelDensity -> selected relation ratio` 当前不是最强链条，不建议作为主结论。

---

## 4. 当前证据如何重排成主表与主图

### Main Table 1: Stage0 Representation Sufficiency

数据：

- `experiments/table3_fair_fix2_aggregate.json`
- nuScenes 700 scenes
- 3 seeds
- fair 16-slot budget

主表列：

| Method | Budget | DynRoll ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ |
|---|---:|---:|---:|---:|---:|

要突出：

- `object_relation` catastrophic failure;
- `object_relation_decoupled` recovers and surpasses object-only;
- `holistic-full` 作为 97-token upper bound，不参与公平比较。

### Main Figure 1: Type Competition Illustration

建议画概念图：

- 左：shared top-K，object/relation 混合竞争，critical object 被挤掉，relation 成 orphan。
- 右：typed-budget abstraction，dynamic path 与 relation path 分离，relation grounded by selected dynamic agents。

### Main Figure 2: Selection Diagnostics

建议 3-panel：

1. CDR / MissRate across `wm_naive`, `wm_object`, `wm_decoupled_no_vis`
2. WastedRel across `wm_naive` and `wm_decoupled_no_vis`
3. MissRate vs Interaction Recall scatter 或 binned plot

当前关键数：

| Setting | CDR ↑ | MissRate ↓ | WastedRel ↓ | Note |
|---|---:|---:|---:|---|
| `wm_naive` 20k seed7 | 0.481 | 0.519 | 0.068 | low CDR |
| `wm_naive` 20k seed42 | 0.769 | 0.231 | 0.081 | best naive seed |
| `wm_naive` 20k seed123 | 0.391 | 0.609 | 0.499 | orphan relation failure |
| `wm_decoupled_no_vis` 50k seed7 | 0.943 | 0.057 | 0.042 | stable |
| `wm_decoupled_no_vis` 50k seed42 | 0.942 | 0.058 | 0.069 | stable |
| `wm_decoupled_no_vis` 50k seed123 | 0.931 | 0.069 | 0.022 | stable |

### Main Table 2: Stage1 Latent Planning on nuPlan

数据：

- `experiments/nuplan_stage1_50k/summary.md`

主表列：

| Condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|

重点：

- `wm_decoupled_no_vis` 在 50k 上三 seed 稳定正 return；
- `wm_object` 高方差；
- `wm_decoupled(+vis)` 碰撞低但 return 不好，不作为主 winner。

### Main Table 3: Planner-like Offline Sanity

数据：

- `experiments/nuplan_bc_baseline_50k/summary.md`
- `experiments/nuplan_bc_baseline_50k/offline_planner_sanity/summary.md`
- `experiments/nuplan_relation_feature_ablation_50k/summary.md`

作用：

- 回答“不是只在 latent return 里赢”；
- 加入 BC external-style baseline；
- 证明 relation 的 TTC/risk 语义确实影响 teacher-action alignment 和 collision。

### Main Figure 3: Relation Semantics Ablation

数据：

- `experiments/nuplan_relation_feature_ablation_50k/summary.md`

关键数字：

- none: CollRate 约 0.260
- no_ttc_risk: CollRate 约 0.895
- no_lane_priority: 近似不变

主结论：

> The useful relation semantics primarily come from TTC/risk, not from all relation features equally.

### Appendix Figure: Cross-dataset / Regime Analysis

数据：

- `docs/experiment_report.md`
- cross-dataset eval
- interaction-conditioned subset analysis
- horizon sensitivity

主结论：

> Relation-aware abstraction is not universally better in every regime; its benefit appears when downstream planning data and interaction regimes make relation semantics decision-relevant.

这要作为成熟论述，不要藏起来：

> The goal is not a single global winner but a predictable abstraction under limited capacity.

---

## 5. 必要实验清单

### Priority A: 必须完成 / 必须写进主文

#### A1. Selection diagnostic 总结图与表

状态：数据已生成。

路径：

- `experiments/selection_diagnostic_nuplan50k/summary.md`
- `experiments/selection_diagnostic_shared_relation_20k/summary.md`
- per-seed CSV:
  - `token_selection_log.csv`
  - `sample_mechanism_metrics.csv`

还需要做：

- 聚合成一张 paper-ready table。
- 画 CDR/MissRate/WastedRel bar plot。
- 画 MissRate vs Interaction Recall scatter 或 binned plot。
- 不建议主推 RelDensity 链条；可以放 appendix 或 negative finding。

成功标准：

- `wm_naive` 低 CDR / 高 MissRate；
- `wm_decoupled_no_vis` 高 CDR / 低 WastedRel；
- MissRate 与 Interaction Recall 强负相关。

#### A2. Stage0 主表重新包装

状态：已有。

路径：

- `experiments/table3_fair_fix2_aggregate.json`
- `docs/stage0.md`

还需要做：

- 将表名从“Representation Sufficiency Ablation”改成：
  - “Limited-budget object-relation abstraction”
  - 或 “Type competition under fixed world-model context”
- 加入一列或一段解释：shared top-K failure is architectural, not metric artefact。

成功标准：

- Reviewer 能一眼看到 naive shared top-K 在 16-slot 下崩。
- Typed-budget 在相同 K 下恢复动态预测和交互召回。

#### A3. Stage1 nuPlan 50k 主表

状态：已有。

路径：

- `experiments/nuplan_stage1_50k/summary.md`

还需要做：

- 将它作为 downstream validation，而不是全文唯一核心。
- 强调 `wm_decoupled_no_vis` 三 seed 稳定。
- 避免宣称所有数据集都稳定赢。

成功标准：

- 证明 Stage0 机制不是纯 representation toy problem，而能传导到 planning-oriented latent rollout。

#### A4. Relation feature-group ablation

状态：已有。

路径：

- `experiments/nuplan_relation_feature_ablation_50k/summary.md`

还需要做：

- 画一个 3-bar / 4-bar 图：
  - none
  - no_ttc_risk
  - no_lane_priority
  - no_relation_semantics
- 主文结论要克制：
  - TTC/risk 是 relation token 的主要决策语义；
  - lane/priority 当前 tokenization 下贡献小。

成功标准：

- 证明“relation 有用”不是口号，而是某些 relation semantics 有用。

#### A5. 外部 baseline / BC anchor

状态：已有。

路径：

- `experiments/nuplan_bc_baseline_50k/summary.md`

还需要做：

- 主文或 appendix 加一张小表。
- 说明它是 external-style planner-target imitation baseline，不是同族 world-model imagination。

成功标准：

- 回答 reviewer：“是不是只和自己比？”

---

### Priority B: 强烈建议，但可放 appendix

#### B1. Typed-budget sensitivity

状态：已有 `10/6`，已有主设置 `12/4`。

路径：

- `experiments/stage0_budget_10_6_nuscenes/summary.md`

建议补充：

- 若成本低，可补 `8/8` 或 `14/2` 的统一 summary。
- 但不建议为了完整网格耗太多时间。

成功标准：

- 说明 `12/4` 不是孤立拍出来的点；
- relation-heavy 预算会牺牲动态体保留和 rollout。

#### B2. Interaction-conditioned subset analysis

状态：已有。

需要做：

- 统一整理成 appendix 表或 main figure side panel。
- 分桶：
  - low TTC
  - lane conflict
  - dense interaction
  - rare-agent dense

成功标准：

- 证明 relation-aware abstraction 的收益集中在确实需要关系推理的场景，而不是全局平均偶然提升。

#### B3. Cross-dataset / ranking reversal

状态：已有。

需要做：

- 用 discussion 讲清：
  - nuScenes Stage1 上 object-only 更稳；
  - nuPlan 50k 上 decoupled-no-vis 更稳；
  - 这不是矛盾，而是 relation utility depends on downstream planning regime。

成功标准：

- 把负结果转化成成熟发现，而不是论文漏洞。

#### B4. Closed-loop smoke / official metric sanity

状态：已有 MVP / sanity，但不够强。

建议：

- 放 appendix 或 limitations。
- 不要把它包装成正式闭环 SOTA。

成功标准：

- 证明工程接口跑通；
- 避免 reviewer 误以为本文主张 full closed-loop leaderboard。

---

### Priority C: 如果还有时间，可考虑

#### C1. 50k `wm_naive` shared-relation Stage1

当前 `wm_naive` 是 20k，足以做机制诊断，但 reviewer 可能问 50k 是否一致。

是否必须：不是。

只有当资源充足时建议跑：

- nuPlan 50k
- seeds 7/42/123
- condition `wm_naive`

价值：

- 让 selection diagnostic 与 50k 主表完全同 scale。

风险：

- 成本较高；
- 如果训练高方差，仍然可作为 failure mode，但写作更复杂。

#### C2. Learned budget allocation

未来方向，不建议当前主文实现。

例如：

- differentiable budget gate;
- type-wise entropy regularization;
- dynamic K allocation by scene context。

当前不做的理由：

- 会把论文从“清晰诊断 + 简洁修复”变成复杂方法；
- 容易引入新调参问题；
- NeurIPS 主线不需要它。

---

## 6. 不建议主打的实验或说法

### 不建议 1: 强推 RelDensity 链条

原设想：

```text
RelDensity ↑ -> Shared selected relation ratio ↑ -> CDR ↓ -> Rare ADE ↑
```

当前结果不够强：

- `wm_naive` 中 `relation_density vs selected_relation_ratio` 相关性弱；
- relation density 不是当前主要 explanatory variable。

建议改写：

> Relation density alone does not determine failure. The decisive factor is whether relation selection remains grounded by retained dynamic endpoints.

### 不建议 2: 宣称 decoupled 在所有数据集和所有下游都赢

nuScenes Stage1 上 object-only 更稳。应写成：

> Type-aware abstraction improves representation sufficiency and can improve downstream planning when relation semantics align with the planning regime; however, downstream policy learning remains sensitive to dataset/token quality and imagination stability.

这更像 NeurIPS 论文，而不是过度宣传。

### 不建议 3: 把 closed-loop 作为主贡献

当前 closed-loop 证据不足以撑主文 SOTA。可写：

> We focus on representation and latent planning diagnostics; full reactive closed-loop evaluation is future work.

或者 appendix smoke。

### 不建议 4: 把方法说成“调了一个 12/4”

必须说成：

> fixed total context budget + type-aware capacity allocation

数字 `12/4` 是一个实例，不是贡献本身。

---

## 7. 新论文标题候选

首推：

1. **When Relations Compete with Objects: Budget-Aware Token Abstraction for Driving World Models**

备选：

2. **Budget-Aware Object-Relational Abstraction for Decision-Sufficient World Models**
3. **Decision-Sufficient Object-Relation Abstraction under Limited World-Model Context**
4. **Fixing Type Competition in Object-Relational World Models for Autonomous Driving**
5. **Grounding Relations under Limited Context: Type-Aware Abstraction for Driving World Models**

推荐标题 1 的优点：

- 问题感强；
- 一眼看出 contribution；
- 比 DOOR-RL 这个缩写更适合 NeurIPS reviewer。

---

## 8. 新 Abstract 草稿

Object-relational world models offer a structured interface for autonomous driving, but they must often operate under a limited latent context budget. We show that this creates a previously under-examined failure mode: type competition. Object tokens are state-bearing entities, while relation tokens encode sparse but decision-dense cues such as time-to-collision and interaction risk. A unified top-K selector treats these heterogeneous tokens as exchangeable, causing relation tokens to displace critical dynamic agents or become orphaned when their endpoint objects are absent from the latent state. We propose a budget-aware, type-aware token abstraction that allocates separate capacity to dynamic agents and relation edges under the same total world-model context budget. To expose the mechanism, we introduce selection diagnostics including Critical Dynamic Retention and Relation Wasted Slot Rate. Across nuScenes and nuPlan, our abstraction improves limited-budget representation sufficiency, reduces critical-agent misses and wasted relation slots, and yields more stable latent planning performance. Our results suggest that relation-aware driving world models require not only richer relation tokens, but also budget-aware mechanisms that keep those relations grounded in retained dynamic state.

---

## 9. 新 Introduction 结构草稿

### Paragraph 1: Motivation

自动驾驶 world models 越来越使用 object-relational representations，因为规划需要同时理解动态体状态和交互关系。

### Paragraph 2: Bottleneck

但是 latent world model 和 downstream policy 通常不能消费所有 token，必须在有限 context budget 下选择 token。这个选择问题在 object-relational 表示中很关键。

### Paragraph 3: Core observation

Object tokens 和 relation tokens 不是可交换 token。Object 是 state-bearing，relation 是 decision-dense edge，且 relation 的价值依赖 endpoint object 是否保留。统一 top-K 会导致 type competition。

### Paragraph 4: Method

提出 budget-aware type abstraction，在固定总 K 下为 dynamic agents 与 relation edges 分配独立预算，并使用 type-aware supervision。

### Paragraph 5: Diagnostics

提出 selection diagnostics，将“关键动态体是否被保留”“relation 是否 orphan”直接量化。

### Paragraph 6: Results

在 nuScenes Stage0、nuPlan Stage1、offline planner-like sanity、relation semantic ablation 上验证。

### Paragraph 7: Contributions

列 3-4 条贡献。

---

## 10. 最小 NeurIPS 投稿版本应包含什么

如果时间有限，主文最小完整包：

1. Problem framing: type competition under fixed world-model context.
2. Method: budget-aware type abstraction + type-aware loss.
3. Stage0 main table: nuScenes fair 16-slot.
4. Selection diagnostic main figure/table:
   - CDR
   - MissRate
   - WastedRel
   - MissRate vs Interaction Recall
5. Stage1 nuPlan 50k downstream table.
6. Relation feature ablation figure.
7. BC baseline table or appendix.
8. Cross-dataset/ranking reversal discussion.

不需要主文承诺：

- official closed-loop SOTA;
- learned budget allocation;
- exhaustive budget grid;
- universal win on every dataset.

---

## 11. 立即执行清单

### 写作任务

- [ ] 新建论文大纲，按本文档 Section 3 重排。
- [ ] 重写 Abstract。
- [ ] 重写 Introduction，避免从 DOOR-RL 缩写开始。
- [ ] Method 中把 `12/4` 写成 typed-budget instance。
- [ ] 新增 Selection Diagnostics section。
- [ ] Results 按“representation -> mechanism -> planning -> semantics -> generalization”排序。
- [ ] Discussion 中主动解释 nuScenes/nuPlan ranking reversal。

### 图表任务

- [ ] Figure 1: type competition 概念图。
- [ ] Table 1: Stage0 fixed-budget representation table。
- [ ] Figure 2: selection diagnostic CDR/MissRate/WastedRel。
- [ ] Figure 3: MissRate vs Interaction Recall。
- [ ] Table 2: Stage1 nuPlan 50k latent planning。
- [ ] Figure 4: relation semantics ablation。
- [ ] Appendix: BC baseline / budget sensitivity / cross-dataset / closed-loop smoke。

### 实验任务

- [x] Stage0 nuScenes fair 16-slot, 3 seeds。
- [x] Stage1 nuPlan 50k, 3 seeds。
- [x] Stage1 shared-relation 20k, 3 seeds。
- [x] Selection diagnostic 50k object vs decoupled。
- [x] Selection diagnostic 20k naive shared。
- [x] BC baseline 50k。
- [x] Relation feature-group ablation 50k。
- [x] Budget sensitivity 10/6。
- [ ] Optional: `wm_naive` 50k, only if time/resource allows。
- [ ] Optional: paper-ready closed-loop smoke summary。

---

## 12. 最终推荐主线

最终论文主线建议压成一句：

> Limited-capacity object-relational world models fail when object and relation tokens compete under a unified selector; budget-aware type abstraction preserves critical dynamic agents, grounds relation tokens, and yields more reliable prediction and planning.

这条主线比“我们做了 decoupled selector”强，也能自然容纳当前所有实验，包括负结果和机制诊断。

