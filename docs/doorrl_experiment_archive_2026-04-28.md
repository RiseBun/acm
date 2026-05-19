# DOOR-RL 全实验档案：设计、设置与原始数据索引

_Created: 2026-04-28_  
_Purpose: 统一归档当前 DOOR-RL / budget-aware object-relation abstraction 项目已经完成和正在运行的实验，包含实验目的、具体设计、运行设置、原始数据路径和关键原始结果。本文档偏“实验台账”，不是论文正文。_

---

## 0. 总览

### 0.1 当前主线

新版 NeurIPS 主线是：

> 在有限 world-model context budget 下，object tokens 与 relation tokens 的语义角色不同。统一 top-K selector 会产生 type competition：relation tokens 可能挤掉 critical dynamic agents，或者形成 endpoint agent 不在 latent state 中的 orphan relations。Typed-budget / type-aware abstraction 通过给 dynamic tokens 与 relation tokens 分配独立预算，在固定总容量下提升 representation sufficiency，并改善 nuPlan latent planning 与 planner-like 指标。

### 0.2 实验分层

| 层级 | 实验 | 作用 | 状态 |
|---|---|---|---|
| 主结果 | Stage0 nuScenes fair 16-slot | 证明 shared top-K type competition 与 decoupled 修复 | 完成 |
| 主下游 | Stage1 nuPlan 20k/50k | 证明 representation 机制可传导到 latent planning | 完成 |
| 机制诊断 | Selection diagnostic | 证明 CDR/MissRate/WastedRel 是 failure mechanism | 完成 |
| 外部锚点 | BC / planner-target imitation | 回答“是否只和自己比” | 完成 |
| 语义解释 | Relation feature-group ablation | 证明 relation 收益主要来自 TTC/risk | 完成 |
| 预算敏感性 | 10/6 vs 12/4 | 回答 `12/4` 是否拍脑袋 | 完成 |
| 交互子集 | nuPlan interaction-conditioned subsets | 证明交互场景中收益更明显 | 完成 |
| 数据集解释 | nuScenes vs nuPlan token/action stats | 解释 ranking reversal | 完成 |
| 外部 wrapper | PlanTF / nuPlan-devkit sanity | 外部系统集成证明，appendix | 完成 |
| 长任务 | `wm_naive` nuPlan 50k closure | 让 shared-relation naive 与 50k 主表同 scale | 运行中 |

### 0.3 关键目录

| 类型 | 路径 |
|---|---|
| 项目根目录 | `/mnt/volumes/cpfs/prediction/lipeinan/code` |
| 主实验目录 | `experiments/` |
| 论文材料目录 | `期刊/` |
| NeurIPS 重构稿 | `期刊/doorrl_neurips_draft_v1_budget_aware_abstraction_2026-04-28.md` |
| NeurIPS 重构方案 | `期刊/neurips_reframe_budget_aware_object_relation_abstraction_2026-04-28.md` |
| Paper-ready closure figures | `期刊/paper_assets/neurips_closure_2026-04-28/` |
| 当前文档 | `docs/doorrl_experiment_archive_2026-04-28.md` |

---

## 1. Stage0：nuScenes 16-slot 表示充分性主实验

### 1.1 实验目的

回答：

1. 在固定 `K=16` world-model context 下，naive shared object+relation top-K 是否会失败？
2. Typed-budget decoupled abstraction 是否能在相同总容量下保留 dynamic agents 与 relation semantics？
3. 加 relation token 后的失败是否是 relation token 本身坏，还是 shared budget 下 type competition 导致 dynamic slots 被挤掉？

### 1.2 具体设计

| 项 | 设置 |
|---|---|
| 数据集 | nuScenes v1.0 trainval + CAN bus |
| 规模 | 700 scenes / 28,096 samples |
| split | scene-level 80/20，560 train scenes / 140 val scenes |
| token schema | 97 tokens × 40 raw dims |
| world-model context | 16 slots |
| seeds | 7, 42, 2026 |
| epochs | 15 |
| batch size | 32 |
| optimizer | Adam |
| lr | 1e-3 |
| loss | typed obs + reward + continue + collision + BC |
| entry script | `run_stage0_table3.py` |

### 1.3 对比条件

| Variant | 设计 |
|---|---|
| `holistic` | 97-token full context，上界参考 |
| `holistic_16slot` | 16 learned query slots |
| `object_only` | top-K over dynamic tokens only |
| `object_relation` | shared top-K over dynamic ∪ relation |
| `object_relation_visibility` | shared top-K + visibility weighting |
| `object_relation_decoupled` | K_dyn=12, K_rel=4 |
| `object_relation_decoupled_visibility` | K_dyn=12, K_rel=4 + visibility |

### 1.4 指标设计

| 指标 | 含义 |
|---|---|
| DynRoll ↓ | dynamic agent next-state rollout MSE，nearest dynamic-slot matching |
| Action MSE ↓ | policy action mean vs teacher action MSE |
| Coll F1 ↑ | predicted collision vs relation-derived collision label |
| Rare ADE ↓ | pedestrian/cyclist nearest-match ADE |
| IntRec@1m ↑ | 20m 内 rare agent 在 1m 内命中的比例 |

### 1.5 原始数据路径

| 数据 | 路径 |
|---|---|
| 聚合 JSON | `experiments/table3_fair_fix2_aggregate.json` |
| seed7 raw | `experiments/table3_fair_fix2_seed7/table3_complete.json` |
| seed42 raw | `experiments/table3_fair_fix2_seed42/table3_complete.json` |
| seed2026 raw | `experiments/table3_fair_fix2_seed2026/table3_complete.json` |
| 详细文档 | `docs/stage0.md` |
| 旧论文表述 | `期刊/doorrl_tiv_draft_v1.md` |

### 1.6 聚合结果

| Variant | Ctx | DynRoll ↓ | Action MSE ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Holistic-16Slot | 16 | 2.1059 ± 0.1600 | 0.2875 ± 0.0145 | 0.9782 ± 0.0105 | 1.4215 ± 0.0119 | 0.6433 ± 0.0153 |
| Object-only-16 | 16 | 3.7449 ± 1.0099 | 0.2854 ± 0.0103 | 0.9463 ± 0.0041 | 1.0964 ± 0.1159 | 0.9009 ± 0.0335 |
| Object+Relation-16 naive | 16 | 40.2822 ± 29.5376 | 0.2808 ± 0.0130 | 0.9803 ± 0.0125 | 7.5060 ± 5.4799 | 0.4295 ± 0.4074 |
| Obj+Rel+Vis-16 | 16 | 15.8023 ± 9.9254 | 0.2840 ± 0.0173 | 0.9330 ± 0.0641 | 2.9624 ± 1.6389 | 0.7283 ± 0.1545 |
| Obj+Rel-Decoupled | 16 | 2.1148 ± 0.1889 | 0.2805 ± 0.0125 | 0.9285 ± 0.0389 | 0.4913 ± 0.1768 | 0.9842 ± 0.0135 |
| Decoupled+Visibility | 16 | 1.8761 ± 0.2271 | 0.2843 ± 0.0234 | 0.9257 ± 0.0290 | 0.5197 ± 0.0495 | 0.9787 ± 0.0078 |
| Holistic-full reference | 97 | 0.1070 ± 0.1165 | 0.2858 ± 0.0112 | 0.9875 ± 0.0057 | 0.2562 ± 0.0234 | 1.0000 ± 0.0000 |

### 1.7 Per-seed 原始结果

Seed 7:

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---:|---:|---:|---:|
| holistic_16slot | 1.951 | 0.983 | 1.408 | 0.628 |
| object_only | 4.398 | 0.949 | 1.005 | 0.919 |
| object_relation | 60.786 | 0.967 | 9.744 | 0.263 |
| object_relation_visibility | 24.892 | 0.958 | 2.923 | 0.746 |
| object_relation_decoupled | 1.918 | 0.917 | 0.695 | 0.969 |
| object_relation_decoupled_visibility | 1.775 | 0.915 | 0.576 | 0.975 |
| holistic | 0.043 | 0.983 | 0.281 | 1.000 |

Seed 42:

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---:|---:|---:|---:|
| holistic_16slot | 2.270 | 0.985 | 1.425 | 0.659 |
| object_only | 4.255 | 0.942 | 1.227 | 0.862 |
| object_relation | 53.635 | 0.982 | 11.512 | 0.132 |
| object_relation_visibility | 17.302 | 0.860 | 4.621 | 0.566 |
| object_relation_decoupled | 2.295 | 0.972 | 0.396 | 0.989 |
| object_relation_decoupled_visibility | 2.136 | 0.958 | 0.482 | 0.988 |
| holistic | 0.241 | 0.994 | 0.234 | 1.000 |

Seed 2026:

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---:|---:|---:|---:|
| holistic_16slot | 2.096 | 0.966 | 1.431 | 0.643 |
| object_only | 2.582 | 0.948 | 1.057 | 0.921 |
| object_relation | 6.426 | 0.992 | 1.261 | 0.894 |
| object_relation_visibility | 5.212 | 0.981 | 1.343 | 0.873 |
| object_relation_decoupled | 2.131 | 0.897 | 0.383 | 0.995 |
| object_relation_decoupled_visibility | 1.717 | 0.903 | 0.502 | 0.974 |
| holistic | 0.036 | 0.986 | 0.254 | 1.000 |

### 1.8 结论

- Naive shared object+relation 在 16-slot 下高方差且均值崩坏，是 type competition 的主证据。
- Decoupled typed-budget 在相同 K 下显著降低 Rare ADE，提高 IntRec@1m。
- Holistic-full 作为 97-token 上界说明 16-slot bottleneck 是实际限制。

---

## 2. Stage1：nuScenes latent imagination RL

### 2.1 实验目的

检验 Stage0 representation sufficiency 是否自动转化为下游 latent imagination policy learning。

### 2.2 设计

| 项 | 设置 |
|---|---|
| 数据集 | nuScenes 700 scenes |
| split | scene-level 80/20 |
| horizon | K=5 |
| epochs | 10 |
| batch size | 128 |
| lr | 4e-3 |
| seeds | 7, 42, 123 |
| warm-start | 同 variant Stage0 checkpoint |
| entry script | `run_stage1_table4.py` |

### 2.3 原始数据路径

| 数据 | 路径 |
|---|---|
| aggregate | `experiments/stage1_pilot_x/X_summary.json` |
| per-seed metrics | `experiments/stage1_pilot_x/seed*/<condition>/stage1_metrics.json` |
| detailed record | `docs/experiment_report.md` |

### 2.4 结果

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) |
|---|---:|---:|---:|---:|
| wm_object | 31.79 ± 19.70 | 0.597 ± 0.048 | 0.610 ± 0.048 | 0.258 ± 0.058 |
| wm_decoupled | 4.34 ± 13.66 | 0.695 ± 0.283 | 0.676 ± 0.243 | 0.636 ± 0.108 |
| wm_decoupled_no_vis | 0.34 ± 15.97 | 0.820 ± 0.260 | 0.814 ± 0.167 | 0.223 ± 0.033 |

### 2.5 结论

- nuScenes Stage1 上 `wm_object` 更稳，说明 Stage0 表示优势不会自动转化为所有下游 regime 的策略优势。
- 该结果应在论文中作为 cross-dataset / planning-regime dependence 的负结果，而不是隐藏。

---

## 3. Stage1：nuPlan 5k pilot

### 3.1 实验目的

测试 nuPlan setting 下 relation-aware abstraction 是否出现与 nuScenes 不同的下游排序。

### 3.2 设计

| 项 | 设置 |
|---|---|
| 数据集 | nuPlan preprocessed NPZ |
| 规模 | 5,000 samples |
| split | 4,000 train / 1,000 val |
| seeds | 7, 42, 123 |
| horizon | 5 |
| entropy_beta | 0.003 |
| action_clip | 5 |
| warm-start | `experiments/nuplan_stage0_5k_seed7` |

### 3.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_stage1_5k/summary.md` |
| summary JSON | `experiments/nuplan_stage1_5k/summary.json` |
| per-seed metrics | `experiments/nuplan_stage1_5k/seed*/<condition>/stage1_metrics.json` |

### 3.4 结果

| condition | Return | CollRate | CollMean | Stab(ego-cos) |
|---|---:|---:|---:|---:|
| wm_object | -6.011 ± 3.101 | 0.348 ± 0.254 | 0.384 ± 0.150 | 0.505 ± 0.441 |
| wm_decoupled | 9.383 ± 9.586 | 0.215 ± 0.095 | 0.281 ± 0.097 | 0.887 ± 0.011 |
| wm_decoupled_no_vis | 12.907 ± 2.687 | 0.247 ± 0.029 | 0.325 ± 0.060 | 0.097 ± 0.062 |
| wm_decoupled_rel_to_critic_only | 5.460 ± 2.258 | 0.231 ± 0.052 | 0.337 ± 0.025 | 0.848 ± 0.058 |

### 3.5 结论

- nuPlan 5k pilot 已经显示 ranking reversal：object-only 不再是最强条件。
- `wm_decoupled_no_vis` 在 Return 上最好，提供后续 20k/50k scale-up 动机。

---

## 4. Stage1：nuPlan 20k scale-up

### 4.1 目的

验证 nuPlan 5k 结果是否能扩展到更大子集。

### 4.2 设计

| 项 | 设置 |
|---|---|
| 数据集 | nuPlan balanced NPZ |
| 规模 | 20,000 samples |
| seeds | 7, 42, 123 |
| epochs | 10 |
| batch size | 128 |
| horizon | 5 |
| warm-start | `experiments/nuplan_stage0_20k_seed7` |
| tokenisation workers | 32 |

### 4.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_stage1_20k/summary.md` |
| summary JSON | `experiments/nuplan_stage1_20k/summary.json` |
| per-seed metrics | `experiments/nuplan_stage1_20k/seed*/<condition>/stage1_metrics.json` |

### 4.4 结果

| condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|
| wm_object | 4.738 ± 13.946 | 0.373 ± 0.083 | 0.395 | 0.426 |
| wm_decoupled | 13.477 ± 4.086 | 0.488 ± 0.217 | 0.509 | 0.546 |
| wm_decoupled_no_vis | 17.497 ± 1.374 | 0.226 ± 0.105 | 0.251 | 0.136 |

Per-seed Return:

| condition | seed7 | seed42 | seed123 |
|---|---:|---:|---:|
| wm_object | 20.581 | -0.694 | -5.674 |
| wm_decoupled | 15.707 | 15.963 | 8.760 |
| wm_decoupled_no_vis | 16.222 | 17.319 | 18.952 |

### 4.5 结论

- `wm_decoupled_no_vis` 是 20k 最强条件：Return 最高、collision 最低、Return 方差最低。
- visibility weighting 在 nuPlan 上不如 no-vis，说明 visibility 是 dataset-dependent inductive bias。

---

## 5. Stage1：nuPlan 50k 主 scale-up

### 5.1 目的

验证 typed-budget no-vis 在更大 nuPlan subset 上是否稳定，并作为 NeurIPS 主下游表。

### 5.2 设计

| 项 | 设置 |
|---|---|
| 数据集 | nuPlan balanced NPZ |
| 规模 | 50,000 samples |
| seeds | 7, 42, 123 |
| epochs | 10 |
| batch size | 128 |
| horizon | 5 |
| warm-start | `experiments/nuplan_stage0_50k_seed7` |
| lazy loading | 32 DataLoader workers per seed |

### 5.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_stage1_50k/summary.md` |
| summary JSON | `experiments/nuplan_stage1_50k/summary.json` |
| per-seed metrics | `experiments/nuplan_stage1_50k/seed*/<condition>/stage1_metrics.json` |

### 5.4 结果

| condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|
| wm_object | 1.723 ± 17.886 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 |
| wm_decoupled | -0.330 ± 4.936 | 0.007 ± 0.012 | 0.277 ± 0.111 | 0.255 |
| wm_decoupled_no_vis | 14.511 ± 2.925 | 0.259 ± 0.045 | 0.277 ± 0.033 | 0.222 |

Per-seed Return:

| condition | seed7 | seed42 | seed123 |
|---|---:|---:|---:|
| wm_object | 1.834 | 19.554 | -16.218 |
| wm_decoupled | -5.308 | -0.245 | 4.562 |
| wm_decoupled_no_vis | 15.833 | 16.542 | 11.158 |

### 5.5 结论

- `wm_decoupled_no_vis` 是 50k 稳定主条件，三 seed 均正 Return。
- `wm_object` 高方差，seed123 collapse。
- `wm_decoupled(+vis)` collision 极低但 Return 接近零，不适合作为最终主胜者。

---

## 6. Shared-relation Stage1 baseline：nuPlan 20k

### 6.1 目的

补齐下游机制链：

1. Stage0 已证明 shared relation mixing 在表示层失败；
2. Stage1 需要证明 shared relation policy learning 也高方差/弱于 typed-budget；
3. 为 selection diagnostic 的 `wm_naive` 提供 checkpoint。

### 6.2 设计

| 项 | 设置 |
|---|---|
| condition | `wm_naive` |
| variant | `object_relation` shared top-K |
| 数据集 | nuPlan balanced NPZ |
| 规模 | 20,000 samples |
| seeds | 7, 42, 123 |
| warm-start | `experiments/nuplan_stage0_20k_seed7/object_relation` |

### 6.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_stage1_shared_relation_20k/summary.md` |
| summary JSON | `experiments/nuplan_stage1_shared_relation_20k/summary.json` |
| per-seed metrics | `experiments/nuplan_stage1_shared_relation_20k/seed*/wm_naive/stage1_metrics.json` |

### 6.4 结果

| condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|
| wm_naive | -2.655 ± 14.632 | 0.252 ± 0.050 | 0.297 ± 0.069 | 0.186 |

Per-seed Return:

| seed | Return | ImagColl |
|---:|---:|---:|
| 7 | -15.597 | 0.263 |
| 42 | 13.222 | 0.198 |
| 123 | -5.591 | 0.297 |

### 6.5 结论

- `wm_naive` runnable 但高方差，弱于 `wm_decoupled_no_vis` 20k。
- 与 Stage0 shared-relation warm-start 弱表现一致，支持 slot-mixing failure story。

---

## 7. Selection diagnostic：50k 主诊断 + 20k shared-relation 诊断

### 7.1 目的

把 type competition 从“表格现象”升级为可量化机制：

- selector 是否保留 critical dynamic agents？
- selected relation 是否仍然 grounded by endpoint agents？
- MissRate 是否和 downstream interaction error 相关？

### 7.2 具体设计

每个 sample / token 保存：

```text
sample_id, dataset, scene_id, token_id, token_type, score, selected,
distance_to_ego, ttc_proxy, lane_conflict, visibility, is_rare_agent,
relation_endpoint_i, relation_endpoint_j
```

定义：

```text
C_dyn(t) = {i : d_i < 20m or TTC_i < 3s or lane_conflict_i = 1 or rare-agent near ego}
CDR = |S_dyn ∩ C_dyn| / |C_dyn|
MissRate = 1 - CDR
WastedRel = selected relations whose non-ego endpoint is not selected / selected relations
ROI = |S_rel| / K
RelDensity = |R| / (|O| + 1)
```

### 7.3 实验设置

| 诊断 | 数据 | 条件 | seeds | 样本 |
|---|---|---|---|---:|
| 主诊断 | nuPlan 50k val | `wm_object`, `wm_decoupled_no_vis` | 7, 42, 123 | 10,000/seed |
| shared 诊断 | nuPlan 20k val | `wm_naive` | 7, 42, 123 | 4,000/seed |

### 7.4 原始数据路径

| 数据 | 路径 |
|---|---|
| 50k summary | `experiments/selection_diagnostic_nuplan50k/summary.md` |
| 50k summary JSON | `experiments/selection_diagnostic_nuplan50k/summary.json` |
| 50k per-seed token log | `experiments/selection_diagnostic_nuplan50k/seed*/<condition>/token_selection_log.csv` |
| 50k per-seed sample metrics | `experiments/selection_diagnostic_nuplan50k/seed*/<condition>/sample_mechanism_metrics.csv` |
| 20k shared summary | `experiments/selection_diagnostic_shared_relation_20k/summary.md` |
| 20k shared summary JSON | `experiments/selection_diagnostic_shared_relation_20k/summary.json` |
| 20k shared token log | `experiments/selection_diagnostic_shared_relation_20k/seed*/wm_naive/token_selection_log.csv` |
| 20k shared sample metrics | `experiments/selection_diagnostic_shared_relation_20k/seed*/wm_naive/sample_mechanism_metrics.csv` |
| figure/table artifacts | `期刊/paper_assets/neurips_closure_2026-04-28/` |

### 7.5 原始结果

| condition | dataset | seed | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI |
|---|---|---:|---:|---:|---:|---:|
| wm_naive | nuPlan20k | 7 | 0.4806 | 0.5194 | 0.0681 | 0.0312 |
| wm_naive | nuPlan20k | 42 | 0.7692 | 0.2308 | 0.0810 | 0.0537 |
| wm_naive | nuPlan20k | 123 | 0.3915 | 0.6085 | 0.4993 | 0.0776 |
| wm_object | nuPlan50k | 7 | 0.893 | 0.107 | N/A | 0.000 |
| wm_object | nuPlan50k | 42 | 0.930 | 0.070 | N/A | 0.000 |
| wm_object | nuPlan50k | 123 | 0.533 | 0.467 | N/A | 0.000 |
| wm_decoupled_no_vis | nuPlan50k | 7 | 0.943 | 0.057 | 0.042 | 0.260 |
| wm_decoupled_no_vis | nuPlan50k | 42 | 0.942 | 0.058 | 0.069 | 0.260 |
| wm_decoupled_no_vis | nuPlan50k | 123 | 0.931 | 0.069 | 0.022 | 0.260 |

20k shared-relation terminal result confirmed on 2026-04-28:

| condition | dataset | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wm_naive | nuPlan20k | 7 | 4000 | 0.48057149 | 0.51942851 | 0.06813866 | 0.031203125 | 0.81236375 | 0.19634427 | 0.0021945832 |
| wm_naive | nuPlan20k | 42 | 4000 | 0.76918973 | 0.23081027 | 0.080979284 | 0.053703125 | 0.81157334 | 0.12740495 | 0.064308212 |
| wm_naive | nuPlan20k | 123 | 4000 | 0.39146155 | 0.60853845 | 0.49931852 | 0.07759375 | 0.81428394 | 0.18470555 | 0.11047333 |

20k shared-relation aggregate:

| condition | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI |
|---|---:|---:|---:|---:|
| wm_naive 20k | 0.547 | 0.453 | 0.216 | 0.054 |

Aggregated closure table:

| condition | CDR mean | MissRate mean | WastedRel mean | ROI mean | MissRate~IntRec rho |
|---|---:|---:|---:|---:|---:|
| wm_naive | 0.547 | 0.453 | 0.216 | 0.054 | -0.680 |
| wm_object | 0.785 | 0.215 | N/A | 0.000 | -0.519 |
| wm_decoupled_no_vis | 0.939 | 0.061 | 0.044 | 0.260 | -0.404 |

### 7.6 结论

- `wm_naive` 低 CDR、高 MissRate，尤其 seed123 的 WastedRel 达到 0.499。
- `wm_decoupled_no_vis` 三 seed CDR 稳定在 0.931-0.943，WastedRel 低。
- MissRate 与 Interaction Recall 强负相关，是当前最强 mechanism link。
- RelDensity 链条不强，不建议作为主结论。

---

## 8. Offline planner-like sanity：nuPlan 50k

### 8.1 目的

验证 latent planning 优势是否延伸到 planner-like teacher-action alignment，而不是只看 latent return。

### 8.2 设计

| 项 | 设置 |
|---|---|
| 数据 | nuPlan 50k val split |
| seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled_no_vis` |
| horizon | 5 |
| loader workers | 32 |
| 类型 | offline planner-like sanity，不是 closed-loop |

### 8.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_planner_sanity_50k/summary.md` |
| summary JSON | `experiments/nuplan_planner_sanity_50k/summary.json` |
| per-seed planner sanity | `experiments/nuplan_planner_sanity_50k/seed*/<condition>/planner_sanity.json` |

### 8.4 结果

| condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Return ↑ | CollRate ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|---:|---:|
| wm_object | 8.863 ± 0.370 | 3.553 ± 0.118 | 1.722 ± 17.888 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 |
| wm_decoupled_no_vis | 6.628 ± 0.110 | 2.115 ± 0.083 | 14.512 ± 2.925 | 0.259 ± 0.045 | 0.277 ± 0.033 | 0.222 |

### 8.5 结论

- `wm_decoupled_no_vis` 显著降低 teacher action MSE 和 collision proxy。
- 该实验是 downstream offline evidence，不可等同正式闭环 benchmark。

---

## 9. BC / planner-target imitation baseline：nuPlan 50k

### 9.1 目的

提供外部风格 baseline，回答 reviewer 可能提出的“是不是只和自己家族比”。

### 9.2 设计

| 项 | 设置 |
|---|---|
| condition | `bc` |
| variant | `object_only` |
| 数据 | nuPlan 50k balanced |
| seeds | 7, 42, 123 |
| epochs | 10 |
| batch size | 128 |
| horizon | 5 |
| warm-start | `experiments/nuplan_stage0_50k_seed7/object_only` |

### 9.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_bc_baseline_50k/summary.md` |
| summary JSON | `experiments/nuplan_bc_baseline_50k/summary.json` |
| per-seed metrics | `experiments/nuplan_bc_baseline_50k/seed*/bc/stage1_metrics.json` |
| offline sanity | `experiments/nuplan_bc_baseline_50k/offline_planner_sanity/summary.md` |

### 9.4 结果

| condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Teacher action MSE ↓ | Action ΔL2 ↓ |
|---|---:|---:|---:|---:|---:|
| bc | 0.433 ± 4.230 | 0.327 ± 0.566 | 0.479 ± 0.368 | 8.824 ± 0.113 | 3.534 ± 0.017 |

Per-seed:

| seed | Return | ImagColl | CollMean | Teacher MSE | Action ΔL2 |
|---:|---:|---:|---:|---:|---:|
| 7 | 5.308 | 0.980 | 0.902 | 8.953 | 3.553 |
| 42 | -2.255 | 0.000 | 0.299 | 8.744 | 3.523 |
| 123 | -1.754 | 0.000 | 0.237 | 8.776 | 3.525 |

### 9.5 结论

- BC baseline 可作为 external-style anchor，但不具竞争力。
- `wm_decoupled_no_vis` 在 Return 和 teacher-action alignment 上更强。

---

## 10. Relation feature-group ablation：nuPlan 50k

### 10.1 目的

回答：

1. relation token 的收益来自哪些语义？
2. 是所有 relation features 都重要，还是 TTC/risk 是核心？

### 10.2 设计

| 项 | 设置 |
|---|---|
| 类型 | evaluation-time ablation，不重训 |
| checkpoint | trained Stage1 `wm_decoupled_no_vis` |
| 数据 | nuPlan 50k val split |
| seeds | 7, 42, 123 |
| horizon | 5 |
| ablations | none, no_ttc_risk, no_lane_priority, no_relation_semantics |

### 10.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_relation_feature_ablation_50k/summary.md` |
| summary JSON | `experiments/nuplan_relation_feature_ablation_50k/summary.json` |
| per-seed metrics | `experiments/nuplan_relation_feature_ablation_50k/seed*/wm_decoupled_no_vis/<ablation>/metrics.json` |
| figure artifact | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_relation_semantics_ablation.png` |

### 10.4 结果

| ablation | Teacher MSE ↓ | Action ΔL2 ↓ | Return ↑ | CollRate ↓ | CollMean ↓ |
|---|---:|---:|---:|---:|---:|
| none | 6.628 ± 0.110 | 2.115 ± 0.083 | 14.510 ± 2.921 | 0.260 ± 0.045 | 0.277 ± 0.033 |
| no_ttc_risk | 7.992 ± 0.805 | 3.154 ± 0.335 | 8.498 ± 1.397 | 0.895 ± 0.126 | 0.891 ± 0.126 |
| no_lane_priority | 6.627 ± 0.111 | 2.113 ± 0.081 | 14.526 ± 2.931 | 0.257 ± 0.052 | 0.274 ± 0.040 |
| no_relation_semantics | 7.989 ± 0.827 | 3.151 ± 0.351 | 8.724 ± 1.629 | 0.894 ± 0.126 | 0.888 ± 0.127 |

### 10.5 结论

- TTC/risk 是 relation token 最关键的决策语义。
- lane/priority 在当前 tokenizer 和数据设置下近似中性。

---

## 11. Typed-budget sensitivity：10/6 vs 12/4

### 11.1 目的

回答 `12/4` 是否 arbitrary，检验更 relation-heavy 的 `10/6` 是否更好。

### 11.2 设计

| 项 | 设置 |
|---|---|
| 数据 | nuScenes 700 scenes |
| variant | `object_relation_decoupled_visibility` |
| budget | `top_k_dyn=10`, `top_k_rel=6` |
| 对照 | `12/4 main` |
| seeds | 7, 42, 2026 |
| epochs | 15 |
| batch size | 32 |

### 11.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/stage0_budget_10_6_nuscenes/summary.md` |
| summary JSON | `experiments/stage0_budget_10_6_nuscenes/summary.json` |
| per-seed raw | `experiments/stage0_budget_10_6_nuscenes/seed*/object_relation_decoupled_visibility/table3_results.json` |
| figure artifact | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_budget_sensitivity_12_4_vs_10_6.png` |

### 11.4 结果

| budget | Dyn Rollout MSE ↓ | Action MSE ↓ | Collision F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|
| 10/6 | 8.761 ± 1.067 | 0.286 ± 0.009 | 0.894 ± 0.007 | 1.574 ± 0.485 | 0.884 ± 0.043 |
| 12/4 main | 1.876 ± 0.227 | 0.284 ± 0.023 | 0.926 ± 0.029 | 0.520 ± 0.049 | 0.979 ± 0.008 |

### 11.5 结论

- Relation-heavy `10/6` 可运行，但显著弱于 `12/4`。
- 支持 `12/4` 是更稳的默认 typed budget，而非孤立拍点。

---

## 12. Interaction-conditioned subset analysis：nuPlan 50k

### 12.1 目的

检验 relation-aware abstraction 的收益是否集中在真正需要交互建模的场景。

### 12.2 设计

| 项 | 设置 |
|---|---|
| 数据 | nuPlan 50k val split |
| checkpoint | existing Stage1 50k |
| conditions | `wm_object`, `wm_decoupled_no_vis` |
| subsets | dense_agents, high_interaction_union, lane_conflict, low_ttc_proxy, rare_agent_dense |
| 类型 | offline validation subset analysis |

### 12.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/nuplan_interaction_subset_50k/summary.md` |
| summary JSON | `experiments/nuplan_interaction_subset_50k/summary.json` |

### 12.4 结果

| subset | condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---|---:|---:|---:|---:|
| all_val | wm_object | 10000 | 8.863 ± 0.370 | 1.718 ± 17.895 | 0.610 ± 0.146 |
| all_val | wm_decoupled_no_vis | 10000 | 6.628 ± 0.110 | 14.512 ± 2.927 | 0.260 ± 0.045 |
| dense_agents | wm_object | 8690 | 8.692 ± 0.414 | 1.423 ± 17.975 | 0.637 ± 0.170 |
| dense_agents | wm_decoupled_no_vis | 8690 | 6.340 ± 0.170 | 14.311 ± 3.127 | 0.272 ± 0.049 |
| lane_conflict | wm_object | 6699 | 7.023 ± 0.393 | 0.943 ± 17.975 | 0.591 ± 0.176 |
| lane_conflict | wm_decoupled_no_vis | 6699 | 4.225 ± 0.125 | 13.330 ± 3.134 | 0.205 ± 0.043 |
| low_ttc_proxy | wm_object | 4745 | 12.796 ± 0.425 | 2.650 ± 18.243 | 0.791 ± 0.084 |
| low_ttc_proxy | wm_decoupled_no_vis | 4745 | 11.534 ± 0.198 | 15.946 ± 2.851 | 0.541 ± 0.090 |
| rare_agent_dense | wm_object | 7280 | 8.729 ± 0.369 | 1.562 ± 17.985 | 0.626 ± 0.149 |
| rare_agent_dense | wm_decoupled_no_vis | 7280 | 6.494 ± 0.137 | 14.496 ± 3.085 | 0.273 ± 0.045 |

### 12.5 结论

- `wm_decoupled_no_vis` 在所有 interaction-conditioned subsets 中均优于 `wm_object`。
- lane_conflict 和 low_ttc_proxy 是最适合支撑 relation-aware abstraction 的子集。

---

## 13. Dataset statistics：nuScenes vs nuPlan

### 13.1 目的

解释为什么 Stage1 在 nuScenes 与 nuPlan 上排序不同。

### 13.2 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/dataset_token_stats/summary.md` |
| summary JSON | `experiments/dataset_token_stats/summary.json` |

### 13.3 结果

| Statistic | nuScenes 700 scenes | nuPlan 50k NPZ | Reading |
|---|---:|---:|---|
| Dynamic tokens / sample | 9.715 ± 3.847 / p90 13.000 | 12.164 ± 2.457 / p90 13.000 | planning density / interaction budget |
| Rare tokens / sample | 2.439 ± 2.906 / p90 7.000 | 3.934 ± 3.490 / p90 9.000 | pedestrian/cyclist pressure |
| Relation tokens / sample | 11.318 ± 2.118 / p90 12.000 | 11.164 ± 2.457 / p90 12.000 | relation-context availability |
| Dynamic visibility | 0.746 ± 0.265 / p90 1.000 | 1.000 ± 0.000 / p90 1.000 | whether visibility weighting has signal |
| Relation TTC | 15.363 ± 7.210 / p90 20.000 | 16.727 ± 6.124 / p90 20.000 | risk/interaction feature scale |
| Teacher action L2 | 0.539 ± 0.632 / p90 1.183 | 3.420 ± 4.178 / p90 10.275 | action-label scale and policy target |

### 13.4 结论

- nuPlan action label 和 dynamic/rare token pressure 与 nuScenes 不同。
- visibility 在 nuPlan 中恒为 1，因此 visibility weighting 在 nuPlan 中没有有效差异信号，解释 no-vis 更稳。

---

## 14. PlanTF external nuPlan-devkit sanity

### 14.1 目的

证明外部 PlanTF checkpoint 能通过 nuPlan-devkit wrapper 跑通，作为 appendix / integration proof。

### 14.2 设计

| 项 | 设置 |
|---|---|
| baseline | PlanTF external checkpoint |
| simulator | nuPlan-devkit closed_loop_nonreactive_agents |
| scenario set | fixed 13 one_of_each_scenario_type tokens |
| max_iterations | 20 |
| workers | 4 |
| metrics | main 13-scenario run disabled metrics to avoid overhead |

### 14.3 原始数据路径

| 数据 | 路径 |
|---|---|
| summary | `experiments/plantf_external_baseline_nuplan/summary.md` |
| summary JSON | `experiments/plantf_external_baseline_nuplan/summary.json` |
| runner report | `cangku/planTF/experiments/exp/simulation/closed_loop_nonreactive_agents/plantf_baseline/one_of_each_h20_20step_tokens_nometric_nosimlog/runner_report.parquet` |
| log | `cangku/planTF/experiments/plantf_baseline_smoke/logs/one_of_each_h20_20step_tokens_nometric_nosimlog.log` |

### 14.4 结果

| item | value |
|---|---:|
| scenarios | 13 |
| successful | 13 |
| failed | 0 |
| success rate | 1.000 |
| mean duration | 397.5 s |

Official metric smoke:

| item | value |
|---|---:|
| scenario | `8de10fd86b825304` |
| succeeded | true |
| duration | 232.0 s |
| final score | 0.0 |

### 14.5 结论

- PlanTF wrapper 跑通，可作为 appendix external integration sanity。
- 官方 metric 单场景 score 为 0，不应作为主性能证据。

---

## 15. Closed-loop MVP / smoke experiments

### 15.1 目的

验证 DOOR-RL checkpoint 能否接入 nuPlan closed-loop wrapper，作为“接口可跑通”的工程证明。

### 15.2 原始数据路径

当前存在多个 smoke / corridor / projection / safety projection 目录：

- `experiments/nuplan_closed_loop_mvp_1scenario_metrics/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_mvp_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_action_rollout_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_action_rollout_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_action_rollout_50scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_safety_projection_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_safety_projection_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_corridor_projection_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`
- `experiments/nuplan_closed_loop_corridor_yield_1scenario_v2/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md`

### 15.3 结论

- 这些实验目前定位为 closed-loop smoke / engineering sanity。
- 不建议作为主文 performance claim；可在 appendix 或 limitation 中说明。

---

## 16. Paper-ready closure artifacts

### 16.1 目的

把已有实验结果转成 NeurIPS 重构所需图表和表格。

### 16.2 生成脚本

| 项 | 路径 |
|---|---|
| script | `scripts/make_neurips_closure_artifacts.py` |
| output root | `期刊/paper_assets/neurips_closure_2026-04-28/` |

### 16.3 输出

| Artifact | 路径 |
|---|---|
| Selection diagnostic bars | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_selection_diagnostics_bars.png` |
| MissRate vs Interaction Recall | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_missrate_vs_interaction_recall.png` |
| Relation semantics ablation | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_relation_semantics_ablation.png` |
| Budget sensitivity | `期刊/paper_assets/neurips_closure_2026-04-28/figures/fig_budget_sensitivity_12_4_vs_10_6.png` |
| Selection diagnostic table | `期刊/paper_assets/neurips_closure_2026-04-28/tables/selection_diagnostic_table.csv` |
| Closure summary | `期刊/paper_assets/neurips_closure_2026-04-28/summary.md` |

### 16.4 汇总结果

| condition | CDR mean | MissRate mean | WastedRel mean | ROI mean | MissRate~IntRec rho |
|---|---:|---:|---:|---:|---:|
| wm_naive | 0.547 | 0.453 | 0.216 | 0.054 | -0.680 |
| wm_object | 0.785 | 0.215 | N/A | 0.000 | -0.519 |
| wm_decoupled_no_vis | 0.939 | 0.061 | 0.044 | 0.260 | -0.404 |

---

## 17. 当前正在运行：`wm_naive` nuPlan 50k closure

### 17.1 目的

把 shared-relation `wm_naive` 与 50k 主表对齐到同一 scale，增强 NeurIPS 主文可比性。

### 17.2 任务链

脚本：

- `scripts/run_wm_naive_50k_closure.sh`

执行顺序：

1. 训练缺失的 Stage0 `object_relation` 50k warm-start。
2. 跑 Stage1 `wm_naive` 50k seeds 7/42/123。
3. 跑 `wm_naive` 50k selection diagnostic。

### 17.3 当前状态

截至本文档创建时：

- Stage0 `object_relation` 50k 已完成。
- 原始结果：`experiments/nuplan_stage0_50k_seed7/object_relation/table3_results.json`
- checkpoint：`experiments/nuplan_stage0_50k_seed7/object_relation/model.pt`
- Stage1 `wm_naive` 50k seeds 7/42/123 已完成。
- Selection diagnostic `wm_naive` 50k 已完成。

### 17.4 日志和未来结果路径

| 数据 | 路径 |
|---|---|
| logs | `experiments/closure_wm_naive_50k_logs/` |
| Stage1 output | `experiments/nuplan_stage1_shared_relation_50k/` |
| Selection diagnostic output | `experiments/selection_diagnostic_shared_relation_50k/` |

### 17.5 已生成的 Stage0 50k object_relation 原始结果

| Metric | Value |
|---|---:|
| DynRollout | 97.4332 |
| Action MSE | 8.9531 |
| CollF1 | 0.6804 |
| RareADE | 14.0387 |
| IntRec@1m | 0.0999 |

### 17.6 已生成的 Stage1 50k `wm_naive` 原始结果

Raw file:

- `experiments/nuplan_stage1_shared_relation_50k/seed7/wm_naive/stage1_metrics.json`
- `experiments/nuplan_stage1_shared_relation_50k/seed42/wm_naive/stage1_metrics.json`
- `experiments/nuplan_stage1_shared_relation_50k/seed123/wm_naive/stage1_metrics.json`

| seed | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability | StabilityGlobal |
|---:|---:|---:|---:|---:|---:|
| 7 | 2.6120 | 0.0000 | 0.3328 | 0.0343 | 0.0139 |
| 42 | -6.7128 | 0.2447 | 0.2934 | 0.2499 | 0.0360 |
| 123 | -2.8071 | 1.0000 | 0.7253 | 0.3279 | 0.0619 |

Aggregate:

| condition | Return ↑ | ImagColl ↓ | CollMean ↓ | Stability |
|---|---:|---:|---:|---:|
| wm_naive 50k | -2.303 ± 4.681 | 0.415 ± 0.518 | 0.450 ± 0.239 | 0.204 ± 0.152 |

Interpretation: shared-relation 50k is runnable but unstable and weaker than `wm_decoupled_no_vis` 50k (`14.511 ± 2.925` return, `0.259 ± 0.045` collision rate).

### 17.7 已生成的 Selection Diagnostic 50k `wm_naive` 原始结果

Raw files:

- `experiments/selection_diagnostic_shared_relation_50k/summary.md`
- `experiments/selection_diagnostic_shared_relation_50k/summary.json`
- `experiments/selection_diagnostic_shared_relation_50k/seed7/wm_naive/summary.json`
- `experiments/selection_diagnostic_shared_relation_50k/seed42/wm_naive/summary.json`
- `experiments/selection_diagnostic_shared_relation_50k/seed123/wm_naive/summary.json`

| seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | IntRec@1m ↑ | RareADE ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 10000 | 0.2978 | 0.7022 | 0.3770 | 0.0819 | 0.9020 | 0.0072 | 18.1017 |
| 42 | 10000 | 0.8119 | 0.1881 | 0.2013 | 0.3894 | 0.9005 | 0.7647 | 5.0239 |
| 123 | 10000 | 0.2475 | 0.7525 | 0.2737 | 0.1601 | 0.9017 | 0.0010 | 21.9421 |

Aggregate:

| condition | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI |
|---|---:|---:|---:|---:|
| wm_naive 50k | 0.452 ± 0.312 | 0.548 ± 0.312 | 0.284 ± 0.088 | 0.210 ± 0.160 |

Interpretation: 50k shared-relation diagnostic gives stronger negative evidence than 20k: seed7 and seed123 both miss most critical dynamic agents, while seed42 is much better. This supports the paper's claim that shared top-k is unstable under type competition rather than consistently preserving decision-critical dynamic state.

### 17.8 预期用途

- 若 `wm_naive` 50k 明显弱于 `wm_decoupled_no_vis`，主文机制链更完整。
- 若 `wm_naive` 50k 高方差或训练不稳，也可作为 shared selector failure mode。
- 若意外接近 `wm_decoupled_no_vis`，需重新审视 20k selection diagnostic 与 50k regime 差异。

---

## 18. 当前论文证据链如何使用这些实验

### 18.1 主文建议使用

| 论文位置 | 使用实验 |
|---|---|
| Main Table 1 | Stage0 nuScenes 16-slot representation sufficiency |
| Main Figure 1 | type competition 概念图，需单独绘制 |
| Main Figure 2 | selection diagnostic bars / MissRate vs Interaction Recall |
| Main Table 2 | Stage1 nuPlan 50k latent planning |
| Main Figure 3 | relation feature-group ablation |
| Appendix Table | BC baseline、budget sensitivity、shared-relation 20k |

### 18.2 Appendix 建议使用

| 主题 | 实验 |
|---|---|
| Budget sensitivity | 10/6 vs 12/4 |
| Dataset/regime explanation | dataset token stats |
| Interaction subset | nuPlan interaction-conditioned subset |
| External integration | PlanTF nuPlan-devkit sanity |
| Closed-loop engineering | DOOR-RL closed-loop MVP smoke |
| Ongoing stronger control | `wm_naive` 50k closure |

### 18.3 不建议主推

- 不建议主推 `RelDensity -> selected relation ratio`，现有相关性不强。
- 不建议把 closed-loop smoke 写成主性能结果。
- 不建议声称 decoupled 在所有数据集和所有下游都赢。

---

## 19. 总结

当前已经形成一条较完整的 NeurIPS 证据链：

1. Stage0 主表证明 shared top-K 在固定容量下崩坏，decoupled typed-budget 修复。
2. Selection diagnostic 证明失败机制是 critical-agent miss 与 orphan relation，而不是 relation token 天然有害。
3. nuPlan Stage1 20k/50k 证明该表示机制在 planning-oriented latent rollout 中有效。
4. Offline planner-like sanity 证明优势延伸到 teacher-action alignment。
5. Relation ablation 证明关键 relation semantics 是 TTC/risk。
6. BC baseline 与 PlanTF sanity 提供外部锚点。
7. Dataset statistics 和 cross-dataset results 解释 ranking reversal。

剩余最有价值的增强项是正在运行的 `wm_naive` nuPlan 50k closure，它将 shared-relation failure 与 50k 主表对齐到同一 scale。

