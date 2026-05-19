# DOOR-RL 实验总报告

_Last updated: 2026-04-25_

本文档汇总当前 DOOR-RL 项目的全部关键实验：实验目标、数据来源、模型设置、实验安排、聚合结果、原始结果路径、已被推翻的历史结论、当前可靠结论和下一步建议。它的定位是“总索引 + 研究叙事 + 可复现实验记录”，详细单项说明仍保留在 `docs/stage0.md` 与 `docs/stage1_pilot.md`。

---

## 1. 项目目标与核心问题

DOOR-RL 研究的是自动驾驶决策学习中的目标-关系表示与潜在空间强化学习。当前核心假设是：

> 关系感知表示不是简单把 relation token 混进 shared top-k bottleneck，而是需要类型化预算：动态智能体和关系边应该通过不同选择头进入同一个固定容量的世界模型上下文。

当前已经完成和正在推进的研究问题：

1. **Stage 0：表示充分性。** 在固定 16-slot 世界模型上下文预算下，解耦目标-关系抽象是否能比 object-only 或 naive object+relation 更好地保留动态预测、稀有交互和碰撞信息？
2. **Stage 1：潜在想象强化学习。** Stage 0 的表示优势是否能转化为更好的 latent imagination policy learning？
3. **跨数据集问题。** 在 nuScenes 与 nuPlan 上，Stage 1 policy-learning 排名是否一致？如果不一致，说明 relation-aware abstraction 的收益依赖下游规划数据/任务设定，而不是单一全局优劣。

当前最重要结论：

- **Stage 0 上，decoupled typed-slot abstraction 明确成立。** 它在 nuScenes 700 scenes 上显著改善动态预测和交互召回，解决 naive object+relation 的 slot competition 问题。
- **Stage 1 在 nuScenes 上，object-only 当前最稳。** decoupled 表示虽然 Stage 0 更好，但在当前 K=5 latent imagination actor-critic 设置下高方差，未稳定转化成更好策略。
- **Stage 1 在 nuPlan 上，排名反转。** 5k pilot 和 20k scale-up 都显示 decoupled 系列优于 object-only，20k 上 `wm_decoupled_no_vis` 最强且最稳定。
- 因此论文叙事应从“decoupled 总是更好”升级为：**relation-aware abstraction 对 policy learning 的价值依赖 downstream planning regime 与 token/data quality。**

---

## 2. 代码与数据位置

### 2.1 代码根目录

项目根目录：

```text
/mnt/volumes/cpfs/prediction/lipeinan/code
```

核心文件：

| 文件 | 作用 |
|---|---|
| `configs/debug_mvp.json` | 基础模型/训练配置 |
| `run_stage0_table3.py` | Stage 0 表示充分性实验入口 |
| `run_stage1_table4.py` | Stage 1 latent imagination RL 实验入口 |
| `scripts/offline_planner_sanity.py` | nuPlan 50k offline planner-like sanity check |
| `scripts/run_nuplan_closed_loop_mvp.py` | nuPlan devkit closed-loop MVP runner |
| `scripts/summarize_nuplan_closed_loop.py` | nuPlan closed-loop parquet 汇总脚本 |
| `src/doorrl/models/doorrl_variant.py` | 模型变体、decoupled abstraction、fusion ablation |
| `src/doorrl/models/policy.py` | Actor-Critic head，支持 actor/critic 不同 latent 输入 |
| `src/doorrl/closed_loop/nuplan_oracle_planner.py` | DOOR-RL checkpoint -> nuPlan `AbstractPlanner` wrapper |
| `src/doorrl/data/real_dataset.py` | nuScenes 数据集与 token cache |
| `src/doorrl/data/nuplan_dataset.py` | nuPlan preprocessed NPZ 数据集，已加入并行 tokenisation |
| `src/doorrl/evaluation/table3_metrics.py` | Stage 0 指标 |
| `src/doorrl/evaluation/stage1_metrics.py` | Stage 1 指标 |
| `docs/stage0.md` | Stage 0 详细报告 |
| `docs/stage1_pilot.md` | Stage 1 详细实验记录 |

### 2.2 数据来源

| 数据集 | 路径 | 用途 |
|---|---|---|
| nuScenes | `/mnt/datasets/e2e-nuscenes/20260302` | Stage 0 主实验、Stage 1 nuScenes |
| nuPlan preprocessed NPZ | `/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split` | Stage 1 cross-dataset 与 scale-up |
| nuScenes token cache | `experiments/_token_cache` | 避免重复 nuScenes devkit tokenisation |
| nuPlan 20k balanced index | `experiments/nuplan_20k_balanced_paths_seed7.json` | 避免大规模 filesystem walk，固定 20k 子集 |
| nuPlan 50k balanced index | `experiments/nuplan_50k_balanced_paths_seed7.json` | 50k scale-up 与 planner-like sanity check |

### 2.3 环境与硬件

当前实验使用环境：

```text
/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python
```

已知硬件条件：

| 项 | 说明 |
|---|---|
| GPU | NVIDIA H20，约 96 GB 显存 |
| CPU | 192 logical cores |
| 内存 | 约 2 TB |
| 并行策略 | nuPlan tokenisation 使用 process workers；Stage 1 多 seed 并发训练 |

20k nuPlan 实验中实际采用：

- Stage 0：nuPlan tokenisation 使用 32 workers。
- Stage 1：3 seeds 并发，每个 seed 32 workers tokenisation。
- GPU 训练阶段观察到 GPU util 可达 99%。
- CPU 总使用率没有打满，瓶颈主要来自大量小 NPZ 文件读取、Python 转换和文件系统元数据开销。

---

## 3. 模型与实验条件

### 3.1 基础模型配置

来自 `configs/debug_mvp.json` 与运行时覆盖：

| 字段 | 值 |
|---|---:|
| `raw_dim` | 40 |
| `model_dim` | 128 |
| `hidden_dim` | 256 |
| `action_dim` | 2 |
| `max_tokens` | 97 |
| `num_token_types` | 8 |
| `top_k` | 16 |
| `top_k_dyn` | 12 |
| `top_k_rel` | 4 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `dropout` | 0.1 |

Stage 0 正式实验覆盖运行参数：

| 项 | 值 |
|---|---|
| Optimizer | Adam |
| lr | 1e-3 |
| weight decay | 1e-5 |
| batch size | 32 |
| epochs | 15 |
| loss weights | obs=1.0, reward=0.5, continue=0.25, collision=0.25, bc=0.1 |
| seeds | 7, 42, 2026 |

Stage 1 正式验证参数：

| 项 | 值 |
|---|---|
| horizon | K=5 |
| epochs | 10 |
| batch size | 128 |
| lr | 4e-3 (`lr_scale=4.0`) |
| actor loss | `-E[log pi * stop_grad(adv)] - beta * entropy` |
| entropy beta | 0.003 |
| critic loss | Huber, delta=10 |
| sanity loss | Stage 0 losses on real t=0 batch, weight=1.0 |
| GAE | gamma=0.97, lambda=0.95 |
| reward | progress=1, collision=5, action=0.01, clipped to [-5, 5] |
| action head | mean = `3*tanh(raw/3)`, log_std in [-2, 0.5] |
| sampled action clip | [-5, 5] |
| detach world model | True |
| seeds | 7, 42, 123 |

### 3.2 模型变体

| 条件/变体 | 表示 | 用途 |
|---|---|---|
| `holistic` | 97 tokens full context | 上界参考，不参与 16-slot 公平比较 |
| `holistic_16slot` | 16 learned queries | 16-slot 压缩参考 |
| `object_only` / `wm_object` | 只从动态目标选 top-k | Stage 0/1 baseline |
| `object_relation` | 动态目标和 relation token 共享 top-k | 验证 naive relation mixing 的失败 |
| `object_relation_visibility` | shared top-k + visibility weighting | shared relation + visibility 对照 |
| `object_relation_decoupled` / `wm_decoupled_no_vis` | K_dyn=12, K_rel=4，两个独立 top-k heads | decoupled 主变体，无 visibility |
| `object_relation_decoupled_visibility` / `wm_decoupled` | decoupled + dynamic path visibility weighting | nuScenes Stage 0 最强 DynRoll，Stage 1 nuScenes 默认 decoupled |
| `wm_decoupled_14_2` | K_dyn=14, K_rel=2 | relation budget ablation |
| `wm_decoupled_rel_to_critic_only` | actor=dyn latent, critic=dyn+rel latent | fusion ablation |

### 3.3 decoupled abstraction 设计

核心机制在 `src/doorrl/models/doorrl_variant.py`：

```text
1. 97 raw tokens -> encoder -> latent tokens
2. dyn_mask = EGO / VEHICLE / PEDESTRIAN / CYCLIST
3. rel_mask = RELATION
4. abstraction_dyn 在 dyn_mask 上选 K_dyn=12
5. abstraction_rel 在 rel_mask 上选 K_rel=4
6. concat -> 16 slots -> world model
7. global_latent = 0.5 * (dyn_global + rel_global)
```

设计约束：

- 总上下文预算仍为 16 slots，与 object-only、shared object+relation 完全公平。
- relation token 不再和 dynamic agent token 抢同一个 top-k budget。
- relation path 不强制选 ego，避免浪费 relation slot。
- typed budget 是结构性归纳偏置，不是简单后处理。

---

## 4. 实验安排总览

### 4.1 已完成实验矩阵

| 阶段 | 数据集 | 规模 | seeds | 条件 | 状态 |
|---|---|---:|---|---|---|
| Stage 0 | nuScenes | 700 scenes / 28,096 samples | 7, 42, 2026 | 7 variants | 完成，主结果 |
| Stage 1 X | nuScenes | 700 scenes | 7, 42, 123 | `wm_object`, `wm_decoupled`, `wm_decoupled_no_vis` | 完成，主验证 |
| Stage 1 14+2 | nuScenes | 700 scenes | 7, 42, 123 | `wm_decoupled_14_2` | 完成，budget ablation |
| Stage 1 fusion | nuScenes | 700 scenes | 7, 42, 123 | `wm_decoupled_rel_to_critic_only` | 完成，fusion ablation |
| Stage 0 | nuPlan | 5k NPZ | seed7 warm-start | 3 variants | 完成 |
| Stage 1 | nuPlan | 5k NPZ | 7, 42, 123 | 4 conditions | 完成，pilot |
| Stage 0 | nuPlan | 20k balanced NPZ | seed7 warm-start | 3 variants | 完成 |
| Stage 1 | nuPlan | 20k balanced NPZ | 7, 42, 123 | 3 conditions | 完成，scale-up |
| Stage 0 | nuPlan | 50k balanced NPZ | seed7 warm-start | 2 variants | 完成 |
| Stage 1 | nuPlan | 50k balanced NPZ | 7, 42, 123 | `wm_object`, `wm_decoupled_no_vis` | 完成，主 scale-up |
| Offline planner-like sanity | nuPlan | 50k val split | 7, 42, 123 | `wm_object`, `wm_decoupled_no_vis` | 完成，下游 sanity check |
| Dataset statistics | nuScenes vs nuPlan | 28,096 / 50,000 samples | seed7 | token/action/future stats | 完成，ranking reversal 分析 |
| Closed-loop MVP | nuPlan devkit | 1 scenario | seed7 checkpoints | `wm_object`, `wm_decoupled_no_vis` | 完成，true closed-loop smoke |
| P1/P2 follow-up | nuPlan / nuScenes | 50k / 700 scenes | 7, 42, 123 where applicable | `wm_decoupled(+vis)`, cross-dataset eval, Fig 4b, horizon K=3/5/7 | K sensitivity 完成；nuPlan-dependent jobs run via mount watchers |

### 4.2 结果可信度分层

| 层级 | 实验 | 是否可作为当前结论 |
|---|---|---|
| 主结果 | Stage 0 nuScenes 3 seeds | 是 |
| 主验证 | Stage 1 nuScenes X 3 seeds | 是 |
| 主扩展 | Stage 1 nuPlan 20k/50k 3 seeds | 是 |
| 下游支持 | nuPlan 50k offline planner-like sanity check | 是，但不能替代 closed-loop |
| 解释性分析 | nuScenes vs nuPlan token/action statistics | 是，用于 discussion |
| 解释性图 | P1 case-study figures | 是；`docs/figures.md` 为索引，Fig 4b 等待 nuPlan mount |
| 追加验证 | P1/P2 follow-up (`wm_decoupled(+vis)` nuPlan 50k, cross-dataset eval, horizon sensitivity) | K sensitivity 已完成；nuPlan-dependent jobs 等 mount 完成前不写入主结论 |
| 支持性结果 | Stage 1 nuPlan 5k pilot | 是，但只作为 pilot |
| 支持性 ablation | 14+2、rel-to-critic-only | 是 |
| 历史/调试 | `stage1_pilot`, `stage1_pilot_ab`, `stage1_pilot_v3`, smoke/sanity | 不作为主结论，只记录排错历史 |

---

## 5. Stage 0：nuScenes 表示充分性实验

### 5.1 实验设置

| 项 | 值 |
|---|---|
| 数据集 | nuScenes v1.0 trainval + CAN bus |
| scenes | 700 |
| samples | 28,096 |
| split | scene-level 80/20，560 train scenes / 140 val scenes |
| token schema | 97 tokens × 40 raw dims |
| context budget | 16 slots |
| seeds | 7, 42, 2026 |
| output aggregate | `experiments/table3_fair_fix2_aggregate.json` |
| detailed doc | `docs/stage0.md` |

### 5.2 Stage 0 指标

| 指标 | 含义 |
|---|---|
| DynRoll ↓ | dynamic agent next-state rollout MSE，nearest dynamic-slot matching |
| Action MSE ↓ | policy action mean vs teacher action MSE |
| Coll F1 ↑ | predicted collision vs relation-derived collision label F1 |
| Rare ADE ↓ | pedestrian/cyclist nearest-match ADE |
| IntRec@1m ↑ | 20m 内 rare agent 在 1m 内命中的比例 |

### 5.3 聚合结果

Source: `experiments/table3_fair_fix2_aggregate.json`

| Variant | Ctx | DynRoll ↓ | Action MSE ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Holistic-16Slot | 16 | 2.1059 ± 0.1600 | 0.2875 ± 0.0145 | 0.9782 ± 0.0105 | 1.4215 ± 0.0119 | 0.6433 ± 0.0153 |
| Object-only-16 | 16 | 3.7449 ± 1.0099 | 0.2854 ± 0.0103 | 0.9463 ± 0.0041 | 1.0964 ± 0.1159 | 0.9009 ± 0.0335 |
| Object+Relation-16 naive | 16 | 40.2822 ± 29.5376 | 0.2808 ± 0.0130 | 0.9803 ± 0.0125 | 7.5060 ± 5.4799 | 0.4295 ± 0.4074 |
| Obj+Rel+Vis-16 | 16 | 15.8023 ± 9.9254 | 0.2840 ± 0.0173 | 0.9330 ± 0.0641 | 2.9624 ± 1.6389 | 0.7283 ± 0.1545 |
| **Obj+Rel-Decoupled** | 16 | **2.1148 ± 0.1889** | 0.2805 ± 0.0125 | 0.9285 ± 0.0389 | **0.4913 ± 0.1768** | **0.9842 ± 0.0135** |
| **Decoupled+Visibility** | 16 | **1.8761 ± 0.2271** | 0.2843 ± 0.0234 | 0.9257 ± 0.0290 | **0.5197 ± 0.0495** | **0.9787 ± 0.0078** |
| Holistic-full reference | 97 | 0.1070 ± 0.1165 | 0.2858 ± 0.0112 | 0.9875 ± 0.0057 | 0.2562 ± 0.0234 | 1.0000 ± 0.0000 |

### 5.4 Stage 0 原始 per-seed 数据

Raw files:

- `experiments/table3_fair_fix2_seed7/table3_complete.json`
- `experiments/table3_fair_fix2_seed42/table3_complete.json`
- `experiments/table3_fair_fix2_seed2026/table3_complete.json`

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

### 5.5 Stage 0 结论

1. **naive Object+Relation 失败不是偶然。** 共享 top-k 下 relation token 与 dynamic token 竞争固定 16 slots，导致 dynamic rollout 和 rare-agent matching 大幅恶化。
2. **decoupled typed budget 解决 slot competition。** `object_relation_decoupled` 在 DynRoll、RareADE、IntRec@1m 上稳定优于 object-only。
3. **visibility 对 Stage 0 有小幅帮助。** `object_relation_decoupled_visibility` 的 DynRoll 最低，但 RareADE/IntRec 与 no-vis 非常接近。
4. **Stage 0 的结论很稳。** decoupled 的 IntRec@1m 标准差从 naive 的 0.407 降到约 0.013，说明这是结构性改进。

---

## 6. Stage 1：nuScenes latent imagination RL

### 6.1 实验设置

| 项 | 值 |
|---|---|
| 数据集 | nuScenes 700 scenes |
| split | scene-level 80/20 |
| val samples | 5,622 |
| horizon | K=5 |
| epochs | 10 |
| batch size | 128 |
| lr | 4e-3 |
| seeds | 7, 42, 123 |
| warm-start | 同 variant Stage 0 checkpoint |
| output | `experiments/stage1_pilot_x/seed*/<cond>/stage1_metrics.json` |
| aggregate | `experiments/stage1_pilot_x/X_summary.json` |

### 6.2 Stage 1 指标

| 指标 | 含义 |
|---|---|
| Return ↑ | K-step imagined latent reward sum |
| CollRate ↓ | imagined rollout 中 collision probability 超阈值比例 |
| CollMean ↓ | mean collision probability |
| Stab(ego-cos) | ego selected latent 相邻 rollout step 的 cosine distance；主要作 divergence/liveness sentinel |
| Stab(L2) | legacy global latent relative L2；不建议跨 variant 解读 |

### 6.3 X 主验证结果：3 seeds × 3 conditions

Source: `experiments/stage1_pilot_x/X_summary.json`

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| **wm_object** | **31.79 ± 19.70** | **0.597 ± 0.048** | **0.610 ± 0.048** | 0.258 ± 0.058 | 0.170 ± 0.034 |
| wm_decoupled | 4.34 ± 13.66 | 0.695 ± 0.283 | 0.676 ± 0.243 | 0.636 ± 0.108 | 0.056 ± 0.001 |
| wm_decoupled_no_vis | 0.34 ± 15.97 | 0.820 ± 0.260 | 0.814 ± 0.167 | 0.223 ± 0.033 | 0.043 ± 0.015 |

Raw per-seed values:

| Condition | Seed 7 | Seed 42 | Seed 123 | Mean ± std |
|---|---:|---:|---:|---:|
| wm_object Return | 21.79 | 54.48 | 19.10 | 31.79 ± 19.70 |
| wm_object CollRate | 0.602 | 0.547 | 0.643 | 0.597 ± 0.048 |
| wm_decoupled Return | -1.45 | 19.94 | -5.48 | 4.34 ± 13.66 |
| wm_decoupled CollRate | 0.837 | 0.369 | 0.878 | 0.695 ± 0.283 |
| wm_decoupled_no_vis Return | -9.79 | 18.75 | -7.94 | 0.34 ± 15.97 |
| wm_decoupled_no_vis CollRate | 0.970 | 0.970 | 0.520 | 0.820 ± 0.260 |

### 6.4 14+2 typed-budget ablation

目的：测试 default `K_dyn=12, K_rel=4` 是否因为 relation budget 太大导致 Stage 1 不稳。

Source:

- `experiments/stage1_pilot_14_2/summary.md`
- `experiments/stage1_pilot_14_2/summary.json`

| Condition | top_k_dyn | top_k_rel | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) |
|---|---:|---:|---:|---:|---:|---:|
| wm_decoupled_14_2 | 14 | 2 | 2.47 ± 4.14 | 0.808 ± 0.196 | 0.774 ± 0.189 | 0.419 ± 0.264 |

Raw values:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 3.157 | 0.987 | 0.986 | 0.598 | 0.206 |
| 42 | -2.903 | 0.535 | 0.528 | 0.614 | 0.128 |
| 123 | 7.167 | 0.903 | 0.808 | 0.046 | 0.041 |

结论：减少 relation budget 到 2 不能修复 Stage 1，甚至 collision 更差。因此 nuScenes Stage 1 的问题不是“relation slots 太多”这么简单。

### 6.5 Rel-to-critic-only fusion ablation

目的：测试 relation branch 是否只应该进入 critic/risk estimation，而不直接驱动 actor mean。

实现：

| Condition | actor input | critic input |
|---|---|---|
| wm_decoupled | dyn+rel | dyn+rel |
| wm_decoupled_rel_to_critic_only | dyn only | dyn+rel |

Source:

- `experiments/stage1_pilot_rel_critic_only/summary.md`
- `experiments/stage1_pilot_rel_critic_only/summary.json`

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_decoupled_rel_to_critic_only | 5.72 ± 7.25 | 0.729 ± 0.169 | 0.736 ± 0.163 | 0.453 ± 0.228 | 0.170 ± 0.036 |

Raw values:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.274 | 0.683 | 0.685 | 0.139 | 0.179 |
| 42 | 15.922 | 0.549 | 0.567 | 0.676 | 0.122 |
| 123 | 1.526 | 0.956 | 0.957 | 0.544 | 0.208 |

结论：critic-only relation 有一点帮助，但不是 rescue。fusion path 确实相关，但仅把 relation 从 actor 中移除不足以解决 nuScenes Stage 1 的不稳定。

### 6.6 nuScenes Stage 1 结论

1. 当前 Stage 1 设置下，**object-only 是 nuScenes 上最强且最稳的 policy-learning baseline**。
2. decoupled 在 Stage 0 的表示优势没有自动转化为 nuScenes Stage 1 策略优势。
3. `wm_decoupled` 有单 seed 竞争力，但跨 seed 高方差。
4. `wm_decoupled_no_vis` 在 nuScenes Stage 1 上最差，说明 visibility 在 nuScenes Stage 1 中更像 stabilizer。
5. 14+2 和 rel-to-critic-only 都没有关闭 object-only gap，后续应关注 imagination-time relation selection drift 或更干净的 risk/control 分离。

### 6.7 Horizon sensitivity sanity (seed7 eval-only)

作为 P1/P2 follow-up，已用同一组 seed7 checkpoint 在完整 nuScenes val split
(5,622 samples) 上做 K=3/5/7 eval-only rollout。输出位于
`experiments/horizon_sensitivity_nuscenes/k*/`。

| Horizon K | Condition | Return ↑ | CollRate ↓ | Ego-cos step ↓ | Max action norm |
|---:|---|---:|---:|---:|---:|
| 3 | `wm_object` | 11.856 | 0.570 | 0.296 | 3.781 |
| 3 | `wm_decoupled` | 1.646 | 0.683 | 0.710 | 4.056 |
| 5 | `wm_object` | 21.814 | 0.603 | 0.190 | 3.846 |
| 5 | `wm_decoupled` | -1.462 | 0.838 | 0.725 | 4.205 |
| 7 | `wm_object` | 31.950 | 0.610 | 0.143 | 3.850 |
| 7 | `wm_decoupled` | -5.914 | 0.916 | 0.733 | 4.209 |

Reading: decoupled 的 collision gap 随 rollout horizon 明显放大
(K=3: +0.113, K=5: +0.234, K=7: +0.306)，同时 ego-slot cosine step
distance 始终维持在约 0.71-0.73，支持“nuScenes 上的 decoupled 问题是
imagination rollout 中逐步累积的稳定性问题，而不只是单步预测误差”。

### 6.8 Cross-dataset eval sanity

作为 P1/P2 follow-up，已用已有 Stage-1 checkpoints 做纯 eval 的跨数据集
rollout：nuScenes-trained checkpoint 在 nuPlan val 上评估，nuPlan-trained
checkpoint 在 nuScenes val 上评估。输出位于
`experiments/cross_dataset_eval/`。

| Train -> Eval | Condition | Return ↑ | CollRate ↓ | Ego-cos step ↓ |
|---|---|---:|---:|---:|
| nuScenes -> nuPlan | `wm_object` | 27.177 ± 11.442 | 0.593 ± 0.261 | 0.369 ± 0.203 |
| nuScenes -> nuPlan | `wm_decoupled(+vis)` | -1.434 ± 17.283 | 0.593 ± 0.320 | 0.853 ± 0.102 |
| nuPlan -> nuScenes | `wm_object` | 5.508 ± 11.651 | 0.624 ± 0.135 | 0.301 ± 0.154 |
| nuPlan -> nuScenes | `wm_decoupled_no_vis` | 16.072 ± 2.197 | 0.434 ± 0.045 | 0.181 ± 0.039 |

Reading: nuScenes-trained `wm_decoupled(+vis)` does not transfer cleanly to
nuPlan despite having relation structure; its ego-cos drift remains high.
Conversely, nuPlan-trained `wm_decoupled_no_vis` transfers to nuScenes better
than nuPlan-trained `wm_object`. This supports the interpretation that the
ranking reversal is dataset/training-regime conditional, not evidence that
relation-aware structure is intrinsically harmful.

---

## 7. Stage 1：nuPlan 5k cross-dataset pilot

### 7.1 实验设置

| 项 | 值 |
|---|---|
| 数据集 | nuPlan preprocessed NPZ |
| root | `/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split` |
| samples | 5,000 per seed |
| split | 4,000 train / 1,000 val |
| Stage 0 warm-start | `experiments/nuplan_stage0_5k_seed7` |
| Stage 1 seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled`, `wm_decoupled_no_vis`, `wm_decoupled_rel_to_critic_only` |
| summary | `experiments/nuplan_stage1_5k/summary.md` |

### 7.2 聚合结果

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | -6.01 ± 3.10 | 0.348 ± 0.254 | 0.384 ± 0.150 | 0.505 ± 0.441 | 0.010 ± 0.006 |
| wm_decoupled | 9.38 ± 9.59 | **0.215 ± 0.095** | **0.281 ± 0.097** | 0.887 ± 0.011 | 0.067 ± 0.029 |
| **wm_decoupled_no_vis** | **12.91 ± 2.69** | 0.247 ± 0.029 | 0.325 ± 0.060 | 0.097 ± 0.062 | 0.037 ± 0.034 |
| wm_decoupled_rel_to_critic_only | 5.46 ± 2.26 | 0.231 ± 0.052 | 0.337 ± 0.025 | 0.848 ± 0.058 | 0.054 ± 0.023 |

### 7.3 原始 per-seed 数据

`wm_object`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | -9.275 | 0.446 | 0.461 | 0.239 | 0.002 |
| 42 | -6.916 | 0.000 | 0.174 | 1.127 | 0.012 |
| 123 | -1.842 | 0.598 | 0.515 | 0.149 | 0.017 |

`wm_decoupled`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.961 | 0.250 | 0.352 | 0.902 | 0.083 |
| 42 | 4.394 | 0.309 | 0.347 | 0.876 | 0.025 |
| 123 | 22.795 | 0.085 | 0.144 | 0.884 | 0.092 |

`wm_decoupled_no_vis`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 16.640 | 0.261 | 0.372 | 0.036 | 0.019 |
| 42 | 10.426 | 0.273 | 0.363 | 0.073 | 0.009 |
| 123 | 11.656 | 0.207 | 0.241 | 0.183 | 0.085 |

`wm_decoupled_rel_to_critic_only`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 6.032 | 0.301 | 0.367 | 0.900 | 0.085 |
| 42 | 7.896 | 0.215 | 0.336 | 0.878 | 0.028 |
| 123 | 2.453 | 0.176 | 0.307 | 0.766 | 0.048 |

### 7.4 nuPlan 5k 结论

1. nuPlan 5k 上 ranking 与 nuScenes 反转：decoupled family 明显优于 object-only。
2. `wm_decoupled_no_vis` Return 最强，`wm_decoupled` CollRate 最低。
3. 这说明 decoupled 不是 Stage 1 全局失败，而是可能依赖 planning-oriented 数据设定。
4. 但 5k 仍然只是 pilot，因此继续扩大到 20k 是必要的。

---

## 8. Stage 0/1：nuPlan 20k / 50k scale-up

### 8.1 nuPlan 20k 数据准备

为避免大规模文件系统扫描，创建了 balanced 20k NPZ index：

```text
experiments/nuplan_20k_balanced_paths_seed7.json
```

nuPlan dataset loader 已加入并行 tokenisation：

- `NuPlanPreprocessedDataset(..., num_workers=N)`
- Stage 0/1 runner 支持 `--nuplan-workers`
- 本轮 20k 使用 32 workers

### 8.2 nuPlan 20k Stage 0 warm-start

| 项 | 值 |
|---|---|
| output root | `experiments/nuplan_stage0_20k_seed7` |
| samples | 20,000 |
| split | 16,000 train / 4,000 val |
| variants | `object_only`, `object_relation_decoupled_visibility`, `object_relation_decoupled` |
| checkpoints | `experiments/nuplan_stage0_20k_seed7/<variant>/model.pt` |

Raw files:

- `experiments/nuplan_stage0_20k_seed7/object_only/table3_results.json`
- `experiments/nuplan_stage0_20k_seed7/object_relation_decoupled_visibility/table3_results.json`
- `experiments/nuplan_stage0_20k_seed7/object_relation_decoupled/table3_results.json`

Results:

| Variant | DynRoll ↓ | Action MSE ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|
| object_only | 168.733 | 5.948 | 0.534 | 17.981 | 0.294 |
| object_relation_decoupled_visibility | 3.210 | 4.242 | 0.724 | 0.848 | 0.987 |
| object_relation_decoupled | 3.210 | 4.242 | 0.724 | 0.848 | 0.987 |

结论：nuPlan 20k Stage 0 仍然极强地支持 decoupled 表示，object-only 在动态 rollout 与 rare ADE 上明显落后。

### 8.3 nuPlan 20k Stage 1 设置

| 项 | 值 |
|---|---|
| output root | `experiments/nuplan_stage1_20k` |
| samples | 20,000 balanced NPZ |
| split | 16,000 train / 4,000 val |
| seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled`, `wm_decoupled_no_vis` |
| epochs | 10 |
| batch size | 128 |
| horizon | 5 |
| Stage 0 root | `experiments/nuplan_stage0_20k_seed7` |
| workers | 32 tokenisation workers per seed |
| summary | `experiments/nuplan_stage1_20k/summary.md` |

### 8.4 nuPlan 20k Stage 1 聚合结果

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | 4.74 ± 13.95 | 0.373 ± 0.083 | 0.395 ± 0.057 | 0.426 ± 0.099 | 0.047 ± 0.009 |
| wm_decoupled | 13.48 ± 4.09 | 0.488 ± 0.217 | 0.509 ± 0.193 | 0.546 ± 0.046 | 0.095 ± 0.049 |
| **wm_decoupled_no_vis** | **17.50 ± 1.37** | **0.226 ± 0.105** | **0.251 ± 0.101** | 0.136 ± 0.022 | 0.043 ± 0.020 |

### 8.5 nuPlan 20k 原始 per-seed 数据

`wm_object`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 20.581 | 0.277 | 0.329 | 0.540 | 0.037 |
| 42 | -0.694 | 0.418 | 0.435 | 0.373 | 0.054 |
| 123 | -5.674 | 0.424 | 0.420 | 0.364 | 0.050 |

`wm_decoupled`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 15.707 | 0.294 | 0.355 | 0.561 | 0.149 |
| 42 | 15.963 | 0.449 | 0.447 | 0.582 | 0.054 |
| 123 | 8.760 | 0.722 | 0.725 | 0.494 | 0.081 |

`wm_decoupled_no_vis`:

| Seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---:|---:|---:|---:|---:|---:|
| 7 | 16.222 | 0.188 | 0.223 | 0.134 | 0.033 |
| 42 | 17.319 | 0.346 | 0.362 | 0.115 | 0.031 |
| 123 | 18.952 | 0.146 | 0.166 | 0.159 | 0.066 |

### 8.6 nuPlan 20k 结论

1. 5k pilot 的 decoupled 信号在 20k 上仍然存在，不是小样本偶然。
2. `wm_decoupled_no_vis` 是 20k 最强条件：Return 最高、碰撞最低、Return 方差最低，三 seed 全部稳定为正。
3. `wm_decoupled(+vis)` Return 高于 object-only，但 collision 更差，并且 seed 123 不稳定。
4. visibility 的角色呈现数据集依赖：nuScenes Stage 1 中 visibility 是 stabilizer；nuPlan 20k 中 no-vis 反而最强。

### 8.7 nuPlan 50k 主结果放大实验

原始 50k 主表只放大两个关键条件：

- `wm_object`
- `wm_decoupled_no_vis`

2026-04-27 P1/P2 follow-up 又补跑了缺失的 `wm_decoupled(+vis)` arm：
先补 `object_relation_decoupled_visibility` 的 50k Stage-0 warm-start，再用同一
root 并发跑 Stage-1 seeds 7/42/123。

设置：

| 项 | 值 |
|---|---|
| index | `experiments/nuplan_50k_balanced_paths_seed7.json` |
| Stage 0 root | `experiments/nuplan_stage0_50k_seed7` |
| Stage 1 root | `experiments/nuplan_stage1_50k` |
| samples | 50,000 balanced NPZ |
| split | 40,000 train / 10,000 val |
| seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled_no_vis`; follow-up adds `wm_decoupled(+vis)` |
| epochs / batch / horizon | 10 / 128 / 5 |
| loading | lazy NPZ loading, 32 DataLoader workers per seed |

50k Stage 0 warm-start:

| Variant | DynRoll ↓ | Action MSE ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|---:|
| object_only | 116.627 | 8.953 | 0.456 | 17.318 | 0.006 |
| object_relation_decoupled_visibility | 2.741 | 6.730 | 0.899 | 1.023 | 0.951 |
| object_relation_decoupled | 2.438 | 6.730 | 0.795 | 0.812 | 0.914 |

50k Stage 1 result:

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Stab(ego-cos) | Stab(L2) |
|---|---:|---:|---:|---:|---:|
| wm_object | 1.72 ± 17.89 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 | 0.025 |
| wm_decoupled(+vis) | -0.33 ± 4.94 | **0.007 ± 0.012** | 0.277 ± 0.111 | 0.255 | 0.005 |
| **wm_decoupled_no_vis** | **14.51 ± 2.93** | **0.259 ± 0.045** | **0.277 ± 0.033** | 0.222 | 0.043 |

Per-seed raw values:

| Condition | Seed 7 | Seed 42 | Seed 123 | Mean ± std |
|---|---:|---:|---:|---:|
| wm_object Return | 1.834 | 19.554 | -16.218 | 1.72 ± 17.89 |
| wm_object CollRate | 0.655 | 0.728 | 0.447 | 0.610 ± 0.146 |
| wm_decoupled(+vis) Return | -5.308 | -0.245 | 4.562 | -0.33 ± 4.94 |
| wm_decoupled(+vis) CollRate | 0.021 | 0.000 | 0.000 | 0.007 ± 0.012 |
| wm_decoupled_no_vis Return | 15.833 | 16.542 | 11.158 | 14.51 ± 2.93 |
| wm_decoupled_no_vis CollRate | 0.307 | 0.216 | 0.256 | 0.259 ± 0.045 |

Conclusion:

1. The 20k conclusion survives the 50k scale-up.
2. `wm_decoupled_no_vis` remains positive on all three seeds and keeps collision much lower than `wm_object`.
3. `wm_object` remains high-variance: seed 42 is strong, seed 123 collapses, and mean collision is high.
4. The added `wm_decoupled(+vis)` arm confirms that visibility is not the final
   nuPlan winner: it gives extremely low binary collision rate, but return is
   near zero / negative on average and training logs show a late sanity-loss
   blow-up. This makes `wm_decoupled_no_vis` the cleaner main policy-learning
   condition for nuPlan 50k.

### 8.8 nuPlan 50k offline planner-like sanity check

定位：

这是更下游的 offline planner-like sanity check，**不是**正式 external closed-loop evaluation 的替代品。它不接 nuPlan devkit、NAVSIM 或 CARLA，不包含 reactive agents 和 simulator；它只复用现有 nuPlan NPZ val split、Stage 1 checkpoint 和 imagination rollout，检查策略输出是否在离线 teacher action 与短 horizon imagined safety 上更合理。

设置：

| 项 | 值 |
|---|---|
| script | `scripts/offline_planner_sanity.py` |
| output | `experiments/nuplan_planner_sanity_50k` |
| data | nuPlan 50k val split |
| checkpoints | `experiments/nuplan_stage1_50k/seed*/{wm_object,wm_decoupled_no_vis}/model.pt` |
| seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled_no_vis` |
| val samples | 10,000 per seed |
| horizon | 5 |
| loading | lazy NPZ loading, 32 DataLoader workers |

主指标：

| Condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Policy action L2 | Return ↑ | CollRate ↓ | CollMean ↓ | Stability | Progress proxy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wm_object` | 8.863 ± 0.370 | 3.553 ± 0.118 | 3.010 | 1.722 ± 17.888 | 0.610 ± 0.146 | 0.614 ± 0.113 | 0.367 | 2.995 |
| **`wm_decoupled_no_vis`** | **6.628 ± 0.110** | **2.115 ± 0.083** | 1.512 | **14.512 ± 2.925** | **0.259 ± 0.045** | **0.277 ± 0.033** | 0.222 | 1.082 |

Per-seed raw values:

| Condition | Seed 7 | Seed 42 | Seed 123 | Mean ± std |
|---|---:|---:|---:|---:|
| wm_object Action MSE | 9.269 | 8.543 | 8.776 | 8.863 ± 0.370 |
| wm_object Return | 1.835 | 19.553 | -16.223 | 1.722 ± 17.888 |
| wm_object CollRate | 0.655 | 0.728 | 0.447 | 0.610 ± 0.146 |
| wm_decoupled_no_vis Action MSE | 6.742 | 6.521 | 6.621 | 6.628 ± 0.110 |
| wm_decoupled_no_vis Return | 15.842 | 16.536 | 11.158 | 14.512 ± 2.925 |
| wm_decoupled_no_vis CollRate | 0.306 | 0.216 | 0.256 | 0.259 ± 0.045 |

Reading:

1. `wm_decoupled_no_vis` 不仅在 Stage 1 latent return/collision 上更强，在 teacher-derived action MSE 上也更接近离线 planner-like action。
2. `wm_object` 的 progress proxy 接近 action bound，但 action MSE 和 collision 都高，说明它更像“强推进但不够安全/不够贴 teacher”的策略。
3. 这增强了 nuPlan 50k 主结论：decoupled no-vis 的优势不是只体现在 latent reward 上，也体现在更接近 planner behavior 的 offline sanity probe 上。
4. 该结果仍不能被表述为 closed-loop success；正式闭环仍需要后续外部 benchmark 接入。

### 8.9 nuPlan interaction-conditioned subset analysis

定位：

这是对 nuPlan 50k offline planner-like sanity check 的进一步条件化分析。它复用同一批 Stage-1 checkpoint 和 val split，不接 closed-loop simulator；目标是回答“relation-aware abstraction 的收益是否集中在更需要交互推理的样本上”。

设置：

| 项 | 值 |
|---|---|
| script | `scripts/interaction_subset_analysis.py` |
| output | `experiments/nuplan_interaction_subset_50k` |
| data | nuPlan 50k val split |
| checkpoints | `experiments/nuplan_stage1_50k/seed*/{wm_object,wm_decoupled_no_vis}/model.pt` |
| seeds | 7, 42, 123 |
| conditions | `wm_object`, `wm_decoupled_no_vis` |
| metrics | teacher action MSE, latent return, imagined collision rate |

Subset definitions:

| Subset | Definition |
|---|---|
| `low_ttc_proxy` | minimum relation-token TTC <= 5s |
| `lane_conflict` | any relation-token lane-conflict flag |
| `dense_agents` | dynamic token count >= 12 |
| `rare_agent_dense` | at least one pedestrian/cyclist and dynamic token count >= 8 |
| `high_interaction_union` | union of the above interaction-heavy filters |

Results:

| Subset | Condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---|---:|---:|---:|---:|
| all_val | `wm_object` | 10000 | 8.863 ± 0.370 | 1.718 ± 17.895 | 0.610 ± 0.146 |
| all_val | **`wm_decoupled_no_vis`** | 10000 | **6.628 ± 0.110** | **14.512 ± 2.927** | **0.260 ± 0.045** |
| low_ttc_proxy | `wm_object` | 4745 | 12.796 ± 0.425 | 2.650 ± 18.243 | 0.791 ± 0.084 |
| low_ttc_proxy | **`wm_decoupled_no_vis`** | 4745 | **11.534 ± 0.198** | **15.946 ± 2.851** | **0.541 ± 0.090** |
| lane_conflict | `wm_object` | 6699 | 7.023 ± 0.393 | 0.943 ± 17.975 | 0.591 ± 0.176 |
| lane_conflict | **`wm_decoupled_no_vis`** | 6699 | **4.225 ± 0.125** | **13.330 ± 3.134** | **0.205 ± 0.043** |
| rare_agent_dense | `wm_object` | 7280 | 8.729 ± 0.369 | 1.562 ± 17.985 | 0.626 ± 0.149 |
| rare_agent_dense | **`wm_decoupled_no_vis`** | 7280 | **6.494 ± 0.137** | **14.496 ± 3.085** | **0.273 ± 0.045** |
| dense_agents | `wm_object` | 8690 | 8.692 ± 0.414 | 1.423 ± 17.975 | 0.637 ± 0.170 |
| dense_agents | **`wm_decoupled_no_vis`** | 8690 | **6.340 ± 0.170** | **14.311 ± 3.127** | **0.272 ± 0.049** |
| high_interaction_union | `wm_object` | 9594 | 8.836 ± 0.410 | 1.588 ± 17.940 | 0.625 ± 0.150 |
| high_interaction_union | **`wm_decoupled_no_vis`** | 9594 | **6.555 ± 0.154** | **14.418 ± 2.994** | **0.270 ± 0.047** |

Reading:

1. `wm_decoupled_no_vis` 在所有 interaction-conditioned subsets 上都优于 `wm_object`：action MSE 更低、latent return 更高、imagined collision rate 更低。
2. 最强的解释性结果来自 `lane_conflict` 子集：decoupled-no-vis 的 action MSE 从 7.023 降到 4.225，collision 从 0.591 降到 0.205，说明 relation-aware abstraction 的收益确实集中在 lane/relation conflict 更明显的样本中。
3. `low_ttc_proxy` 是最难子集：两个模型 action MSE 和 collision 都升高，但 decoupled-no-vis 仍保持更低 collision（0.541 vs 0.791）和更高 return。
4. 这比 official closed-loop 小表更适合支撑论文解释：official closed-loop 目前被 wrapper 主导，而 interaction-conditioned offline probe 更直接说明“什么时候 relation 有用”。

Figure 4b follow-up: `scripts/plot_paper_fig_imagination_nuplan.py` 已在
5,000-sample val subset 上完成 lane-conflict per-sample imagination 图，输出
位于 `experiments/figures/case_studies/`。该 subset 中 lane-conflict 样本数
为 3,348；`wm_decoupled_no_vis` 的 return 为 15.211 vs `wm_object` 的
0.926，imagined collision rate 为 0.246 vs 0.652，max action norm 为
1.404 vs 3.266。

### 8.10 nuScenes vs nuPlan token/statistics 分析

为了解释“Stage 0 最优不等于 Stage 1 最优、nuScenes 最优不等于 nuPlan 最优”，补充了无需重训的 dataset statistics：

| 项 | 值 |
|---|---|
| script | `scripts/dataset_token_stats.py` |
| output | `experiments/dataset_token_stats` |
| nuScenes | 700 scenes / 28,096 samples |
| nuPlan | 50,000 balanced NPZ samples |
| metrics | token counts, rare-agent density, visibility, relation features, teacher action scale, short-horizon displacement |

核心统计：

| Statistic | nuScenes 700 scenes | nuPlan 50k NPZ | Interpretation |
|---|---:|---:|---|
| Dynamic tokens / sample | 9.715 ± 3.847 / p90 13.000 | 12.164 ± 2.457 / p90 13.000 | nuPlan 更常接近 dynamic token budget 上限 |
| Rare tokens / sample | 2.439 ± 2.906 / p90 7.000 | 3.934 ± 3.490 / p90 9.000 | nuPlan rare-agent 压力更大 |
| Relation tokens / sample | 11.318 ± 2.118 / p90 12.000 | 11.164 ± 2.457 / p90 12.000 | 两者 relation token 都接近上限 |
| Dynamic visibility | 0.746 ± 0.265 / p90 1.000 | 1.000 ± 0.000 / p90 1.000 | nuPlan visibility 几乎无区分度，解释 no-vis 更强 |
| Relation TTC | 15.363 ± 7.210 / p90 20.000 | 16.727 ± 6.124 / p90 20.000 | relation 风险特征尺度不同 |
| Teacher action L2 | 0.539 ± 0.632 / p90 1.183 | 3.420 ± 4.178 / p90 10.275 | 两个数据集 action-label scale 明显不同 |
| Ego next displacement | 0.000 ± 0.000 / p90 0.000 | 0.342 ± 0.418 / p90 1.027 | nuPlan NPZ 明确提供短未来 motion target |

Reading:

1. nuPlan 的 dynamic/rare-agent 密度更高，更容易让 typed dyn/rel budget 的归纳偏置发挥作用。
2. nuPlan 的 dynamic visibility 基本恒为 1，因此 visibility weighting 在 nuPlan 上缺少有效区分信号，`wm_decoupled_no_vis` 更强是合理的。
3. teacher action scale 的差异很大，说明 nuScenes 和 nuPlan 的 Stage 1 policy landscape 不是同一个问题，不能期待 ranking 完全一致。
4. 这些统计不是性能指标，而是解释 ranking reversal 的辅助证据。

### 8.11 nuPlan devkit closed-loop MVP

按照“先做 oracle-token true closed-loop MVP”的路线，已经把 DOOR-RL 两个主 checkpoint 接入 nuPlan 官方 simulation framework：

| 项 | 值 |
|---|---|
| runner | `scripts/run_nuplan_closed_loop_mvp.py` |
| planner wrapper | `src/doorrl/closed_loop/nuplan_oracle_planner.py` |
| adapter | `src/doorrl/adapters/nuplan_adapter.py` |
| devkit | `cangku/nuplan-devkit` |
| env | `flow_planner` conda env + local nuPlan source |
| data | `/mnt/datasets/e2e-nuplan/20260302/val` |
| maps | `/mnt/datasets/e2e-nuplan/20260302/maps` |
| mode | official nuPlan closed-loop nonreactive simulation |
| planners | `doorrl_wm_object`, `doorrl_wm_decoupled_no_vis` |
| 1-scenario output | `experiments/nuplan_closed_loop_mvp_1scenario_metrics/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents` |
| 5-scenario output | `experiments/nuplan_closed_loop_mvp_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents` |
| action-rollout 1-scenario output | `experiments/nuplan_closed_loop_action_rollout_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents` |
| action-rollout 5-scenario output | `experiments/nuplan_closed_loop_action_rollout_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents` |
| action-rollout 50-scenario output | `experiments/nuplan_closed_loop_action_rollout_50scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents` |

1-scenario smoke result:

| Planner | Success | Score | No at-fault collision | Drivable | Progress ratio | Comfort | TTC | Speed limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | yes | 0.000 | 0.000 | 1.000 | 0.013 | 0.000 | 0.000 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | yes | 0.000 | 1.000 | 1.000 | 0.095 | 1.000 | 1.000 | 1.000 |

Reading:

1. 这已经是真正的 nuPlan devkit simulation loop：官方 scenario builder、map API、controller、metrics、nuboard/metric parquet 都参与运行。
2. 当前只是 1-scenario smoke，不能作为正式 closed-loop 结论；它的价值是证明两个 DOOR-RL checkpoint 能被同一外部 evaluator 公平调用。
3. 在这个 smoke 场景上，`wm_decoupled_no_vis` 在 safety/comfort/TTC 上优于 `wm_object`，方向与 offline planner-like sanity 一致。
4. 下一步应扩到 5 scenarios sanity，再扩到 50–100 scenarios 形成正式闭环表。

5-scenario sanity result:

| Planner | Success | Score | No at-fault collision | Drivable | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | 5/5 | 0.000 | 0.400 | 0.600 | 0.014 | 0.000 | 0.000 | 1.000 | 0.300 |
| `doorrl_wm_decoupled_no_vis` | 5/5 | 0.000 | 0.600 | 1.000 | 0.105 | 1.000 | 0.600 | 1.000 | 0.700 |

5-scenario reading:

1. 两个 planner 都能稳定完成 nuPlan official simulation：10/10 simulations succeeded。
2. `wm_decoupled_no_vis` 在安全、可行驶区域、舒适性、TTC 和 progress ratio 上均优于 `wm_object`。
3. 两者总 score 仍为 0，主要因为 `ego_is_making_progress=0`，说明当前 constant-action trajectory wrapper 还偏保守/短视；正式 50–100 scenarios 前应改进 action-to-trajectory rollout 或加入 route-following projection。
4. 这组结果可以作为“闭环链路跑通 + 初步方向一致”的 sanity，不应写成最终闭环主结果。

Action-to-trajectory rollout update:

当前 `DoorRLNuPlanPlanner` 已从 constant-action rollout 改为 action-to-trajectory rollout。新的 wrapper 使用当前 ego speed 作为基线，将模型输出 action 转成 target speed delta 和 yaw-rate，再在固定 8s horizon 内用 rear-axle kinematic integration 生成 `InterpolatedTrajectory`。同时修正了 nuPlan `EgoState` 中 velocity / acceleration 应写入 rear-axle local frame 的问题。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | 1 scenario | 1/1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.023 | 1.000 | 0.000 | 0.902 | 0.000 |
| `doorrl_wm_decoupled_no_vis` | 1 scenario | 1/1 | 0.000 | 0.000 | 0.000 | 1.000 | 0.291 | 1.000 | 0.000 | 0.969 | 0.000 |
| `doorrl_wm_object` | 5 scenarios | 5/5 | 0.000 | 0.000 | 0.000 | 0.200 | 0.032 | 1.000 | 0.000 | 0.983 | 0.200 |
| `doorrl_wm_decoupled_no_vis` | 5 scenarios | 5/5 | 0.000 | 0.000 | 0.000 | 0.400 | 0.299 | 1.000 | 0.000 | 0.983 | 0.000 |
| `doorrl_wm_object` | 50 scenarios | 47/50 | 0.000 | 0.234 | 0.000 | 0.000 | 0.039 | 1.000 | 0.000 | 0.904 | 0.298 |
| `doorrl_wm_decoupled_no_vis` | 50 scenarios | 47/50 | 0.000 | 0.021 | 0.000 | 0.894 | 0.387 | 0.915 | 0.000 | 0.905 | 0.947 |

Action-rollout reading:

1. Progress 已经不再系统性为 0，说明主要 wrapper bottleneck 被定位并初步修复。
2. `wm_decoupled_no_vis` 在 1/5/50-scenario sanity 中都给出更高 progress ratio，方向与 nuPlan 50k Stage 1 / offline planner-like sanity 一致。
3. 50-scenario 结果并不是单向胜利：`wm_decoupled_no_vis` 的 progress、direction 明显更好，但 no at-fault collision 明显差于 `wm_object`；两个 planner 的 drivable 和 TTC 仍为 0。
4. 因此当前结论应写成：action-to-trajectory wrapper 成功解决 zero-progress failure mode，但还没有形成可作为论文主闭环表的 planner quality。下一步若继续闭环，应优先加 route-following / drivable-area projection 或更合理的 lateral control，而不是直接把 100-scenario 当最终结果。

Safety projection update:

为了解决 action-rollout 版本的 collision / TTC / drivable 缺陷，`DoorRLNuPlanPlanner` 增加了一个推理期 safety projection。它保留模型输出作为 nominal action，但在生成轨迹前枚举一组减速、停车和小幅 yaw 调整候选，并用当前 tracked objects 的线性外推、4s 内 TTC proxy、drivable-area map query、nominal deviation 和 progress reward 综合打分后选择轨迹。这个改动不使用未来 GT，也不改变 checkpoint。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | safety 1 scenario | 1/1 | 0.000 | 1.000 | 0.000 | 0.000 | 0.032 | 1.000 | 1.000 | 0.878 | 0.500 |
| `doorrl_wm_decoupled_no_vis` | safety 1 scenario | 1/1 | 0.000 | 1.000 | 1.000 | 0.000 | 0.156 | 1.000 | 1.000 | 1.000 | 0.500 |
| `doorrl_wm_object` | safety 5 scenarios | 5/5 | 0.000 | 1.000 | 0.200 | 0.000 | 0.033 | 0.000 | 0.600 | 1.000 | 0.400 |
| `doorrl_wm_decoupled_no_vis` | safety 5 scenarios | 5/5 | 0.000 | 1.000 | 0.200 | 0.200 | 0.267 | 0.000 | 1.000 | 1.000 | 0.600 |

Safety projection reading:

1. Safety projection 明显修复了 collision/TTC：5-scenario 上两个 planner 的 no at-fault collision 都到 1.0，`wm_decoupled_no_vis` 的 TTC 到 1.0。
2. Drivable 不再完全为 0，但只有 0.2，说明简单点查询和候选减速还不足以解决 route / lane-level 可行驶区域问题。
3. 代价是 comfort 变差，尤其 5-scenario 上 comfort 为 0，说明当前安全层主要通过较强减速/停车规避风险，缺少平滑控制约束。
4. 因此下一步不是再盲目加 collision 权重，而是把 safety projection 做成 smooth safety projection：限制 jerk / decel，加入 route-following 或 lane-center projection，并让 drivable query 参与连续轨迹修正。

Smooth safety projection update:

当前进一步加入了 smooth safety projection：候选轨迹评分显式惩罚过大 deceleration、jerk proxy 和 yaw-rate change，并增加 route/lane-center proxy。route proxy 优先使用 `PlannerInitialization.route_roadblock_ids` 关联的 lane / lane-connector baseline；如果无法匹配 route，则退化到 ego 附近 lane baseline。实验中发现 lane-center 权重过强会牺牲 collision/TTC，因此当前最稳的小样本配置先关闭 lane-center 权重，仅保留 smoothness 软惩罚和 emergency low-speed candidates。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | smooth safety 1 scenario | 1/1 | 0.000 | 1.000 | 1.000 | 0.000 | 0.032 | 1.000 | 1.000 | 0.887 | 0.500 |
| `doorrl_wm_decoupled_no_vis` | smooth safety 1 scenario | 1/1 | 0.000 | 1.000 | 1.000 | 0.000 | 0.164 | 1.000 | 1.000 | 1.000 | 0.000 |
| `doorrl_wm_decoupled_no_vis` | smooth safety 5 scenarios partial | 3/5 usable | 0.000 | 1.000 | 1.000 | 0.000 | 0.164 | 1.000 | 1.000 | 1.000 | 0.000 |

Smooth safety reading:

1. Smooth safety projection fixes the previous safety-vs-comfort trade-off on the smoke scenario: collision, drivable, TTC, and comfort all reach 1.0 for `wm_decoupled_no_vis`.
2. The cost is that `ego_is_making_progress` remains 0, so this is still a safe-but-conservative planner wrapper rather than a final route-following planner.
3. The 5-scenario run is only a partial result: 6/10 simulations succeeded; two scenarios failed inside nuPlan devkit's `drivable_area_compliance` metric with a time-series length assertion. The available aggregate keeps the 1-scenario pattern for `wm_decoupled_no_vis`, but should not be used as a formal 5-scenario table.
4. 下一步应修复 making-progress 与 metric failure，再扩场景；不建议直接上 50/100 scenarios。

Progress / drivable metric failure fix:

`ego_is_making_progress` 的阈值是 `ego_progress_along_expert_route >= 0.2`。smooth safety smoke 中 `wm_decoupled_no_vis` 的 progress ratio 为 0.164，距离阈值很近但未通过。当前 wrapper 增加了 `min_progress_speed` 候选，并提高 progress reward；同时本地 nuPlan devkit 的 `drivable_area_compliance` 增加了 time-series 对齐保护，避免 lane-change helper 产出的 corner route 长度短于 ego history 时触发 assertion。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_decoupled_no_vis` | progress fix 1 scenario | 1/1 | 0.000 | 1.000 | 0.000 | 1.000 | 0.614 | 1.000 | 1.000 | 1.000 | 0.000 |
| `doorrl_wm_decoupled_no_vis` | progress+drivable balance 1 scenario | 1/1 | 0.000 | 1.000 | 0.000 | 1.000 | 0.620 | 1.000 | 1.000 | 1.000 | 0.000 |

Progress / drivable reading:

1. Progress failure is fixed on the smoke scenario: `wm_decoupled_no_vis` now passes `ego_is_making_progress=1.0` with progress ratio around 0.62.
2. The devkit metric failure is fixed at source by aligning `drivable_area_compliance` time series lengths instead of allowing a simulation-level assertion failure.
3. Actual drivable score remains 0.0 once progress is enforced. This is now a real planner-quality issue rather than a metric crash: the route-forward/progress floor can move ego outside the drivable area.
4. Next step should be true route-following projection, not just scalar lane-center cost: project candidate trajectory points onto the route baseline / lane corridor before rollout, then rerun 5-scenario validation.

Route/lane corridor projection update:

当前 wrapper 已加入 route/lane corridor projection。推理时先从 `PlannerInitialization.route_roadblock_ids` 和 `map_api` 提取 route lane / lane-connector baseline；若 route 匹配失败，则退化为 ego 附近 lane baseline。候选轨迹不再只做自由空间 kinematic rollout，而是在少量最相关 corridor 上按弧长积分生成 ego future trajectory；collision/TTC/smoothness 仍作为候选评分项参与选择。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | corridor 1 scenario | 1/1 | 0.264 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.859 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | corridor 1 scenario | 1/1 | 0.264 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.862 | 1.000 |
| `doorrl_wm_object` | corridor 5 scenarios | 5/5 | 0.537 | 0.700 | 1.000 | 1.000 | 0.822 | 1.000 | 0.400 | 0.894 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | corridor 5 scenarios | 5/5 | 0.537 | 0.700 | 1.000 | 1.000 | 0.821 | 1.000 | 0.400 | 0.895 | 1.000 |

Frozen official small closed-loop table:

这是当前唯一建议保留的 official nuPlan closed-loop 小规模主表：wrapper 冻结为 route/lane corridor projection，只比较 `wm_object` 与 `wm_decoupled_no_vis`，不启用后续 lead-controller / TTC-proxy 试验项，指标只使用官方 nuPlan metric aggregate。5-scenario frozen rerun 成功完成 10/10 simulations；20/10-scenario 扩展在当前 devkit 会话中启动时间过长，因此没有作为主表结果使用。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | frozen corridor official 5 scenarios | 5/5 | 0.537 | 0.700 | 1.000 | 1.000 | 0.822 | 1.000 | 0.400 | 0.894 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | frozen corridor official 5 scenarios | 5/5 | 0.537 | 0.700 | 1.000 | 1.000 | 0.821 | 1.000 | 0.400 | 0.895 | 1.000 |

Frozen-table reading:

1. 这张表可以作为 appendix / sanity-strengthening 的 official closed-loop 结果：链路、官方 metric、固定小规模子集都成立。
2. 它不应升格为论文主结果：两个模型在官方 5-scenario 子集上几乎完全并列，差异明显由 wrapper 主导，而不是由 `wm_object` vs `wm_decoupled_no_vis` 的策略差异主导。
3. 当前最稳闭环结论是“corridor projection 解决 drivable/progress/comfort，但 local safety 仍是瓶颈”，不是“decoupled 在官方 closed-loop 中显著胜出”。

Corridor projection reading:

1. Route/lane corridor projection fixes the main drivable/progress issue: 5-scenario 上 drivable、making-progress、comfort、direction 都达到 1.0，progress ratio 约 0.82。
2. The patched drivable metric no longer fails: 10/10 simulations completed successfully on the 5-scenario run.
3. Overall score rises to 0.537 on the 5-scenario subset, much higher than previous closed-loop wrappers.
4. The remaining bottleneck is local safety, not route feasibility: no at-fault collision is 0.7 and TTC is 0.4. Next step should combine corridor projection with stronger obstacle-aware speed/yield control before scaling to 50-100 scenarios.

Obstacle-aware yield control attempt:

在 corridor projection 基础上尝试加入 obstacle-aware speed/yield control：将 tracked objects 投影到当前 corridor 上，如果 agent 位于 ego 前方且横向距离接近 lane corridor，则自动生成跟车、减速、让行速度候选。这个版本是推理期局部控制，不改变 checkpoint。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | corridor+yield 1 scenario | 1/1 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.829 | 0.000 |
| `doorrl_wm_decoupled_no_vis` | corridor+yield 1 scenario | 1/1 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.825 | 0.000 |

Yield-control reading:

1. 简单 corridor-yield speed candidates 能保住 no at-fault collision / drivable / progress，但会破坏 comfort、TTC 和 direction，overall score 退回 0。
2. 因此这个版本不应扩到 5/50 scenarios，也不应作为默认闭环 wrapper。
3. 当前最稳的默认仍是 corridor projection；下一步局部安全不应只是追加低速候选，而应做 lead-vehicle following / stop-line style longitudinal controller，并显式约束 jerk、方向和 TTC。

Smooth lead-vehicle controller attempt:

在 corridor projection 基础上进一步尝试可开关的 lead-vehicle following / stop-line style longitudinal controller。该 controller 将前方障碍物投影到当前 lane corridor 上，用 time headway、minimum gap 和 per-step speed-drop cap 生成更平滑的跟车速度上限；默认 corridor projection 保持不变，只有显式打开 `--enable-lead-vehicle-controller` 时启用。为避免 devkit metric 因生成轨迹 time-series 长度不一致而中断，额外对 `speed_limit_compliance` 和 `driving_direction_compliance` 做了与先前 `drivable_area_compliance` 相同的诊断序列对齐补丁。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | lead controller 1 scenario | 1/1 | 0.000 | 0.000 | 1.000 | 1.000 | 0.471 | 1.000 | 0.000 | 1.000 | 0.500 |
| `doorrl_wm_decoupled_no_vis` | lead controller 1 scenario | 1/1 | 0.388 | 1.000 | 1.000 | 1.000 | 0.281 | 1.000 | 1.000 | 1.000 | 0.500 |
| `doorrl_wm_object` | aggressive lead 5 scenarios | 5/5 | 0.213 | 0.400 | 1.000 | 1.000 | 0.529 | 0.000 | 0.200 | 1.000 | 0.400 |
| `doorrl_wm_decoupled_no_vis` | aggressive lead 5 scenarios | 5/5 | 0.213 | 0.900 | 1.000 | 1.000 | 0.430 | 0.000 | 0.600 | 1.000 | 0.400 |
| `doorrl_wm_object` | capped lead 5 scenarios | 5/5 | 0.078 | 0.700 | 1.000 | 1.000 | 0.769 | 0.000 | 0.400 | 0.893 | 0.300 |
| `doorrl_wm_decoupled_no_vis` | capped lead 5 scenarios | 5/5 | 0.078 | 0.700 | 1.000 | 1.000 | 0.768 | 0.000 | 0.400 | 0.895 | 0.300 |
| `doorrl_wm_object` | gradual lead soft 5 scenarios | 5/5 | 0.000 | 0.000 | 1.000 | 1.000 | 0.336 | 1.000 | 0.000 | 1.000 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | gradual lead soft 5 scenarios | 5/5 | 0.000 | 0.000 | 1.000 | 1.000 | 0.343 | 1.000 | 0.000 | 1.000 | 1.000 |
| `doorrl_wm_object` | gradual lead tuned 5 scenarios | 5/5 | 0.145 | 0.300 | 1.000 | 1.000 | 0.333 | 1.000 | 0.000 | 1.000 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | gradual lead tuned 5 scenarios | 5/5 | 0.145 | 0.300 | 1.000 | 1.000 | 0.333 | 1.000 | 0.000 | 1.000 | 1.000 |

Lead-controller reading:

1. 1-scenario smoke 对 `wm_decoupled_no_vis` 是正向的：collision/TTC/drivable/comfort/making-progress 全为 1.0，但 progress ratio 低于默认 corridor projection。
2. 5-scenario 上 aggressive lead controller 能把 `wm_decoupled_no_vis` 的 no at-fault collision 从 0.70 提到 0.90、TTC 从 0.40 提到 0.60，但 comfort 降到 0、direction 降到 0.40，overall score 从默认 corridor 的 0.537 降到 0.213。
3. capped lead controller 保守限制单步降速后，安全指标回到默认 corridor 水平（collision 0.70、TTC 0.40），但 comfort 仍为 0、direction 降到 0.30，overall score 仅 0.078。
4. gradual lead controller 改成“风险触发 + speed cap”后能保住 comfort/drivable/progress/direction，但局部安全反而更差：soft 版 collision/TTC 为 0.00/0.00，tuned 版也只有 0.30/0.00，均明显低于默认 corridor 的 0.70/0.40。
5. 因此当前 lead-controller 设计不满足扩到 50-scenario 的条件。当前可引用的稳定 closed-loop wrapper 仍是默认 corridor projection；local safety 的下一步需要真正的 jerk-aware longitudinal profile 或 stop-line/lead-vehicle state machine，而不是只在候选速度层做 cap。

TTC proxy / path-risk control attempt:

在默认 corridor projection 基础上加入可开关的 TTC proxy：对尚未发生 overlap、但在短时间窗口内 clearance 低于阈值的候选轨迹增加 TTC-like cost，目标是在碰撞前就让安全评分偏向更大 clearance 的 path/speed 候选。随后增加 `corridor_candidate_limit`，允许安全层在更多 route/lane corridor 上选择低风险路径。该实验不启用 lead speed cap。

| Planner | Setting | Success | Score | No at-fault collision | Drivable | Making progress | Progress ratio | Comfort | TTC | Speed limit | Direction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doorrl_wm_object` | TTC proxy mild 5 scenarios | 5/5 | 0.536 | 0.700 | 1.000 | 1.000 | 0.820 | 1.000 | 0.400 | 0.888 | 1.000 |
| `doorrl_wm_decoupled_no_vis` | TTC proxy mild 5 scenarios | 5/5 | 0.535 | 0.700 | 1.000 | 1.000 | 0.818 | 1.000 | 0.400 | 0.887 | 1.000 |
| `doorrl_wm_object` | TTC proxy wide-path 5 scenarios | 5/5 | 0.000 | 0.400 | 1.000 | 0.000 | 0.115 | 0.000 | 0.400 | 1.000 | 0.200 |
| `doorrl_wm_decoupled_no_vis` | TTC proxy wide-path 5 scenarios | 5/5 | 0.000 | 0.200 | 1.000 | 0.000 | 0.112 | 0.000 | 0.200 | 1.000 | 0.300 |

TTC-proxy reading:

1. Mild TTC proxy 保住了默认 corridor projection 的 drivable、comfort、making-progress 和 direction，但几乎没有改变 collision/TTC；score 从 0.537 小幅降到约 0.535。
2. 增大 TTC proxy 并扩大 corridor 候选数后，planner 会选择过于保守或不连续的 corridor/path，导致 making-progress 和 comfort 崩溃，overall score 变成 0。
3. 这说明当前局部安全瓶颈不是简单提高 TTC/collision scoring 可以解决的；候选轨迹集合本身缺少可舒适执行的避障动作。默认 corridor projection 仍是当前最稳定闭环 wrapper，不应扩展 TTC proxy / path-risk 版本到 50-scenario。

---

## 9. 历史实验与被推翻结论

以下实验保留为排错和叙事演化记录，不应作为当前主结论引用：

| 实验目录 | 状态 | 说明 |
|---|---|---|
| `experiments/stage1_pilot/` | 历史 | 早期 seed7 pilot，数值受后续 bug/hparam 修正影响 |
| `experiments/stage1_pilot_ab/` | 历史 | A+B 数值修复后的单 seed |
| `experiments/stage1_pilot_v3/` | 历史 | v3 单 seed，曾显示 decoupled collision 更好，但 X 证明不稳 |
| `experiments/stage1_pilot_y/` | 历史/子集 | seed7 no-vis collapse，后来被 X 多 seed 覆盖 |
| `experiments/stage1_nanfix/`, `stage1_sanity/`, `stage1_fixed/` | 调试 | NaN、critic、action clamp、reward clip 排错 |
| `experiments/table3_smoke*`, `table3_fair_sanity*` | 调试 | Stage 0 smoke/sanity runs |

被推翻或修正的结论：

1. **“Stage 1 decoupled 在 nuScenes 上稳定优于 object-only”被推翻。** 这是早期单 seed 结论；X 多 seed 显示 object-only 更稳。
2. **“nuScenes no-vis 崩说明 decoupled 全局失败”被推翻。** nuPlan 5k/20k 显示 no-vis decoupled 在 planning-oriented NPZ 数据上很强。
3. **“relation budget 太大导致 Stage 1 失败”没有被支持。** 14+2 没有修复 nuScenes Stage 1。
4. **“把 relation 只给 critic 就能救 decoupled”没有被支持。** rel-to-critic-only 只有部分改善。

---

## 10. 原始结果与日志索引

### 10.1 Stage 0 nuScenes

| 内容 | 路径 |
|---|---|
| aggregate JSON | `experiments/table3_fair_fix2_aggregate.json` |
| seed7 raw | `experiments/table3_fair_fix2_seed7/table3_complete.json` |
| seed42 raw | `experiments/table3_fair_fix2_seed42/table3_complete.json` |
| seed2026 raw | `experiments/table3_fair_fix2_seed2026/table3_complete.json` |
| per-variant checkpoints | `experiments/table3_fair_fix2_seed*/<variant>/model.pt` |
| detailed report | `docs/stage0.md` |

### 10.2 Stage 1 nuScenes

| 内容 | 路径 |
|---|---|
| X aggregate | `experiments/stage1_pilot_x/X_summary.json` |
| X raw metrics | `experiments/stage1_pilot_x/seed*/<condition>/stage1_metrics.json` |
| X logs | `experiments/stage1_pilot_x/logs/` |
| 14+2 summary | `experiments/stage1_pilot_14_2/summary.md` |
| 14+2 raw metrics | `experiments/stage1_pilot_14_2/seed*/wm_decoupled_14_2/stage1_metrics.json` |
| rel-to-critic summary | `experiments/stage1_pilot_rel_critic_only/summary.md` |
| rel-to-critic raw metrics | `experiments/stage1_pilot_rel_critic_only/seed*/wm_decoupled_rel_to_critic_only/stage1_metrics.json` |
| detailed report | `docs/stage1_pilot.md` |

### 10.3 nuPlan experiments

| 内容 | 路径 |
|---|---|
| nuPlan 5k Stage0 | `experiments/nuplan_stage0_5k_seed7/` |
| nuPlan 5k Stage1 summary | `experiments/nuplan_stage1_5k/summary.md` |
| nuPlan 5k raw metrics | `experiments/nuplan_stage1_5k/seed*/<condition>/stage1_metrics.json` |
| nuPlan 20k index | `experiments/nuplan_20k_balanced_paths_seed7.json` |
| nuPlan 20k Stage0 | `experiments/nuplan_stage0_20k_seed7/` |
| nuPlan 20k Stage0 log | `experiments/nuplan_stage0_20k_seed7/logs/shared_loader_stage0.log` |
| nuPlan 20k Stage1 summary | `experiments/nuplan_stage1_20k/summary.md` |
| nuPlan 20k raw metrics | `experiments/nuplan_stage1_20k/seed*/<condition>/stage1_metrics.json` |
| nuPlan 20k logs | `experiments/nuplan_stage1_20k/logs/seed*.log` |
| nuPlan 50k index | `experiments/nuplan_50k_balanced_paths_seed7.json` |
| nuPlan 50k Stage0 | `experiments/nuplan_stage0_50k_seed7/` |
| nuPlan 50k Stage1 summary | `experiments/nuplan_stage1_50k/summary.md` |
| nuPlan 50k raw metrics | `experiments/nuplan_stage1_50k/seed*/<condition>/stage1_metrics.json` |
| nuPlan 50k logs | `experiments/nuplan_stage1_50k/logs/seed*.log` |
| nuPlan 50k planner sanity summary | `experiments/nuplan_planner_sanity_50k/summary.md` |
| nuPlan 50k planner sanity raw | `experiments/nuplan_planner_sanity_50k/seed*/<condition>/planner_sanity.json` |
| nuPlan closed-loop MVP 1-scenario summary | `experiments/nuplan_closed_loop_mvp_1scenario_metrics/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan closed-loop MVP metrics | `experiments/nuplan_closed_loop_mvp_1scenario_metrics/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric}/` |
| nuPlan closed-loop MVP logs | `experiments/nuplan_closed_loop_mvp_1scenario_metrics/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{simulation_log,nuboard_*.nuboard,runner_report.parquet}` |
| nuPlan closed-loop MVP 5-scenario summary | `experiments/nuplan_closed_loop_mvp_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan closed-loop MVP 5-scenario raw | `experiments/nuplan_closed_loop_mvp_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan action-rollout 1-scenario summary | `experiments/nuplan_closed_loop_action_rollout_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan action-rollout 5-scenario summary | `experiments/nuplan_closed_loop_action_rollout_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan action-rollout 50-scenario summary | `experiments/nuplan_closed_loop_action_rollout_50scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan action-rollout 50-scenario raw | `experiments/nuplan_closed_loop_action_rollout_50scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan interaction-conditioned subset summary | `experiments/nuplan_interaction_subset_50k/summary.md` |
| nuPlan interaction-conditioned subset raw | `experiments/nuplan_interaction_subset_50k/{summary.json,seed*/<condition>/subset_metrics.json}` |
| nuPlan safety projection 1-scenario summary | `experiments/nuplan_closed_loop_safety_projection_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan safety projection 5-scenario summary | `experiments/nuplan_closed_loop_safety_projection_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan smooth safety 1-scenario summary | `experiments/nuplan_closed_loop_smooth_safety_1scenario_v4/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan smooth safety 5-scenario partial summary | `experiments/nuplan_closed_loop_smooth_safety_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan progress fix 1-scenario summary | `experiments/nuplan_closed_loop_progress_fix_1scenario_v2/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan progress+drivable balance 1-scenario summary | `experiments/nuplan_closed_loop_progress_drivable_fix_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan corridor projection 1-scenario summary | `experiments/nuplan_closed_loop_corridor_projection_1scenario_fast/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan corridor projection 5-scenario summary | `experiments/nuplan_closed_loop_corridor_projection_5scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan frozen corridor official 5-scenario raw | `experiments/nuplan_closed_loop_official_corridor_5scenario_frozen/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan corridor+yield 1-scenario summary | `experiments/nuplan_closed_loop_corridor_yield_1scenario_v2/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/summary.md` |
| nuPlan lead controller 1-scenario raw | `experiments/nuplan_closed_loop_lead_controller_1scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan aggressive lead 5-scenario raw | `experiments/nuplan_closed_loop_lead_controller_5scenario_v2/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan capped lead 5-scenario raw | `experiments/nuplan_closed_loop_lead_controller_5scenario_v4/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan gradual lead soft 5-scenario raw | `experiments/nuplan_closed_loop_gradual_lead_5scenario_soft/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan gradual lead tuned 5-scenario raw | `experiments/nuplan_closed_loop_gradual_lead_5scenario_tuned/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan TTC proxy mild 5-scenario raw | `experiments/nuplan_closed_loop_ttc_proxy_5scenario_mild/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuPlan TTC proxy wide-path 5-scenario raw | `experiments/nuplan_closed_loop_ttc_proxy_5scenario_wide/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents/{metrics,aggregator_metric,simulation_log,runner_report.parquet}` |
| nuScenes vs nuPlan dataset stats | `experiments/dataset_token_stats/summary.md` |
| dataset stats raw JSON | `experiments/dataset_token_stats/summary.json` |

---

## 11. 当前可靠结论

### 11.1 表示学习层面

在固定 16-slot world-model context 下，decoupled typed-budget abstraction 是当前最可靠的表示设计。它解决了 naive relation mixing 的 token budget competition，并在 nuScenes Stage 0 上稳定提升动态预测、稀有交互和 interaction recall。

### 11.2 策略学习层面

Stage 0 表示优势不保证直接转化为 Stage 1 策略优势。nuScenes Stage 1 中，object-only 更稳；decoupled 高方差。这个 gap 更可能来自：

- relation branch 与 actor/critic fusion 不匹配；
- imagination rollout 中 relation selection drift；
- nuScenes token/action/reward 设定下，relation-aware latent 对 actor landscape 不友好。

### 11.3 跨数据集层面

nuPlan 结果显示 decoupled 不是 Stage 1 全局失败。20k 和 50k 规模下 `wm_decoupled_no_vis` 都明确优于 object-only，说明 relation-aware abstraction 的 policy-learning 价值依赖数据集和规划设定。

### 11.4 下游 sanity check 层面

nuPlan 50k offline planner-like sanity check 进一步支持 `wm_decoupled_no_vis`：它在 teacher-derived action MSE、imagined collision 和 latent return 上都优于 `wm_object`。Interaction-conditioned subset analysis 进一步显示，这个优势在 lane-conflict、low-TTC、rare-agent dense、dense-agent 等更需要交互推理的子集上仍然成立，其中 lane-conflict 子集最清楚（Action MSE 4.225 vs 7.023，collision 0.205 vs 0.591）。这不是闭环证明，但说明 Stage 1 主结果没有只停留在 latent reward 指标上，而是能通过更靠近 planner behavior 的离线 probe 解释“什么时候 relation 有用”。

### 11.5 外部闭环层面

nuPlan devkit closed-loop MVP 已经跑通：`wm_object` 与 `wm_decoupled_no_vis` 都能作为 oracle-token planner 接入官方 nuPlan simulation loop，并在同一 evaluator 下评测。constant-action 版本证明了链路可行但 progress 偏弱；action-to-trajectory rollout 已让 progress 不再系统性为 0。当前最稳定 wrapper 是 route/lane corridor projection：5-scenario 上 drivable、making-progress、comfort 均为 1.0，overall score 为 0.537。frozen official 5-scenario table 显示两种策略几乎并列，因此 closed-loop 结果目前更适合放在 appendix / sanity，而不应升格为主结果。后续 lead-controller 和 TTC-proxy 尝试未通过 5-scenario 净收益门槛，因此不扩到 50-scenario。

### 11.6 为什么 ranking 会变

dataset statistics 显示 nuPlan 与 nuScenes 的 token/action 分布并不相同：nuPlan dynamic tokens 更密、rare agent 更多、teacher action scale 更大，并且 visibility 几乎恒为 1。这支持当前解释：表示质量、policy 学习稳定性和下游 planner-like 行为是三个相关但不等价的维度。

### 11.7 跨阶段 ranking 总结

| Setting | Best variant | Evidence | Interpretation |
|---|---|---|---|
| nuScenes Stage 0 | Decoupled / Decoupled+Vis | DynRoll 1.88-2.11, IntRec 0.979-0.984 | typed-slot abstraction solves shared-budget relation/object competition |
| nuScenes Stage 1 | Object-only | Return 31.79 ± 19.70, CollRate 0.597 ± 0.048 | better representation does not automatically produce a better actor-critic state under current nuScenes imagination setup |
| nuPlan Stage 1 20k | Decoupled-no-vis | Return 17.50 ± 1.37, CollRate 0.226 ± 0.105 | planning-oriented NPZ setting makes relation-aware abstraction useful for policy learning |
| nuPlan Stage 1 50k | Decoupled-no-vis | Return 14.51 ± 2.93, CollRate 0.259 ± 0.045 | scale-up confirms the 20k ranking; object-only remains high-variance |
| nuPlan 50k offline planner-like sanity | Decoupled-no-vis | Action MSE 6.63 ± 0.11, CollRate 0.259 ± 0.045 | downstream offline probe supports the same main condition, without claiming closed-loop success |
| nuPlan interaction-conditioned subsets | Decoupled-no-vis | lane-conflict Action MSE 4.225 vs 7.023, collision 0.205 vs 0.591 | relation-aware benefit is strongest when relation/lane conflict is actionable |
| nuPlan closed-loop MVP | Decoupled-no-vis on constant-action 5-scenario sanity | no at-fault collision 0.60 vs 0.40, comfort 1.00 vs 0.00, TTC 0.60 vs 0.00 | official simulation loop is connected; progress remained weak under the first wrapper |
| nuPlan action-rollout closed-loop | Mixed on 50-scenario sanity | Decoupled-no-vis progress ratio 0.387 vs 0.039, making-progress 0.894 vs 0.000; object-only no at-fault collision 0.234 vs 0.021 | wrapper fixes zero-progress bottleneck, but safety/drivable/TTC are not yet good enough for a final closed-loop claim |
| nuPlan safety-projection closed-loop | Decoupled-no-vis on 5-scenario safety sanity | no at-fault collision 1.00, TTC 1.00, progress ratio 0.267 vs object-only 0.033 | safety projection fixes collision/TTC direction but harms comfort; needs smooth safety control before scaling |
| nuPlan smooth-safety closed-loop | Decoupled-no-vis on 1-scenario smoke; 5-scenario partial | 1-scenario no collision/drivable/comfort/TTC all 1.00, progress ratio 0.164 | smoothness fixes comfort on smoke, but making-progress remains 0 and 5-scenario has metric failures; not ready for scale-up |
| nuPlan corridor-projection closed-loop | Corridor wrapper on 5-scenario sanity | score 0.537, drivable 1.00, making-progress 1.00, progress ratio 0.821, comfort 1.00 | route/lane corridor projection fixes feasibility/progress; remaining bottleneck is local obstacle safety (collision 0.70, TTC 0.40) |
| nuPlan frozen official closed-loop | No clear model winner on 5-scenario official table | `wm_object` and `wm_decoupled_no_vis` both score 0.537 with identical collision/TTC/drivable/comfort/progress | useful appendix sanity table, but wrapper dominates and it should not be promoted to a main result |
| nuPlan corridor+yield attempt | Negative on 1-scenario smoke | no collision/drivable/progress 1.00, but comfort/TTC/direction 0.00 | simple yield-speed candidates are too disruptive; use corridor projection as default and design a smoother lead-vehicle controller next |
| nuPlan lead-controller attempt | Negative on 5-scenario validation | aggressive lead improves decoupled collision/TTC to 0.90/0.60 but drops comfort to 0 and score to 0.213; capped lead reverts safety to 0.70/0.40 and score to 0.078 | local speed caps are not enough; do not scale this controller to 50-scenario |
| nuPlan gradual lead ablation | Negative on 5-scenario validation | comfort/drivable/progress/direction stay high, but collision/TTC fall to 0.00/0.00 or 0.30/0.00 | gradual caps preserve comfort but do not solve local safety; keep default corridor projection |
| nuPlan TTC-proxy path-risk ablation | Negative / neutral on 5-scenario validation | mild proxy preserves corridor score but does not improve collision/TTC; wide-path proxy collapses progress/comfort | stronger TTC scoring alone cannot fix local safety without richer feasible avoidance trajectories |

论文正文可使用的 compact table：

| Dataset / Stage | Best Variant | Key Takeaway |
|---|---|---|
| nuScenes Stage 0 | Decoupled / Decoupled+Vis | better decision-critical representation |
| nuScenes Stage 1 | Object-only | better policy-learning stability under current short-horizon imagination |
| nuPlan Stage 1 | Decoupled-no-vis | better planning-oriented policy learning |
| nuPlan planner-like / interaction subset sanity | Decoupled-no-vis | better downstream offline planner-like behavior, especially in lane-conflict / low-TTC subsets |

最适合论文的当前叙事：

> Decoupled typed-slot abstraction reliably improves representation sufficiency. Its downstream policy-learning benefit is not universal: on nuScenes short-horizon latent imagination, object-only remains the most stable policy baseline, while on the more planning-oriented nuPlan preprocessed benchmark, decoupled no-visibility becomes the robust winner. This shows that relation-aware abstractions are useful when the downstream planning regime and token quality make relation structure actionable.

---

## 12. 下一步实验安排

### 12.1 可视化 case study

50k 主实验、offline planner-like sanity check 和 dataset statistics 已经完成，下一步最有价值的是补 4 组可视化例子，而不是继续扩大条件数量。每组图建议包含 token/slot 选择、关键 agent 位置或 imagined trajectory、collision/action/planner proxy 对比：

| Case type | Purpose | Candidate source | Figure contents |
|---|---|---|---|
| naive mixing misses key agent | 解释 shared top-k 中 relation 与 dynamic agent 抢 slot 的失败模式 | Stage 0 `object_relation` vs `object_relation_decoupled` | selected token type bars + missed dynamic agent position |
| decoupled keeps key agent | 展示 typed dyn/rel budget 如何保住关键 agent | Stage 0 `object_relation_decoupled` / `object_relation_decoupled_visibility` | dyn slots + rel slots + nearest dynamic prediction |
| nuScenes object-only is more stable | 展示 nuScenes Stage 1 中表示更好不等于 policy 更稳 | `experiments/stage1_pilot_x/seed*/{wm_object,wm_decoupled}` | imagined collision/action magnitude/stability comparison |
| nuPlan decoupled-no-vis is better | 展示 nuPlan 上 decoupled-no-vis 的 planner-like 优势 | `experiments/nuplan_stage1_50k` and `experiments/nuplan_planner_sanity_50k` | action MSE + imagined collision + selected dyn/rel context |

### 12.2 Appendix: typed-budget sensitivity

不建议再做大规模 budget sweep，但可以补一个低成本 appendix sensitivity，回答 reviewer 可能会问的“为什么是 12/4，是否拍脑袋调参”。推荐只做一个数据集、少量 seed、少量 budget ratio：

- 10/6；
- 12/4；
- 14/2。

定位应是 appendix sanity，不重新打开 Stage 1 主叙事，也不继续扩大 actor/critic/fusion sweep。当前已经测试过 default 12+4、no-vis、14+2、rel-to-critic-only；这些已经足够说明简单 budget/fusion 微调不是 nuScenes Stage 1 的主要解法。若补 typed-budget sensitivity，更适合在 Stage 0 或 nuPlan 小规模设置上报告“性能对 budget ratio 不敏感 / 12+4 不是孤立最优点”，而不是再追求新的最佳数值。

仍然更值得优先做的是：

1. imagination-time relation selection consistency regularization；
2. relation/risk head 与 actor state 更彻底分离；
3. 用 closed-loop 或 reactive setting 检验 relation 结构是否在交互场景中真正发挥作用。

### 12.3 Appendix: interaction-heavy subset analysis

如果还有余力，比继续扫新 benchmark 更有价值的是做 interaction-heavy subset analysis，回答“什么时候 relation 有用”：

| Subset | Selection idea | Expected use |
|---|---|---|
| rare-agent heavy | 按 rare dynamic object 数或动态 token 密度分桶 | 检查 decoupled-no-vis 优势是否集中在多主体/长尾 agent 场景 |
| low-TTC / high-risk | 用 offline TTC proxy、future clearance 或 collision proxy 选取高风险样本 | 检查 relation-aware abstraction 是否主要改善真正需要交互推理的样本 |
| merge / crossing / cut-in | 用 nuPlan scenario type 或 nuScenes interaction tags 过滤 | 支撑“relation helps when relation is actionable”的论文叙事 |

这个分析只需要复用已有 checkpoint 和离线/日志数据，优先级高于继续做新 wrapper 或新 benchmark。

### 12.4 外部闭环 status and limitation

当前已经完成 nuPlan devkit closed-loop MVP：`wm_object` 与 `wm_decoupled_no_vis` 可以作为 oracle-token planner 接入官方 nonreactive closed-loop simulation，并输出官方 metric parquet / nuboard / runner report。action-to-trajectory rollout 已取代 constant-action wrapper，并在 1/5/50-scenario sanity 中让 progress 不再系统性为 0。冻结后的 corridor projection 小规模官方表可作为 appendix sanity：它解决 drivable/progress/comfort，但两个 planner 几乎并列，说明该闭环结果仍主要由 wrapper 决定。NAVSIM 仍未接入，CARLA 暂不建议作为当前优先项。

论文 limitation 建议写法：

> We provide small nuPlan-devkit closed-loop smoke/sanity tests using oracle-token DOOR-RL planners, but do not claim a full-scale official closed-loop benchmark result. After freezing the wrapper to route/lane corridor projection, the official small-subset metrics show strong drivable/progress/comfort behavior but little separation between object-only and decoupled-no-vis policies, suggesting that the wrapper still dominates the closed-loop outcome. The main nuPlan results therefore rely on preprocessed log-derived NPZ data, latent imagination rollouts, and offline planner-like probes.

### 12.5 Stage 2/3 规划

| Stage | 目标 | 当前状态 |
|---|---|---|
| Stage 2 | 反应式闭环训练/评估，验证 reactive training 的必要性 | 尚未实现 |
| Stage 3 | 高保真迁移评估，如 CARLA/NAVSIM/nuPlan closed-loop | 尚未实现 |

---

## 13. 复现实验命令草稿

### 13.1 Stage 0 nuScenes

```bash
PYTHONPATH=src python run_stage0_table3.py \
  --config configs/debug_mvp.json \
  --dataset nuscenes \
  --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
  --num-scenes 700 \
  --variant all_with_decoupled \
  --epochs 15 \
  --batch-size 32 \
  --seed 7
```

三 seed 正式结果由脚本批量运行并聚合：

```bash
bash scripts/run_fix2_3seeds.sh
PYTHONPATH=src python scripts/aggregate_fix2_seeds.py
```

### 13.2 Stage 1 nuScenes X

```bash
bash scripts/run_stage1_pilot_x.sh
PYTHONPATH=src python scripts/aggregate_stage1_x.py
```

### 13.3 Stage 1 nuPlan 20k 单条件示例

```bash
PYTHONPATH=src python run_stage1_table4.py \
  --dataset nuplan \
  --nuplan-root /mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split \
  --nuplan-num-samples 20000 \
  --nuplan-index-json experiments/nuplan_20k_balanced_paths_seed7.json \
  --nuplan-workers 32 \
  --condition wm_decoupled_no_vis \
  --stage0-root experiments/nuplan_stage0_20k_seed7 \
  --epochs 10 \
  --batch-size 128 \
  --horizon 5 \
  --entropy-beta 0.003 \
  --action-sample-clip 5 \
  --seed 7
```

### 13.4 nuPlan 50k offline planner-like sanity check

```bash
PYTHONPATH=src python scripts/offline_planner_sanity.py \
  --nuplan-root /mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split \
  --nuplan-num-samples 50000 \
  --nuplan-index-json experiments/nuplan_50k_balanced_paths_seed7.json \
  --stage1-root experiments/nuplan_stage1_50k \
  --output-dir experiments/nuplan_planner_sanity_50k \
  --seeds 7 42 123 \
  --conditions wm_object wm_decoupled_no_vis \
  --batch-size 128 \
  --loader-workers 32 \
  --horizon 5
```

### 13.5 nuScenes vs nuPlan dataset statistics

```bash
PYTHONPATH=src python scripts/dataset_token_stats.py \
  --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
  --nuscenes-num-scenes 700 \
  --token-cache-dir experiments/_token_cache \
  --nuplan-root /mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split \
  --nuplan-num-samples 50000 \
  --nuplan-index-json experiments/nuplan_50k_balanced_paths_seed7.json \
  --output-dir experiments/dataset_token_stats \
  --batch-size 256 \
  --loader-workers 16
```

### 13.6 nuPlan devkit closed-loop MVP

```bash
PYTHONPATH="src:/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/nuplan-devkit" \
  /mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/flow_planner/bin/python \
  scripts/run_nuplan_closed_loop_mvp.py \
  --nuplan-devkit-root cangku/nuplan-devkit \
  --nuplan-data-root /mnt/datasets/e2e-nuplan/20260302/val \
  --nuplan-maps-root /mnt/datasets/e2e-nuplan/20260302/maps \
  --scenario-filter all_scenarios \
  --limit-total-scenarios 50 \
  --conditions wm_object wm_decoupled_no_vis \
  --horizon-seconds 8.0 \
  --speed-scale 2.0 \
  --yaw-rate-scale 0.6 \
  --output-dir experiments/nuplan_closed_loop_action_rollout_50scenario
```

汇总：

```bash
/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/flow_planner/bin/python \
  scripts/summarize_nuplan_closed_loop.py \
  experiments/nuplan_closed_loop_action_rollout_50scenario/doorrl_closed_loop_mvp/seed7_closed_loop_nonreactive_agents
```

---

## 14. 文档维护规则

后续新增实验时建议同时更新：

1. 对应实验目录下的 `summary.md` 与 `summary.json`；
2. `docs/stage1_pilot.md` 的对应小节；
3. 本文档的结果表、原始路径索引与“当前可靠结论”；
4. 若某个旧结论被推翻，把它移到“历史实验与被推翻结论”，不要删除。

