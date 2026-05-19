# 服务器主仓阅读与论文对齐记录

更新时间：2026-04-27

这份文档的目的，是把服务器 `Host lipeinan` 上主仓

`/mnt/volumes/cpfs/prediction/lipeinan/code`

里的代码、实验、结果和当前论文主线重新对齐，并用一份适合“不熟悉项目的合作者”阅读的中文说明，把下面几个问题讲清楚：

1. 这个仓库里到底有什么，哪些文件是“路线图”，哪些文件是“当前事实”。
2. 我们的方法在代码里究竟是怎么实现的。
3. 我们已经做过哪些实验，哪些结果是当前可靠结论。
4. 我们现在应该写成一篇什么样的论文。
5. 还缺什么，哪些工作不值得继续投入。

---

## 1. 结论先行

先给一个最重要的总判断：

**远端仓库已经足够支撑我们当前这篇 T-IV 主线。**

更准确地说：

- 远端代码和实验结果支持我们现在的论文定位：
  - 一篇关于 **typed-budget object-relational abstraction（类型化预算的目标-关系抽象）** 的方法与分析论文
  - 核心问题是：**在固定 latent budget（潜在上下文预算）下，relation token（关系 token）什么时候有用，为什么有用**
- `docs/experiment_report.md` 已经是成熟、可信的“主叙事”版本
- 顶层 `README.md` 仍然保留较早期的路线图口径，不能当作当前论文结论来源
- official nuPlan closed-loop 已经真正接通，但 **目前还不足以作为主结果**
- 真正最有解释力的“下游行为证据”不是 frozen official 5-scenario 小表，而是：
  - `nuPlan 20k / 50k Stage 1`
  - `offline planner-like sanity`
  - `interaction-conditioned subset analysis`

一句话概括：

**这篇论文现在最稳的形态不是“我们在 official closed-loop benchmark 上赢了”，而是“typed-budget abstraction 在表示层稳定有效，在 planning-oriented 的 nuPlan regime 下还能稳定兑现为 policy-learning 收益，且优势在 lane conflict / low TTC / dense interaction 等真正需要关系建模的子集上最明显”。**

---

## 2. 远端仓库概况

### 2.1 根目录

服务器主仓根目录：

```text
/mnt/volumes/cpfs/prediction/lipeinan/code
```

顶层结构可以粗分为：

- `src/doorrl/`
  - 核心源码
- `configs/`
  - 配置
- `docs/`
  - 实验报告和设计文档
- `experiments/`
  - 结果目录、日志、缓存、汇总
- `scripts/`
  - 实验驱动脚本、分析脚本、closed-loop runner
- `cangku/nuplan-devkit`
  - 本地 nuPlan devkit checkout
- `cangku/navsim`
  - NAVSIM 相关 checkout

### 2.2 哪些文档是“当前事实”

这一步非常重要，因为远端有新旧两套口径并存。

#### 应当作为当前事实来源的文档

- `docs/experiment_report.md`
- `docs/stage0.md`
- `docs/stage1_pilot.md`
- `experiments/nuplan_stage1_50k/summary.md`
- `experiments/nuplan_planner_sanity_50k/summary.md`
- `experiments/nuplan_interaction_subset_50k/summary.md`

#### 只能作为旧路线图/历史参考的文档

- `README.md`
- `docs/stage1_design.md`

原因：

- `README.md` 仍把 Stage 1 写成“进行中”，把 Stage 2/3 写成“规划中”，没有反映后续 `nuPlan 20k / 50k`、`planner-like sanity`、`interaction subset`、official closed-loop wrapper 演化
- `docs/stage1_design.md` 是早期 Stage 1 设计草稿，价值在于看“当时怎么想”，不是看“最后实际做成了什么”

### 2.3 与本地材料的一致性

我核对过：

- 远端 `docs/experiment_report.md`
- 本地 `experiment_report.md`

两者内容一致，可以视为同一份“主实验总报告”。

这意味着：

- 当前本地论文已经没有“结果口径落后远端”的问题
- 接下来的工作重点不是同步数字，而是把这些数字和代码事实讲得更清楚

---

## 3. 仓库里的核心概念，用中文解释

为了让没参与项目的合作者也能快速进入状态，这里把几个高频术语先翻成更直白的话。

### 3.1 World Model

这里的 **world model（世界模型）** 不是“做出逼真图片”的模型，而是：

**给定当前场景 token 和 ego 动作，预测下一步关键 token 会怎么变、奖励大概是多少、是否继续、有没有碰撞风险。**

也就是说，它更像一个：

- 面向决策的动态预测器
- 而不是一个视觉生成器

### 3.2 Latent Imagination

**latent imagination（潜在空间想象）** 是指：

- 不去真实 simulator 里一步一步开车
- 也不去渲染图像
- 而是在 world model 学出来的 latent / token 空间里，往前 rollout 几步

它的意义是：

- 训练便宜
- 可以做很多策略尝试
- 但风险是：如果 latent state 本身不够“决策充分”，策略学到的东西会不可靠

### 3.3 Object Token / Relation Token

- **object token（目标 token）**：表示车、人、骑行者等动态智能体
- **relation token（关系 token）**：表示 ego 和其他体之间，或局部交互之间的关系，例如：
  - 相对位置
  - 相对速度
  - TTC（time-to-collision，碰撞时间估计）
  - lane conflict（车道冲突）
  - priority（优先级）
  - risk（风险分数）

### 3.4 Typed-Budget Decoupling

这是当前方法的核心。

传统做法是：

- 把 object token 和 relation token 一起放进同一个 top-k 选择器

这会导致：

- relation token 和 dynamic agent token 抢同一个 16-slot 预算

我们的方法是：

- 动态体单独选 `K_dyn`
- 关系边单独选 `K_rel`
- 最后拼起来送进同一个 world model

也就是：

```text
K_dyn + K_rel = K
```

在当前主实验里是：

```text
K_dyn = 12, K_rel = 4, K = 16
```

这就是所谓的：

- `typed-budget abstraction`
- `decoupled abstraction`
- `typed-slot abstraction`

本质上都是一件事：**按语义角色分预算，而不是让所有 token 混在一起抢名额。**

### 3.5 Oracle-Token Evaluation

这篇论文现在不是端到端感知控制论文。

所谓 **oracle-token evaluation** 指的是：

- token 不是从 noisy perception 网络直接端到端出来的
- 而是来自标注、预处理数据、或者 simulator state 的可信前端

这样做的意义是：

- 先把“表示是否足够支撑决策”这个科学问题单独研究清楚
- 不让 perception noise 把结论搅乱

### 3.6 Planner-Like Sanity

这不是 official closed-loop benchmark。

它是一个更“靠近 planner 行为”的离线 sanity check：

- 看策略动作和 teacher-derived planner action 的差距
- 看 imagined collision
- 看 imagined return

它能说明：

- 策略是不是更像 planner

但它不能说明：

- 已经在真正 external closed-loop 里更强

### 3.7 Closed-Loop Wrapper

这里的 **wrapper（包装层）** 指的是：

- 把 DOOR-RL policy 输出的 2D action
- 转成 nuPlan 官方 simulator 需要的 ego trajectory

为什么这个东西重要？

因为如果 wrapper 很强、很保守，它本身就能把很多指标“救起来”；
反过来，如果 wrapper 很弱，policy 再好也会死在执行层。

这也是为什么现在 official closed-loop 小表不能当主结果：

**指标里混进了太多 wrapper 本身的影响。**

---

## 4. 代码层面的核心实现

这一节不是逐文件背代码，而是讲清楚“方法在代码里到底怎么落地”。

### 4.1 模型变体：`src/doorrl/models/doorrl_variant.py`

这是最关键的源码文件之一。

它定义了几个主要变体：

- `holistic`
  - 97 token 全量上下文
  - 上界参考
- `holistic_16slot`
  - 用 learned queries 把 97 token 压到 16 个 slot
  - 公平压缩参考
- `object_only`
  - 只选动态体
- `object_relation`
  - object + relation 混在同一个 shared top-k
  - 这是 naive mixing
- `object_relation_visibility`
  - 在 shared top-k 基础上加 visibility weighting
- `object_relation_decoupled`
  - 两路独立 top-k，`12 + 4`
- `object_relation_decoupled_visibility`
  - decoupled + visibility weighting

最重要的实现事实：

1. `top_k_dyn + top_k_rel` 必须等于 `top_k`
2. dynamic 和 relation 用的是两个独立的 abstraction head
3. relation head 不强制选择 ego
4. 最后 dynamic slot 和 relation slot 拼接成统一 16-slot 输入

这说明 typed-budget 不是“训练后分析时的人为解释”，而是模型结构本身的一部分。

### 4.2 Actor-Critic Head：`src/doorrl/models/policy.py`

这个文件揭示了 Stage 1 一个很重要的数值稳定性设计。

策略头输出是二维连续动作高斯分布：

- action 维度 1：前向速度相关量
- action 维度 2：yaw rate（偏航角速度）相关量

为了避免 imagination RL 早期爆炸，代码做了两个关键限制：

1. `action_mean` 使用 `3 * tanh(raw / 3)`
   - 防止动作均值发散到几十、几百
2. `action_log_std` 限制在 `[-2, 0.5]`
   - 防止高斯采样方差无限大或过小

这和实验报告里 Stage 1 的“数值修复历史”是对得上的。

另外，这个 head 支持：

- actor 和 critic 用相同 latent
- 或者 critic 用更丰富 latent、actor 只看 dynamic latent

这就是后面 `rel_to_critic_only` fusion ablation 的代码基础。

### 4.3 nuPlan 数据集加载：`src/doorrl/data/nuplan_dataset.py`

这个文件说明了为什么 nuPlan 会和 nuScenes 形成很不一样的 Stage 1 行为。

关键点：

1. nuPlan 使用的是 **preprocessed NPZ**
   - 每个 NPZ 基本就是一个 anchor frame 样本
   - 不是像 nuScenes 那样从原始 scene/sample 动态 tokenise
2. 代码支持：
   - 从预先生成的 JSON index 读路径
   - 避免在几十万 NPZ 文件上做大规模 filesystem walk
3. 可以 lazy load，也可以并行预 materialize cache
4. 20k / 50k 实验里大量使用并行 worker 提高 tokenisation 吞吐

这个数据管线的实际后果是：

- nuPlan 的 token 分布更密
- rare agent 更多
- teacher action scale 更大
- visibility 近乎恒定

这些都能帮助解释：

- 为什么 `wm_decoupled_no_vis` 会在 nuPlan 上变成最强

### 4.4 nuPlan 适配器：`src/doorrl/adapters/nuplan_adapter.py`

这个文件很关键，因为它告诉我们：

**relation token 在 nuPlan 路径里到底长什么样。**

当前适配器会从 ego 和 tracked objects 构造：

- 相对位置 `dx, dy`
- 相对速度 `rel_vx, rel_vy`
- `distance`
- `ttc`
- `risk`
- `lane_conflict`
- `priority`
- `is_interactive`
- `visibility`

其中一个非常重要的事实是：

- 在这个 adapter 里，`visibility` 基本被写成 `1.0`

这和数据统计里“nuPlan visibility 近乎恒定”完全一致，也解释了为什么：

- visibility weighting 在 nuScenes 上可能有帮助
- 在 nuPlan 上则未必必要，甚至可能成为噪声

### 4.5 official nuPlan planner wrapper：`src/doorrl/closed_loop/nuplan_oracle_planner.py`

这是把 DOOR-RL checkpoint 接进官方 nuPlan simulation loop 的核心包装器。

它做的事情不是“直接输出轨迹”，而是：

1. 把 nuPlan `PlannerInput` 转成 DOOR-RL token
2. 让 Stage 1 checkpoint 输出 2D action
3. 把这个 action 映射成目标速度和 yaw rate
4. 再通过安全投影、lane corridor、lead-vehicle controller、TTC proxy 等逻辑
5. 生成最终 ego trajectory

这正是我们现在 closed-loop 结果必须谨慎解读的原因：

- policy 不是直接裸奔进 simulator
- 中间有一层很强的 trajectory wrapper / safety wrapper

所以如果 wrapper 质量主导了指标，就不能把 final score 解释成“policy 本身更强”。

### 4.6 closed-loop runner：`scripts/run_nuplan_closed_loop_mvp.py`

这个 runner 说明：

- official nuPlan closed-loop 不是“概念上想做”，而是已经在工程上打通了
- 它能加载：
  - `wm_object`
  - `wm_decoupled_no_vis`
- 能切换：
  - corridor projection
  - safety projection
  - lead vehicle controller
  - TTC proxy

从论文角度，这个脚本证明了：

- 我们不是没接过 external closed-loop
- 而是接通以后发现 **当前结果更像 wrapper feasibility，而不是方法优劣证据**

### 4.7 interaction subset 脚本：`scripts/interaction_subset_analysis.py`

这是当前最有价值的新脚本之一。

它复用 `nuPlan 50k Stage 1` checkpoint 和 val split，在离线验证集上划出更“需要关系推理”的子集：

- `low_ttc_proxy`
- `lane_conflict`
- `dense_agents`
- `rare_agent_dense`
- `high_interaction_union`

然后比较：

- `teacher_action_mse`
- `latent_return`
- `imagined_collision_rate`

这个脚本的研究价值非常高，因为它直接回答了：

**relation-aware abstraction 什么时候真正有用？**

答案不是模糊的“平均更好”，而是：

- 在 `lane_conflict`
- 在 `low_ttc`
- 在 `dense / rare interactive` 场景

优势最明显。

---

## 5. 已完成实验的完整脉络

### 5.1 Stage 0：nuScenes 表示充分性

这是当前最稳、最没有争议的一部分。

数据规模：

- 700 scenes
- 28,096 samples
- 3 seeds
- 公平 16-slot budget

主结论：

- naive `object+relation` 在 shared top-k 下会崩
- `typed-budget decoupling` 可以恢复并超过 `object-only`

最值得记住的指标差异：

- `Object-only-16` vs `Obj+Rel-Decoupled`
  - DynRoll: `3.7449 -> 2.1148`
  - Rare ADE: `1.0964 -> 0.4913`
  - IntRec@1m: `0.9009 -> 0.9842`

这部分回答的是：

**在固定上下文预算下，怎样的抽象更“决策充分”。**

### 5.2 Stage 1：nuScenes latent imagination RL

这里得到的是一个重要的“负结果，但有研究价值”的结论。

主结论：

- Stage 0 表示更好，不代表 Stage 1 policy learning 自动更好
- 在当前 `K=5`、short-horizon、nuScenes imagination 设定里，`wm_object` 更稳

这非常重要，因为它阻止我们把论文写成“representation 好，policy 就一定好”的过度简化故事。

### 5.3 Stage 1：nuPlan 5k / 20k / 50k

这是项目后期最大的突破。

#### nuPlan 20k

`wm_decoupled_no_vis` 明显优于 `wm_object`：

- Return: `17.50 ± 1.37` vs `4.74 ± 13.95`
- Collision: `0.226 ± 0.105` vs `0.373 ± 0.083`

#### nuPlan 50k

20k 的结论在更大规模上得到了确认：

- Return: `14.511 ± 2.925` vs `1.723 ± 17.886`
- Imagined collision: `0.259 ± 0.045` vs `0.610 ± 0.146`

更重要的是：

- `wm_decoupled_no_vis` 三个 seed 全部为正
- `wm_object` 仍然高方差

这说明：

**nuPlan 是当前最能稳定兑现 decoupled policy-learning 收益的下游设定。**

### 5.4 Offline planner-like sanity

这是一个“更接近 planner 行为”的离线 probe。

`wm_decoupled_no_vis` 优于 `wm_object`：

- Teacher action MSE: `6.628 ± 0.110` vs `8.863 ± 0.370`
- Action ΔL2: `2.115 ± 0.083` vs `3.553 ± 0.118`
- Collision: `0.259 ± 0.045` vs `0.610 ± 0.146`

这部分说明：

- `wm_decoupled_no_vis` 的优势不是只体现在 latent reward 上
- 它也更接近 teacher-derived planner action

### 5.5 Interaction-conditioned subset analysis

这是当前最有解释力的下游分析。

最关键的子集是 `lane_conflict`：

| Condition | Action MSE ↓ | Return ↑ | Collision ↓ |
|---|---:|---:|---:|
| `wm_object` | 7.023 | 0.943 | 0.591 |
| `wm_decoupled_no_vis` | 4.225 | 13.330 | 0.205 |

其他子集也一致支持：

- `low_ttc_proxy`
  - collision `0.791 -> 0.541`
- `rare_agent_dense`
  - collision `0.626 -> 0.273`
- `dense_agents`
  - collision `0.637 -> 0.272`
- `high_interaction_union`
  - collision `0.625 -> 0.270`

这个结果的重要意义是：

**我们不再只是说 decoupled 在平均数上更好，而是能说清楚“它在哪些场景里更有用”。**

### 5.6 official nuPlan closed-loop sanity

这部分的地位必须讲清楚。

已经完成的事实：

- DOOR-RL checkpoint 已经通过 oracle-token planner wrapper 接入 official nuPlan simulation loop
- corridor projection 等 wrapper 已经把 drivable / progress / comfort 基本稳定住

但 frozen official 5-scenario 小表没有拉开：

| Planner | Score | Collision | Drivable | Progress | Comfort | TTC |
|---|---:|---:|---:|---:|---:|---:|
| `wm_object` | 0.537 | 0.700 | 1.000 | 1.000 | 1.000 | 0.400 |
| `wm_decoupled_no_vis` | 0.537 | 0.700 | 1.000 | 1.000 | 1.000 | 0.400 |

所以结论是：

- official closed-loop 已打通
- 但当前这张表主要反映的是 wrapper feasibility
- 还不能证明 `wm_decoupled_no_vis` 在 official closed-loop 中优于 `wm_object`

这部分只能放：

- appendix
- discussion
- limitation

不能升格成主结果。

---

## 6. 这篇论文现在到底是什么论文

根据远端代码和全部实验，最稳的论文定位是：

**一篇关于 typed-budget object-relational abstraction 的方法与分析论文。**

更具体地说，是：

**在固定 latent budget 下，relation-aware abstraction 何时有效、为什么有效，以及何时它的表示收益能够兑现为 policy-learning 收益。**

不应该写成：

- 一个 official closed-loop benchmark SOTA 论文
- 一个 end-to-end perception-to-control 论文
- 一个“大而全”的系统论文

应该写成：

- 有明确机制发现
- 有结构性方法设计
- 有跨数据集 policy-learning 分析
- 有解释“什么时候 relation 有用”的子集分析

---

## 7. 当前最可靠的研究故事

我建议所有合作者统一使用下面这条主线。

### 7.1 Stage 0

在固定 16-slot latent budget 下，naive object+relation mixing 会发生 **inter-type budget competition（类型间预算竞争）**，导致 dynamic agent slot 被 relation slot 抢占，表示质量反而崩掉。

typed-budget decoupling 通过显式分配 `K_dyn=12`、`K_rel=4`，稳定解决了这个问题。

### 7.2 Stage 1 on nuScenes

更好的表示质量不会自动转化为更好的 imagination policy learning。

在当前 short-horizon nuScenes imagination 设定下，`wm_object` 仍然更稳。

这说明：

- representation gain
- policy-learning gain

不是同一个问题。

### 7.3 Stage 1 on nuPlan

在 planning-oriented 的 nuPlan preprocessed benchmark 上，ranking 发生反转：

- `wm_decoupled_no_vis` 成为稳定最强条件

这说明：

**relation-aware abstraction 的 policy-learning 价值是 downstream-regime dependent（依赖下游规划设定的）。**

### 7.4 Why relation matters

interaction-conditioned subset analysis 进一步表明：

- decoupled 的优势不是平均偶然更好
- 而是在 lane conflict、low TTC、dense/rare interactive scenes 这类真正需要关系建模的场景里更明显

这正好回答了：

**relation 什么时候真正有用？**

---

## 8. 对论文主稿的直接影响

结合这次远端阅读，我认为主稿应该坚持下面几个边界。

### 8.1 应当强调的点

1. `typed-budget abstraction` 是结构性方法，不是后处理 trick
2. `inter-type budget competition` 是被 Stage 0 明确观测到的 failure mode
3. `nuPlan` 上 `wm_decoupled_no_vis` 的优势，与其 tokenization / planning regime 有关
4. `interaction-conditioned subset analysis` 是“why relation matters”的核心证据

### 8.2 应当克制的点

1. 不要把 current official closed-loop 写成主胜利
2. 不要写成“decoupled everywhere wins”
3. 不要把 `planner-like sanity` 当成 external benchmark
4. 不要把 oracle-token 结果描述成 end-to-end autonomy

### 8.3 最值得在文中补充的代码事实

1. nuPlan relation token 的在线构造包含：
   - relative geometry
   - TTC
   - lane conflict
   - risk
   - priority
2. nuPlan adapter 中 visibility 近乎恒定，这有助于解释为什么 `no_vis` 在 nuPlan 上更强
3. Stage 1 actor 输出使用：
   - bounded mean
   - clamped log std
   - detached world model

这些事实能让论文对“为什么是 dataset-dependent”讲得更实。

---

## 9. 还缺什么，哪些事不用再做

### 9.1 现在还缺的，主要不是新核心实验

如果按当前主线投稿，最缺的是：

1. 首图 / teaser
2. 方法图
3. 图表 summary
4. case-study 可视化
5. 替换占位参考文献
6. 作者信息、单位、bio
7. supplement 整理

### 9.2 可以作为可选增强项的实验

如果还想补一个低成本附录实验，优先级最高的是：

- `typed-budget sensitivity`
  - 例如补一个 `10/6`

这主要是为了回答 reviewer 可能会问的：

- 为什么是 `12/4`？

### 9.3 不建议继续投入的方向

基于远端仓库现状，我不建议继续无上限投入：

1. official closed-loop wrapper engineering
2. 更多 actor/critic fusion 横扫
3. 更多 budget 组合横扫
4. 没有 stop rule 的新 benchmark 扩展

原因很简单：

- 这些方向已经证明“能做”
- 但继续堆时间，不一定继续提升当前论文的说服力

---

## 10. 推荐给新合作者的阅读顺序

如果有完全不了解项目的合作者加入，我建议按这个顺序看：

1. 先看本文件  
   建立对“项目现在真正是什么”的整体认识
2. 再看 `experiment_report.md`  
   了解完整实验链和当前可靠结论
3. 再看 `doorrl_tiv_main.tex`  
   看论文主线如何组织
4. 如果要看代码，优先看：
   - `src/doorrl/models/doorrl_variant.py`
   - `src/doorrl/models/policy.py`
   - `src/doorrl/data/nuplan_dataset.py`
   - `src/doorrl/adapters/nuplan_adapter.py`
   - `src/doorrl/closed_loop/nuplan_oracle_planner.py`
5. 如果要看“为什么 relation 在某些场景更有用”，直接看：
   - `scripts/interaction_subset_analysis.py`
   - `experiments/nuplan_interaction_subset_50k/summary.md`

---

## 11. 最终判断

远端主仓已经证明三件事：

1. **我们的方法在表示层面是站得住的。**
2. **它对 policy learning 的收益不是普遍的，而是依赖下游规划设定。**
3. **在真正需要关系推理的 nuPlan interaction-heavy 子集上，它的优势是清楚而且一致的。**

因此，现在最值得做的不是继续改论文定位，而是：

- 把图画出来
- 把 case study 做好
- 把论文语言打磨到更像 T-IV regular paper

如果后面真的还要补实验，优先级也应该是：

- 附录级 `typed-budget sensitivity`

而不是再把 official closed-loop 硬推成主结果。
