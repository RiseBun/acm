# NeurIPS 补实验计划（2026-04-27）

这份文档的目标不是重新打开无限扩实验，而是把当前工作收束成一套**足以支撑 NeurIPS 投稿**的最小实验计划。重点是回答：

1. 现在这篇工作离 NeurIPS 还差什么；
2. 哪些实验最能证明方法有效，而不是继续堆工程；
3. 哪些实验必须做，哪些可以只放 appendix，哪些不建议再做。

---

## 1. 当前论文已经有什么

目前已经有一条相当完整的主证据链：

- `Stage 0 / nuScenes / 3 seeds`
  证明 typed-budget decoupling 在固定 16-slot budget 下显著提升 representation sufficiency。
- `Stage 1 / nuScenes / 3 seeds`
  证明更好的表示不会自动转化为更好的 policy learning；当前 `wm_object` 更稳。
- `Stage 1 / nuPlan 20k / 3 seeds`
  证明 ranking reversal 开始出现。
- `Stage 1 / nuPlan 50k / 3 seeds`
  证明 `wm_decoupled_no_vis` 是稳定主条件。
- `planner-like sanity / nuPlan 50k`
  证明优势不只停留在 latent return，而能延伸到 teacher-action alignment。
- `interaction-conditioned subset analysis`
  证明 relation-aware abstraction 的优势主要出现在 `lane_conflict / low_ttc / dense interaction / rare_agent_dense` 这些真正需要交互建模的场景。
- `cross-dataset eval`
  证明排名反转与 dataset / planning regime 有关，而不是“某个模型天然全局更强”。
- `nuScenes horizon sensitivity`
  证明 nuScenes Stage 1 上 decoupled 的问题会随着 rollout horizon 增大而累积，更像 imagination stability 问题。

因此，**当前最大短板已经不是“内部证据不够”，而是“外部锚点不足”和“Stage 1 机制链条还差最后一个下游对照”。**

---

## 2. NeurIPS 视角下最可能被问的三个问题

如果按 NeurIPS 的 reviewer 视角看，当前最容易被追问的是：

1. **是不是主要在和自己比？**
   现在内部 family ablation 很完整，但外部 baseline 还不够。
2. **typed-budget 的下游收益，和 naive shared relation 相比到底如何？**
   Stage 0 已经说明 shared mixing 会崩，但 Stage 1 还缺一个直接下游对照。
3. **`12/4` 是不是拍出来的？**
   目前有 `12/4` 和 `14/2`，但还缺一个更 relation-heavy 的点，比如 `10/6`。

所以 NeurIPS 方向不需要再加很多新 benchmark，最值钱的是补齐这三类问题中的前两类。

---

## 3. 必须补的实验（Priority A）

以下两组实验建议视为 **NeurIPS 主文增强项**。如果只能补最少的内容，优先做这两组。

### A1. 外部非同族 baseline

### 目的

给论文一个“不是只和自己家族比”的锚点。

### 推荐方案

优先采用成本最低、解释最直接的 baseline：

1. `BC / planner-target imitation`
   输入保持相同 token schema，直接监督 teacher action。
2. 如果已有现成脚本或 checkpoint，则优先复用。
3. 除非已经有非常低成本的现成实现，否则不要为了这个 baseline 再接入复杂外部系统。

### 推荐数据与设置

- 主数据集：`nuPlan 50k`
- 如成本允许：补一个 `nuScenes 700` 小规模版本，放 appendix 即可
- seeds：`7, 42, 123`

### 最低报告指标

- `latent_return_mean`
- `imagined_collision_rate`
- `collision_mean`
- `teacher_action_mse`
- `action_delta_l2`

### 产物要求

- `summary.md`
- `summary.json`
- 每个 seed 的原始 metrics
- 和现有 `wm_object / wm_decoupled_no_vis` 完全同格式，便于并表

### 命名建议

- `experiments/nuplan_bc_baseline_50k/`
- 或 `experiments/nuplan_planner_imitation_50k/`

### 成功标准

这个 baseline **不需要打赢** `wm_decoupled_no_vis`。  
它的价值是回答 reviewer 的问题：

> 你们的方法不是只和自己家的变体比吗？

只要能稳定进入同一张表，它就有价值。

---

### A2. Stage 1 下游的 naive shared-relation baseline

### 目的

把整篇论文的机制链条真正闭合：

- Stage 0：shared relation mixing 失败
- Stage 1：如果直接拿 shared relation 去做 policy learning，会怎样
- Stage 1：typed-budget decoupling 是否能恢复并超过它

### 推荐方案

在 `nuPlan 20k` 或 `nuPlan 50k` 上跑一个最小 shared-relation Stage 1 条件：

- `wm_shared_relation`
- 或者从现有 `object_relation` / `object_relation_visibility` 变体改出一个可以稳定跑 Stage 1 的条件

### 推荐执行顺序

1. `1 seed smoke`
2. 如果训练可运行、数值不完全崩坏，再补到 `3 seeds`

### 最低报告指标

- `latent_return_mean`
- `imagined_collision_rate`
- `collision_mean`
- `rollout_stability`
- 如果现成可取，再加 `teacher_action_mse`

### 命名建议

- `experiments/nuplan_stage1_shared_relation_20k/`
- `experiments/nuplan_stage1_shared_relation_50k/`

### 成功标准

这组实验的关键不是“shared relation 一定要特别差”，而是要形成一个明确、可解释的对照：

- 如果它显著差于 `wm_decoupled_no_vis`，机制链条最完整；
- 如果它训练高方差或频繁崩坏，也仍然是有价值的负结果，但要清楚记录 failure mode；
- 只有在它和 `wm_decoupled_no_vis` 很接近且稳定时，才需要重新评估主叙事。

---

## 4. 建议补、但更适合 appendix 的实验（Priority B）

这些实验有帮助，但不建议在 Priority A 之前投入太多时间。

### B1. typed-budget sensitivity：补 `10/6`

当前已有：

- `12/4`
- `14/2`

还缺一个更 relation-heavy 的点，例如：

- `10/6`

### 目的

回答 reviewer 的问题：

> 为什么正好是 `12/4`？是不是调参拍出来的？

### 推荐设置

- 首选：`Stage 0 / nuScenes`
- 如果成本低：再加一个 `nuPlan small-scale`
- seeds：能保持 `3 seeds` 最好；如果代价太高，至少先做 `1 seed smoke`

### 进入论文的方式

- appendix table
- supplementary figure

不需要进主文。

---

### B2. relation feature-group ablation

### 目的

现在我们已经能说“relation 在某些场景有用”，但还不太能说“哪类 relation 最关键”。  
如果还有余力，建议做一个**粗粒度** feature-group ablation。

### 推荐分组

- 去掉 `TTC / risk`
- 去掉 `lane_conflict / priority`
- `visibility on/off` 当前已有，可直接复用已有结论，不需要重新大做

### 推荐设置

- 数据：`nuPlan 20k` 或 `nuPlan 50k`
- 先做单 seed feasibility
- 如果结果明确，再决定是否扩成 `3 seeds`

### 进入论文的方式

- appendix
- discussion support

它的主要作用是增强解释，不是决定主线是否成立。

---

## 5. 不建议继续投入的方向

以下方向当前不建议作为 NeurIPS 主线继续投入：

### 5.1 继续深挖 official closed-loop wrapper engineering

原因：

- frozen official small table 已经说明当前 official closed-loop aggregate 更像 wrapper feasibility；
- 继续投入很可能是工程耗时高、论文增益低；
- 对 NeurIPS 而言，这种“被 wrapper 主导”的结果不如机制清晰的 analysis 有说服力。

保留原则：

- closed-loop 结果只放 appendix / sanity
- 不再把它作为主结果路线

---

### 5.2 继续做全条件大扫

包括但不限于：

- 更多 visibility 变体
- 更多 actor/critic fusion 变体
- 更多 budget ratio sweep
- 更多单 seed exploratory pilot

原因：

- 当前主结论已经形成；
- 新扫参很容易增加噪声，却不能明显提升主文说服力。

---

### 5.3 继续依赖单 seed 结果得出主结论

统一原则：

- 单 seed 只作为 smoke / feasibility
- 主结论必须以 `3 seeds` 为准

---

## 6. 推荐执行顺序

如果按投入产出比排序，建议按下面顺序推进：

1. **A2：Stage 1 shared-relation baseline**
   这是最直接补机制链条缺口的实验。
2. **A1：外部 BC / planner-target imitation baseline**
   这是最直接补“不是只和自己比”的实验。
3. **B1：`10/6` budget sensitivity**
   解决 reviewer 对 `12/4` 的自然质疑。
4. **B2：relation feature-group ablation**
   如果前面结果顺利，再补解释性增强。

---

## 7. 最小可交付边界

如果资源有限，建议把 NeurIPS 补实验的最小边界定成：

### 主文新增

- 一个 `external baseline` 行
- 一个 `shared-relation Stage 1 baseline` 行

### Appendix 新增

- 一个 `10/6` ratio 点
- 一个粗粒度 relation-group ablation（可选）

只要做到这一层，整篇论文就会从：

> 内部机制很完整，但外部锚点偏弱

变成：

> 既有机制性 insight，也有下游验证和最基本的外部参照

这已经足以支撑一版更像 NeurIPS 的投稿稿。

---

## 8. 除了实验，还要同步推进的事

即使实验补齐，NeurIPS 版仍然还需要同步做下面几件事：

1. 主文压到 `9 pages content`
2. 全文匿名化（作者、机构、自引、repo 线索）
3. 从 `T-IV` 叙事改成 `ML / structured world model / capacity allocation` 叙事
4. 主文只保留最关键的 2-3 张表和 2-3 张图，其他放 appendix

也就是说，这轮补实验的目标不是把论文写得更“大”，而是让核心论点更“干净”和更可被 ML reviewer 接受。
