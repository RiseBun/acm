# T-IV 补充实验要求（2026-04-27）

这份文档的目标不是重新打开无限扩实验，而是明确：

1. 当前论文是否已经足够形成一篇完整投稿稿；
2. 如果还要补实验，哪些最值得做；
3. 哪些方向不建议继续投入。

---

## 1. 当前结论是否已经足够

结论：**已经足够形成一篇完整的主论文，不存在必须补做的硬性 blocker。**

当前已经具备的主证据链：

- `Stage 0 / nuScenes / 3 seeds`
  证明 typed-budget decoupling 在固定 16-slot budget 下显著提升 representation sufficiency。
- `Stage 1 / nuScenes / 3 seeds`
  证明更好的表示不会自动转化为更好的 imagination policy learning；当前 `wm_object` 更稳。
- `Stage 1 / nuPlan 20k / 3 seeds`
  证明 ranking reversal 开始出现。
- `Stage 1 / nuPlan 50k / 3 seeds`
  证明 `wm_decoupled_no_vis` 是稳定主条件。
- `offline planner-like sanity / nuPlan 50k`
  证明优势不只停留在 latent return，而能延伸到 teacher-derived action 对齐。
- `interaction-conditioned subset analysis`
  证明关系结构的收益在 lane conflict / low TTC / dense interaction 场景中最明显。
- `cross-dataset eval`
  证明 ranking reversal 与 dataset / planning regime 有关，而不是简单架构输赢。
- `nuScenes horizon sensitivity`
  证明 nuScenes Stage 1 上 decoupled 的问题随 rollout horizon 累积，更像 imagination stability 问题。

因此：

- 当前稿件已经可以按“**typed-budget abstraction 的方法与分析论文**”继续推进。
- 后续实验只应服务于“补外部锚点”和“补审稿人会问的关键空缺”，不应该再打开大规模探索。

---

## 2. 最值得补的实验

这里按优先级分成 `Priority A` 和 `Priority B`。

---

## 2.1 Priority A：如果只补 1-2 组实验，优先做这些

### A1. 外部非同族 baseline

### 目的

当前论文最大的短板不是内部对比不够，而是**外部 published-style baseline 不够**。  
我们已经证明了：

- `typed-budget` 比 `shared relation mixing` 更合理；
- 在 nuPlan 上 `wm_decoupled_no_vis` 比 `wm_object` 更强；

但还缺一个“不是只和自己比”的锚点。

### 推荐实现

优先选择**成本最低**的外部参考，而不是重新接入复杂新系统：

1. `BC / planner-target imitation` baseline  
   输入保持同样的 token schema，直接监督 teacher action。
2. 如果 `BC` 已经存在现成 checkpoint 或代码路径，就优先复用。
3. 如果要更进一步，再考虑一个简单 `holistic latent policy` 作为外部风格参考。

### 最低要求

- 数据：`nuPlan 50k`
- seeds：`7, 42, 123`
- 指标：
  - `latent_return_mean`
  - `imagined_collision_rate`
  - `collision_mean`
  - `teacher_action_mse`
  - `action_delta_l2`
- 输出：
  - `summary.md`
  - `summary.json`
  - 每个 seed 的原始 metrics

### 命名建议

- `experiments/nuplan_bc_baseline_50k/`
- 或 `experiments/nuplan_planner_imitation_50k/`

### 进入论文的标准

只要它能作为一个合理锚点进入表格即可，不要求一定赢它。  
目标是回答 reviewer 的问题：

> 你们的方法不是只和自己家族里的变体比较吗？

---

### A2. Stage 1 下游的 shared-relation baseline

### 目的

当前机制链条在表示层已经完整：

- shared relation mixing 在 `Stage 0` 会崩；
- decoupled 会恢复；

但在 `Stage 1` 下游里，主表还缺一个**naive shared-relation policy baseline**。  
这会让 reviewer 问：

> 你们说 typed-budget 对 policy learning 有帮助，那和 naive shared relation 的下游表现相比呢？

### 推荐实现

在 `nuPlan 20k` 或 `nuPlan 50k` 上跑一个最小 shared-relation 下游条件：

- `wm_shared_relation`
- 或从现有 `object_relation_visibility / object_relation` 改成 Stage 1 可运行条件

### 最低要求

- 先做 `1 seed smoke`
- 如果结果可运行、不会完全数值崩坏，再补 `3 seeds`
- 只需要主指标：
  - `latent_return_mean`
  - `imagined_collision_rate`
  - `collision_mean`
  - `rollout_stability`

### 命名建议

- `experiments/nuplan_stage1_shared_relation_20k/`
- `experiments/nuplan_stage1_shared_relation_50k/`

### 进入论文的标准

如果 shared-relation 下游显著劣于 `wm_decoupled_no_vis`，它就可以非常强地闭合整篇论文的机制链条。  
如果它训练完全不稳定，也仍然是有价值的负结果，但必须清楚记录失败模式。

---

## 2.2 Priority B：适合放 appendix 的增强实验

### B1. typed-budget sensitivity：补 `10/6`

当前已有：

- `12/4`
- `14/2`

还缺一个更“relation-heavy”的点，比如：

- `10/6`

### 目的

回答 reviewer 典型问题：

> 为什么正好是 `12/4`？是不是拍脑袋调出来的？

### 建议

- 首选 `Stage 0 / nuScenes`
- 如果成本低，再加一个 `nuPlan small-scale`
- 不需要上主文；放 appendix 即可

### 输出

- `summary.md`
- `summary.json`
- 和已有 `12/4`、`14/2` 同格式

---

### B2. relation feature-group ablation

### 目的

现在我们已经能说“relation 在某些场景有用”，但还不太能说“哪类 relation 最关键”。  
如果还有余力，可以做一个**粗粒度** feature-group ablation：

- 去掉 `visibility`
- 去掉 `TTC / risk`
- 去掉 `lane conflict / priority`

### 建议

- 不要做太细碎的单 feature ablation
- 只做 2-3 组 coarse ablation
- 优先在 `nuPlan 20k` 或 planner-like sanity 上做

### 进入论文的标准

更适合放 appendix，帮助解释：

- 为什么 `no_vis` 在 nuPlan 上更强
- 到底是 `TTC` 还是 `lane conflict` 更关键

---

## 3. 不建议继续做的方向

以下方向当前**不建议继续投入为主线**：

### 3.1 继续深挖 official closed-loop wrapper engineering

原因：

- frozen official small table 已经说明现阶段 wrapper 主导结果；
- 继续投入很可能是工程耗时高、论文增益低。

保留原则：

- closed-loop 结果放 appendix / sanity
- 不再把它作为主结果路线

---

### 3.2 继续全条件大扫

包括但不限于：

- 更多 visibility 变体
- 更多 actor/critic fusion 变体
- 更多 budget ratio 大面积 sweep

原因：

- 当前论文的主要结论已经形成；
- 新扫参很容易只增加噪声，不增加主结论强度。

---

### 3.3 继续依赖单 seed pilot 做结论

当前应统一原则：

- 单 seed 只作为 smoke / feasibility
- 主结论必须以 `3 seeds` 为准

---

## 4. 推荐的最终实验边界

如果资源有限，建议直接把实验边界定在下面：

### 主文保留

- Stage 0 / nuScenes
- Stage 1 / nuScenes
- Stage 1 / nuPlan 20k
- Stage 1 / nuPlan 50k
- planner-like sanity
- interaction-conditioned subsets
- cross-dataset eval
- horizon sensitivity

### Appendix / Support

- official closed-loop sanity
- 14+2
- rel-to-critic-only
- typed-budget sensitivity (`10/6` 如果补出来)
- relation feature-group ablation（如果补出来）

---

## 5. 所有新增实验的统一记录要求

后续如果再补实验，必须统一满足：

1. `summary.md`
2. `summary.json`
3. per-seed 原始 metrics
4. 使用的 checkpoint 路径
5. 使用的数据 split / cache 路径
6. 明确标注：
   - `main-claim eligible`
   - `appendix-only`
   - `discarded`

建议所有新实验在 `docs/discarded_vs_reliable.md` 里同步更新状态。

---

## 6. 一句话执行建议

如果只补最值得的实验：

1. 先补 **一个外部非同族 baseline**
2. 再补 **一个 Stage 1 shared-relation downstream baseline**

如果资源再紧张：

**停止扩实验，直接做图、补参考文献、统一术语、冲投稿版。**
