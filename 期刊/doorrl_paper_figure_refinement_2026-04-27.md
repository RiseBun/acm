# DOOR-RL 论文细化与 Figure 设计清单

更新时间：2026-04-27

这份文档的目标，是把当前论文从“已经有主线和结果”推进到“接近投稿版”的状态。它主要回答 3 个问题：

1. 这篇论文最终应该写成什么样。
2. 现在正文各部分还要怎么细化。
3. 每一张 figure 具体应该画什么，用什么数据，传达什么信息，caption 怎么写。

这份文档默认读者是项目组内部作者，所以我会同时保留：

- 中文解释
- 英文论文中的标准表达

---

## 1. 当前论文的最终形态

### 1.1 最终定位

这篇论文最合适的定位不是：

- 一个 official closed-loop benchmark SOTA 论文
- 一个 end-to-end perception-to-control 系统论文
- 一个“大而全”的自动驾驶 RL 平台论文

而是：

**一篇关于 typed-budget object-relational abstraction 的方法与分析论文。**

更准确地说，是：

**在固定 latent budget 下，relation-aware abstraction 何时有效、为何有效，以及何时它的表示收益能够兑现为 policy-learning 收益。**

### 1.2 一句话 thesis

建议所有作者统一用下面这句作为核心 thesis：

**Relation-aware driving world models do not fail or succeed only because of what they encode, but also because of how latent capacity is allocated across semantic roles.**

如果需要更具体一点，可以用：

**Under a fixed latent budget, naive shared dyn/rel selection causes inter-type budget competition, whereas typed-budget decoupling restores decision-sufficient interaction structure and makes relation-aware policy learning useful in planning-oriented downstream regimes.**

### 1.3 当前最稳的结论链

现在最稳的论文故事应该始终按这 4 步讲：

1. `Stage 0 / nuScenes`
   - typed-budget decoupling 在表示层稳定成立
2. `Stage 1 / nuScenes`
   - 更好的表示不自动意味着更好的 policy learning
3. `Stage 1 / nuPlan 20k + 50k`
   - ranking reversal，`wm_decoupled_no_vis` 成为稳定最强条件
4. `interaction-conditioned subset analysis`
   - relation-aware gain 在 lane conflict / low TTC / dense interaction 场景里最强

这个结构非常重要，因为它让论文不是“一个 everywhere-win 的宣传稿”，而是“一个有条件结论的成熟研究论文”。

---

## 2. 正文细化计划

### 2.1 标题

当前标题：

`Typed-Budget Object-Relational Abstractions for Imagination-Based Driving Policy Learning`

这个标题已经是可投稿级别，可以先保留。

如果后面要微调，最多考虑两种方向：

1. 更方法化  
   `Typed-Budget Object-Relational Abstraction for Imagination-Based Driving Policy Learning`
2. 更分析化  
   `When Relation-Aware Abstractions Help Driving Policy Learning: Typed-Budget Object-Relational World Models`

当前我更建议保留原版，不再改标题。

### 2.2 摘要

摘要现在已经有正确主线，但还可以进一步优化成“更快抓住贡献”的版本。

摘要里必须保留的 4 个点：

1. 问题不是 relation token 本身，而是 semantic capacity allocation
2. `nuScenes Stage 0`：typed-budget 表示优势成立
3. `Stage 1`：这个收益是 dataset-dependent
4. `nuPlan 50k + subset analysis`：优势在 planning-oriented / interaction-heavy 条件下兑现

摘要里不建议展开的内容：

- official closed-loop wrapper 演化过程
- 太多具体 ablation 细节
- 太多 visibility 的旁支解释

### 2.3 引言

引言最终应当完成 4 个任务：

1. 说明为什么 driving policy learning 需要 relation-aware structure
2. 说明“把 relation token 加进去”不是充分答案
3. 说明 fixed latent budget 下存在 semantic competition
4. 说明我们不是在写 final benchmark paper，而是在研究 representation-to-policy interface

引言结尾最好形成一个非常清晰的读图路径：

- Fig. 1：总览论文故事
- Fig. 2：方法结构
- Table 1 / Table 2 / Fig. 3 / Fig. 4：实验链

### 2.4 Related Work

Related Work 不需要再继续扩很大，但要更明确地区分 3 条线：

1. world models / imagination RL
2. object-centric / relation-aware representations
3. driving benchmarks / planning-oriented evaluation

不要把它写成“大综述”；重点是把我们的工作嵌入这 3 条线，并说明我们回答的是“latent budget allocation”这个更具体的问题。

### 2.5 Method

Method 现在要达到的目标，不只是“能看懂”，而是“让 reviewer 觉得这个方法真的落在结构设计上”。

Method 里必须说透的点：

1. token schema 里 relation token 到底编码什么
2. 为什么 shared top-k 会产生 slot competition
3. `K_dyn + K_rel = K` 是结构约束，不是后处理
4. Stage 1 actor-critic 的关键数值稳定设计：
   - bounded mean
   - clamped log std
   - detached world model
   - sanity loss

### 2.6 Experiments

实验部分的组织建议固定为：

1. `Stage 0: Representation Sufficiency on nuScenes`
2. `Stage 1: Cross-Dataset Imagination Policy Learning`
3. `nuPlan Planner-Like Sanity`
4. `Interaction-Conditioned Subset Analysis`
5. `Discussion / limitation of official closed-loop sanity`

这意味着 official closed-loop 不再有独立主结果 subsection，而是：

- discussion
- appendix

### 2.7 Discussion

Discussion 应该明确回答 3 个问题：

1. 为什么 Stage 0 和 Stage 1 不总是一致？
2. 为什么 nuScenes 和 nuPlan 排名相反？
3. 为什么 visibility 的作用是 dataset-dependent？

这里可以继续强调：

- visibility 在 nuPlan adapter 里几乎恒定
- nuPlan token / action statistics 明显不同
- subset analysis 证明 gain 的确集中在 interaction-heavy 条件

### 2.8 Limitation

limitation 里最好主动承认 4 件事：

1. 仍然是 oracle-token evaluation
2. Stage 1 主要是 imagination policy learning，不是 final external benchmark
3. official closed-loop 已接通，但 wrapper 仍然主导结果
4. visibility 的作用还没有完全被理论解释

---

## 3. Figure 总体策略

### 3.1 总原则

当前最推荐的主文图结构是 5 张：

1. **Fig. 1 Teaser**
2. **Fig. 2 Method Overview**
3. **Fig. 3 Stage-0 Mechanism Figure**
4. **Fig. 4 Summary Charts**
5. **Fig. 5 Cross-Dataset Case Studies**

当前 LaTeX 主稿已经给了这些图位：

- teaser
- method
- stage0 slot distribution
- charts summary
- cross-dataset case-study

外加一张当前已插入的 `case_02` 单案例图。

最终建议是：

- `case_02` 不一定单独保留成最终主文独立 figure
- 更理想的做法是把它吸收到 `Fig. 5 Cross-Dataset Case Studies` 里

### 3.2 哪些图是“已有资产可直接复用”

当前已经有的、可直接复用或轻微调整的图：

1. `figures/stage0_slot_distribution.png`
2. `figures/stage0_variance_summary.png`
3. `figures/scenes/case_00_idx53.png`
4. `figures/scenes/case_01_idx54.png`
5. `figures/scenes/case_02_idx78.png`
6. `figures/scenes/case_03_idx81.png`
7. `figures/scenes/case_04_idx82.png`

服务器上的对应生成脚本：

1. `scripts/plot_slot_distribution.py`
2. `scripts/plot_stage0_variance.py`
3. `scripts/plot_slot_scenes.py`
4. `scripts/dataset_token_stats.py`

### 3.3 哪些图还需要新制作

需要新制作的主要有：

1. Teaser figure
2. Method overview figure
3. Summary charts 的最终 composite 版本
4. nuPlan / Stage-1 case-study panels

这些图里：

- `Teaser` 和 `Method Overview` 更适合手工矢量图
- `Summary Charts` 更适合脚本生成
- `Case Studies` 需要“已有 stage0 scene plot + 新的 nuPlan case export”

---

## 4. Figure 1：Teaser / 首图

### 4.1 目的

这张图的任务不是解释全部细节，而是让读者 5 秒钟内知道这篇论文在讲什么。

必须让读者一眼看到 3 件事：

1. naive shared top-k 会错失关键动态体
2. typed-budget decoupling 能修复这个问题
3. Stage 1 的收益不是 everywhere-win，而是在 nuPlan 上兑现

### 4.2 推荐版式

建议横向 3 栏：

1. **左栏：Failure**
   - 选一个最典型的 Stage 0 case
   - 显示 naive `Object+Relation-16` 漏掉近场关键 agent
   - 可直接用 `case_02_idx78` 的 naive panel 裁图

2. **中栏：Fix**
   - 同一个 case 的 decoupled panel
   - 直接对比“0/5 near-field dyn selected” vs “5/5 selected”

3. **右栏：Outcome**
   - 一个极简 summary 小图
   - 最好是两行文字 + 两个小箭头/条形图：
     - `nuScenes Stage 1: WM-Object best`
     - `nuPlan 20k/50k: WM-Decoupled-NoVis best`

### 4.3 推荐内容元素

- 一个圆圈标出 `15 m near-field`
- 一个红色 callout：
  - `shared top-K -> relation steals dynamic slots`
- 一个蓝色 callout：
  - `typed 12+4 budget -> recover decision-critical agents`
- 右栏可以加一句小字：
  - `policy-learning gain is downstream-regime dependent`

### 4.4 数据/素材来源

- 左/中：`figures/scenes/case_02_idx78.png`
- 右：来自主表数值
  - `nuScenes Stage 1`
  - `nuPlan 20k`
  - `nuPlan 50k`

### 4.5 Caption 草案

**Draft caption**

`Fig. 1. Paper overview. Under a fixed 16-slot latent budget, naive shared object-relation selection can allow relation tokens to displace near-field dynamic agents that are critical for driving decisions. The proposed typed-budget abstraction restores those agents by allocating separate dynamic and relation budgets under the same total capacity. This representation gain does not translate uniformly to policy learning: object-only remains the strongest Stage-1 baseline on nuScenes, whereas the no-visibility decoupled variant becomes the robust winner on planning-oriented nuPlan.`

### 4.6 制作方式

建议手工做：

- Figma
- Keynote / PowerPoint
- Illustrator

这张图不建议完全靠脚本自动生成，因为它需要讲“故事顺序”。

---

## 5. Figure 2：Method Overview / 方法图

### 5.1 目的

这张图要让 reviewer 一眼看明白：

1. 输入 token 长什么样
2. naive shared top-k 和 typed-budget decoupling 的区别
3. 选出来的 slot 如何进入 world model
4. Stage 1 的 imagination actor-critic loop 在哪里

### 5.2 推荐版式

建议 4 段从左到右：

1. **Scene tokenization**
   - ego
   - object tokens
   - relation tokens
   - map tokens

2. **Naive shared top-k**
   - 一个 16-slot shared selection box
   - 显示 dyn / rel 混在一起
   - 标注 `inter-type budget competition`

3. **Typed-budget decoupling**
   - dyn branch -> top 12
   - rel branch -> top 4
   - concat -> 16 slots

4. **Shared world model + imagination RL**
   - selected slots + action -> world model
   - world model -> next tokens / reward / collision / continue
   - global latent -> actor / critic

### 5.3 必须写清楚的小字

建议图中直接标：

- `K = 16`
- `K_dyn = 12`
- `K_rel = 4`
- `same total budget as shared top-K`
- `oracle-token evaluation`

### 5.4 最值得强调的视觉对比

方法图一定要让人看到：

- shared top-k 是“所有 token 抢同一排椅子”
- decoupled 是“动态体和关系边分区入场”

这是整篇论文的视觉核心。

### 5.5 Caption 草案

**Draft caption**

`Fig. 2. Method overview of DOOR-RL. A structured driving scene is tokenized into ego, dynamic-object, relation, and map tokens. The key design choice is the abstraction stage: instead of selecting all tokens through one shared top-K bottleneck, the proposed typed-budget abstraction allocates separate budgets to dynamic-agent and relation tokens and then concatenates them into the same 16-slot world-model context. The selected slots feed a shared action-conditioned world model, while the pooled latent supports imagination-based actor-critic policy learning.`

### 5.6 制作方式

建议手工矢量图。

这张图是论文最重要的“方法识别图”，不建议完全用脚本拼。

---

## 6. Figure 3：Stage-0 Mechanism Figure

### 6.1 目的

这是机制图，不是总结图。

它的核心任务是：

**证明 naive mixing 的失败确实来自 slot-type composition，而不是纯随机。**

### 6.2 推荐主图

当前已经有非常合适的图：

- `figures/stage0_slot_distribution.png`

这张图的价值在于：

- 直接显示各 variant 的 16-slot 里，有多少给了 `REL`
- 有多少给了 dynamic agents

### 6.3 当前图的优点

它已经把最重要的机制讲出来了：

- `Object+Relation-16 (naive)`：
  - `10.5 / 16` 个 slot 给了 relation
  - 只有大约 `3.7 / 16` 给了 dynamic agents
- `Obj+Rel-Decoupled`：
  - 动态体和关系边分预算
  - 没有被 relation 抢掉 dyn slots

### 6.4 建议微调

当前图已经能用，但为了主文更紧凑，建议后面做 3 个小优化：

1. 标题缩短
2. legend 更紧凑
3. 图内只保留最关键的数值标签

### 6.5 Caption 草案

**Draft caption**

`Fig. 3. Slot-type composition under the fair 16-slot budget. In the naive shared-budget variants, relation tokens consume a large fraction of the available context, leaving too few slots for near-field dynamic agents. The proposed typed-budget abstraction enforces separate dynamic and relation allocations under the same total budget, preventing this inter-type competition.`

### 6.6 数据源与脚本

- 图文件：
  - `figures/stage0_slot_distribution.png`
- 服务器脚本：
  - `scripts/plot_slot_distribution.py`

---

## 7. Figure 4：Summary Charts / 总结图表

### 7.1 目的

这张图不是机制图，而是“结果的视觉总结图”。

它应该完成 3 个任务：

1. 让 Stage 0 的优势更直观
2. 让 Stage 1 的 ranking reversal 一眼可见
3. 让 dataset-dependent 的解释更容易接受

### 7.2 推荐版式

建议做成横向 3-panel：

1. **Panel (a): Stage 0 summary**
   - 用已有 `stage0_variance_summary.png` 的信息
   - 但最终最好不是原图整张直接塞进去
   - 更建议提炼成：
     - Dyn Rollout
     - Rare ADE
     - IntRec@1m
   - 只保留最关键的 4 个 variant：
     - Object-only
     - Naive Object+Relation
     - Obj+Rel-Decoupled
     - Decoupled+Vis

2. **Panel (b): Stage 1 cross-dataset ranking reversal**
   - 一个紧凑 grouped bar chart
   - x 轴：
     - nuScenes
     - nuPlan 20k
     - nuPlan 50k
   - 每组两个条：
     - `WM-Object`
     - `WM-Decoupled-NoVis`
   - y 轴建议两个小图并排：
     - Return
     - Collision Rate

3. **Panel (c): dataset statistics mini-panels**
   - dynamic tokens / sample
   - rare tokens / sample
   - visibility
   - teacher action L2

### 7.3 当前可复用资产

- `figures/stage0_variance_summary.png`

### 7.4 需要新生成的部分

需要新生成的内容：

- Stage 1 ranking reversal bar chart
- dataset statistics comparison mini-panels

最推荐的做法是新写一个论文专用脚本，例如：

- `plot_paper_summary_charts.py`

让它直接读取：

- `experiments/table3_fair_fix2_aggregate.json`
- `experiments/nuplan_stage1_20k/summary.md` 或原始 JSON
- `experiments/nuplan_stage1_50k/summary.md` 或原始 JSON
- `experiments/dataset_token_stats/summary.json`

### 7.5 Caption 草案

**Draft caption**

`Fig. 4. Compact visual summary of the main findings. (a) Stage-0 decision-oriented metrics show that naive shared relation mixing is unstable, whereas typed-budget variants preserve rare-agent and interaction-critical information under the same 16-slot budget. (b) Stage-1 ranking reverses across datasets: object-only is the most stable baseline on nuScenes, while decoupled-no-visibility becomes the robust winner on nuPlan 20k and 50k. (c) Dataset statistics provide explanatory context for this reversal: nuPlan has denser dynamic scenes, more rare agents, nearly constant visibility, and a larger teacher-action scale.`

### 7.6 版式建议

不要直接把当前 `stage0_variance_summary.png` 原封不动塞成 panel (a)，因为它太高、信息太多。

更好的做法是重新排一个“为主文服务”的版本。

---

## 8. Figure 5：Cross-Dataset Case Studies

### 8.1 目的

这张图要回答：

**为什么 Stage 0 的优势在 nuScenes Stage 1 不自动兑现，但在 nuPlan 上会兑现。**

也就是把“什么时候 relation matters”从表格变成可视化。

### 8.2 推荐版式

建议 4 panel：

1. **(a) Stage 0 failure case: naive misses key near-field agent**
2. **(b) Stage 0 fix case: decoupled recovers correct dynamic context**
3. **(c) nuScenes Stage 1: object-only is more stable**
4. **(d) nuPlan Stage 1: decoupled-no-vis aligns better and collides less**

### 8.3 Panel (a) / (b)

这两块可以直接基于现有 `plot_slot_scenes.py` 输出的 case 图。

最优先候选：

- `figures/scenes/case_02_idx78.png`

如果想更丰富，也可以从：

- `case_00_idx53.png`
- `case_01_idx54.png`
- `case_03_idx81.png`
- `case_04_idx82.png`

里再挑。

### 8.4 Panel (c) 需要什么

这一块当前还没有现成图资产。

应该画的是：

- `nuScenes Stage 1`
- `wm_object` vs `wm_decoupled`

建议内容：

- imagined collision probability
- action magnitude / stability
- 一个代表 scene 下的 rollout summary

重点不是“谁分数更高”，而是：

**为什么 object-only 更稳。**

### 8.5 Panel (d) 需要什么

这一块也需要新生成。

应该画的是：

- `nuPlan 50k`
- `wm_object` vs `wm_decoupled_no_vis`

建议内容：

- teacher action alignment
- imagined collision proxy
- interaction-heavy scene 的 selected dyn/rel context

重点不是“平均更强”，而是：

**在 lane conflict / low TTC 场景里，decoupled-no-vis 的优势可视化是什么样。**

### 8.6 Caption 草案

**Draft caption**

`Fig. 5. Cross-dataset case studies. (a) Under a shared object-relation bottleneck, naive mixing can miss near-field dynamic agents that are critical for decision making. (b) The proposed typed-budget abstraction recovers those agents under the same total 16-slot budget. (c) On nuScenes Stage 1, better Stage-0 representation does not automatically translate into a more stable imagination policy. (d) On planning-oriented nuPlan, the no-visibility decoupled policy achieves better action alignment and lower imagined collision, especially in interaction-heavy scenes.`

### 8.7 当前最现实的策略

如果短期内来不及把 (c)/(d) 做成完整 BEV rollout 图，最低限度可以先做：

1. `(a)/(b)` 用现有 Stage 0 case
2. `(c)` 用一个 nuScenes Stage 1 summary panel
3. `(d)` 用一个 nuPlan planner-like / subset summary panel

这样也能先把“图位和叙事”立住。

---

## 9. Table 也要一起细化

虽然你这次问的是 figure，但主文的 table 也应该和图一起收束。

### 9.1 主文保留的 table

建议主文保留这些：

1. Stage 0 representation ablation
2. Stage 1 cross-dataset table
3. planner-like sanity
4. interaction-conditioned subsets

### 9.2 不建议主文化的 table

以下内容更适合 appendix：

1. frozen official closed-loop sanity
2. 14+2 typed-budget ablation
3. relation-to-critic-only fusion ablation
4. full per-seed raw tables

---

## 10. 现在最值得立刻做的图工作

如果只按性价比排优先级，我建议：

### 第一优先级

1. Fig. 1 teaser
2. Fig. 2 method overview
3. Fig. 4 summary charts

因为这 3 张图决定 reviewer 第一遍浏览时，是否能快速抓住主线。

### 第二优先级

4. Fig. 5 case studies

这张图更像“证据加深器”，非常重要，但可以在 teaser / method / summary 之后做。

### 低优先级

5. official closed-loop sanity appendix figure

这个不急，不应再成为主文视觉中心。

---

## 11. 最小交付清单

### 11.1 论文本身

- 摘要压一轮
- 引言压一轮
- 方法里保持代码事实与文字对齐
- discussion / limitation 再 polish 一轮

### 11.2 图

- `Fig. 1 teaser`
- `Fig. 2 method overview`
- `Fig. 3 slot distribution` 微调版
- `Fig. 4 summary charts` 新版
- `Fig. 5 case studies`

### 11.3 附件

- per-seed tables
- dataset stats full table
- official closed-loop sanity appendix

---

## 12. 我对当前最优动作的建议

如果下一步要真正推进论文，我建议按这个顺序做：

1. 先把 `Fig. 2 method overview` 画出来  
   这是最核心的方法识别图
2. 再做 `Fig. 1 teaser`  
   把整篇故事压成一张图
3. 然后做 `Fig. 4 summary charts`  
   让 ranking reversal 和 dataset-dependent 证据一眼看见
4. 最后做 `Fig. 5 case studies`  
   把“什么时候 relation matters”视觉化

如果时间有限，最少也要完成前 3 张。
