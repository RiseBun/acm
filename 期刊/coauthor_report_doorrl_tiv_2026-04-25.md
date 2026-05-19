# DOOR-RL 给合作者的详细说明（面向第一次了解项目的人）

更新时间：2026-04-25

这份文档的目标，是用尽量清楚的中文，把我们现在这篇论文到底在做什么、为什么值得写、已经做了哪些实验、目前最可靠的结论是什么、还缺什么才能更稳地投稿到 IEEE Transactions on Intelligent Vehicles（T-IV），一次性说明白。

这份说明默认读者对项目背景并不熟悉，所以会先解释名词，再讲研究故事和实验结论。

---

## 1. 一句话总结

我们现在最适合写的，不是一篇“我们做了一个在所有设定上都最强的自动驾驶强化学习系统”的论文，而是一篇更聚焦、也更成熟的方法与分析论文：

**在自动驾驶的 world model（世界模型）里，relation token（关系 token）有没有用，关键不只取决于有没有 relation，而取决于有限的 latent budget（潜在表示容量）是否按语义角色被正确分配。**

我们的方法是：

**typed-budget object-relational abstraction**

中文可以理解为：

**一种带“类型化预算分配”的目标-关系抽象方法。**

它的核心思想是：

- 不要让动态目标 token 和关系 token 一起抢同一个 top-k 名额。
- 而是显式规定：给动态目标若干个 slot，给关系若干个 slot，总预算不变。
- 这样可以在相同计算预算下，保留更多真正对驾驶决策有用的信息。

当前最可靠的研究结论是：

- 在 **Stage 0（表示层分析）** 上，这个方法是明确成立的。
- 在 **Stage 1（latent imagination policy learning，潜在想象强化学习）** 上，这个方法不是“到处都赢”，而是**收益依赖下游数据集和规划设定**。
- 在 `nuScenes` 的 Stage 1 设定里，`object-only` 目前更稳。
- 在 `nuPlan 20k` 的 Stage 1 设定里，`wm_decoupled_no_vis` 明显最强且最稳定。

这比“方法 everywhere wins”的故事更克制，但也更像一篇可信的研究论文。

---

## 2. 先解释名词

为了让没有参与项目的人也能读懂，下面先解释这篇论文反复出现的术语。

### 2.1 什么是 world model（世界模型）

world model 可以理解成：

- 一个学习出来的“环境近似器”
- 它输入当前场景表示和 ego 车动作
- 输出下一时刻可能会发生什么

在我们的项目里，它会预测：

- 下一步的 token 表示
- reward（奖励）
- continue / terminal 信号
- collision risk（碰撞风险）

它的作用是：不必每一步都真的去昂贵环境里采样，而是在学习到的潜在空间里做“想象 rollout”。

### 2.2 什么是 latent（潜在表示）

latent 可以理解成：

- 原始场景经过神经网络编码后的压缩表示
- 不是图片，也不是原始检测框
- 而是一种更适合模型计算的中间状态

我们说 latent imagination，就是：

- 不在真实环境里一步一步跑
- 而是在 latent 空间里模拟未来几步

### 2.3 什么是 token

token 在这里不是大语言模型里的词 token，而是：

- 场景中的一个结构化元素

例如：

- ego 车本身可以是一个 token
- 附近的一辆车可以是一个 token
- 一个行人可以是一个 token
- 一段地图元素可以是一个 token
- 一个“关系”也可以是一个 token

### 2.4 什么是 object token 和 relation token

object token：

- 表示一个具体的实体
- 例如车辆、行人、自行车

relation token：

- 表示两个实体之间或者实体与环境之间的关系
- 例如时间到碰撞（TTC）、车道冲突、优先权、可见性等

直观上说：

- object token 告诉你“谁在那”
- relation token 告诉你“他们之间的交互是什么”

### 2.5 什么是 bottleneck / top-k / budget

bottleneck：

- 表示模型每一步真正能“看到”的信息容量有限

top-k：

- 从很多 token 里只选最重要的 `k` 个送进 world model

budget：

- 这个 `k` 就是预算
- 比如现在我们公平比较时统一用 `16` 个 slot

核心问题是：

- 如果总预算只有 16
- object token 和 relation token 应该怎么分？

### 2.6 什么是 typed-budget

typed-budget 的意思是：

- 按“类型”分预算

这里的类型主要是：

- dynamic tokens，也就是动态交通参与者
- relation tokens，也就是交互关系

我们当前的主设置是：

- `K_dyn = 12`
- `K_rel = 4`
- 总预算 `K = 16`

这不是增加预算，而是在相同预算下改变“谁有资格占这些位置”。

### 2.7 什么是 imagination-based policy learning

policy learning 指的是学一个驾驶策略，也就是：

- 给定当前状态
- 输出动作

imagination-based 的意思是：

- 策略不是只在真实环境里试出来
- 也会在 world model 生成的 imagined rollout 里更新

这和标准 model-free RL 的区别是：

- model-free 主要靠真实交互
- 我们这里会借助 learned world model

### 2.8 什么是 oracle-token evaluation

oracle-token 的意思是：

- 我们默认输入给策略和 world model 的 token 来自可靠标注或可信前端
- 不是完整的端到端 perception-to-control

这样做的好处是：

- 可以先把“表示和策略本身”这个科学问题单独研究清楚

这样做的限制是：

- 还不能直接声称“我们完成了端到端自动驾驶系统”

### 2.9 什么是 nuScenes 和 nuPlan

`nuScenes`：

- 一个广泛使用的自动驾驶数据集
- 很适合做感知、预测、结构化场景建模

`nuPlan`：

- 一个更偏规划和闭环评估逻辑的数据集/基准
- 从我们现在结果看，它更容易体现“表示是否真正对策略有帮助”

简化理解：

- `nuScenes` 更像“结构化驾驶场景建模数据”
- `nuPlan` 更像“更贴近规划决策的数据设定”

### 2.10 什么是 seed

seed 就是随机种子。

做 3 个 seed 的意思是：

- 同样的方法跑 3 次
- 看结果是否稳定

如果只看单个 seed，很容易误把偶然波动当作真实结论。

---

## 3. 我们真正想解决的问题是什么

这篇论文真正要解决的问题，不是“怎么做一个更复杂的大系统”，而是下面这个更尖锐的问题：

**在自动驾驶 world model 的固定表示容量下，怎样保留真正对驾驶决策有用的对象与交互信息？**

这个问题之所以重要，是因为当前自动驾驶强化学习通常面临三个常见矛盾。

### 3.1 高保真环境很贵

如果每一步训练都依赖高保真渲染和真实闭环模拟：

- 吞吐很低
- 训练成本很高
- 很难大规模做 RL

所以大家会自然想到 world model。

### 3.2 但 world model 不是天然就对决策有用

world model 只要能“预测下一步”，不代表它就保留了：

- 少数关键目标
- 近场危险对象
- 决策时真正需要的交互关系

驾驶决策经常由少量关键元素决定，而不是场景平均信息。

### 3.3 relation 不是加进去就一定有用

很多直觉上会觉得：

- 既然驾驶是交互问题
- 那加 relation token 应该就更强

但我们现在的实验告诉我们：

- 如果 relation token 和 dynamic token 抢同一个 top-k 名额
- 反而会把真正重要的动态目标挤出去
- 最终表示更差，策略也可能更差

所以真正的问题不是：

- “要不要 relation”

而是：

- **relation 应该怎样进入一个容量有限的 latent world model**

---

## 4. 这篇论文的动机

如果用适合写在引言里的方式来概括，我们的动机可以分成三层。

### 4.1 第一层动机：自动驾驶 RL 需要高效训练

真实或高保真的闭环训练很贵，所以 latent imagination 是合理方向。

### 4.2 第二层动机：驾驶不是平均意义上的场景理解，而是少数关键交互

驾驶决策往往取决于：

- 近场车辆
- 行人/自行车
- 横穿、并线、冲突车道
- 遮挡与优先权

所以表示是否“决策充分”，比是否“整体重建漂亮”更重要。

### 4.3 第三层动机：现有方法没有认真解决语义预算分配问题

如果把 object token 和 relation token 全都混进一个共享 bottleneck：

- 看上去更有结构
- 实际上可能发生 slot competition

这就是我们现在识别出的核心 failure mode：

**inter-type budget competition**

中文可以理解为：

**不同语义类型的 token 在有限容量里互相挤占。**

---

## 5. 这篇论文的故事应该怎么讲

现在最成熟、最稳的故事线，不是“我们的方法在所有地方都赢”，而是下面这条。

### 5.1 Stage 0：表示层面，typed-budget 明确有效

在固定 16-slot world-model context budget 下：

- naive object+relation mixing 会失败
- decoupled typed-budget abstraction 会稳定修复这个问题
- 并且在 rare-agent 和 interaction-critical 指标上优于 object-only

这一步证明的是：

- 我们的方法确实改善了表示质量
- 而且改善不是偶然 seed 漂动

### 5.2 Stage 1 on nuScenes：更好的表示，不一定自动变成更好的策略

在当前 nuScenes 的短 horizon latent imagination actor-critic 设定下：

- object-only 仍然最稳
- decoupled 在表示层更强，但策略层高方差

这一步很重要，因为它提醒我们：

- representation gain 不等于 policy gain

### 5.3 Stage 1 on nuPlan：排名反转，decoupled 的策略优势兑现

在 nuPlan 20k 上：

- `wm_decoupled_no_vis` 成为最强、最稳定条件
- object-only 不再是稳优 baseline

这说明：

- decoupled abstraction 不是全局失败
- 它的 policy-learning 价值是 **downstream planning regime dependent**

中文可以解释为：

**它是否真正帮助策略学习，取决于下游任务设定、数据组织方式和 token 质量。**

### 5.4 这个故事为什么反而更强

因为这篇论文现在不只是“提出一个方法”，而是在回答一个更研究型的问题：

**relation-aware abstraction 什么时候有用，为什么有用，什么时候又不能直接兑现为策略收益。**

这比“方法 everywhere wins”更像成熟论文。

---

## 6. 我们的方法到底做了什么

### 6.1 输入表示

每个场景被表示成最多 `97` 个 token，每个 token 有 `40` 维原始特征。

这些 token 包括：

- ego token
- dynamic object tokens
- map tokens
- relation tokens
- padding

关系特征里包含：

- TTC
- lane conflict
- priority
- visibility 等

对应设置见 [stage0.md](/Users/hb40355/Desktop/期刊/stage0.md:42) 和 [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:95)。

### 6.2 决策充分抽象模块

我们不是把全部 97 个 token 原样送进 world model，而是先做 token 选择。

一般做法是：

- 所有 token 一起打分
- 选 top-k

我们的方法不是这样。

我们的方法是：

- 先把 dynamic token 和 relation token 分开
- 在 dynamic token 上单独选 `K_dyn`
- 在 relation token 上单独选 `K_rel`
- 再拼接送进 world model

也就是：

`K_dyn + K_rel = K`

当前主设置是：

- `K = 16`
- `K_dyn = 12`
- `K_rel = 4`

这一点写在主稿的 [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:121)。

### 6.3 为什么这样设计

因为我们观察到：

- 如果 relation token 和 dynamic token 一起竞争同一个 16-slot shared top-k
- model 有时会选中太多 relation token
- 结果动态目标覆盖不够
- 最终动态预测和 rare-agent recall 崩掉

所以 decoupled abstraction 的核心不是“加更多 token”，而是：

**让有限预算更符合语义角色。**

### 6.4 world model 和 policy learning

world model 接收：

- 选出来的 latent tokens
- ego action

然后预测：

- 下一步 latent tokens
- reward
- continuation
- collision risk

策略学习部分使用：

- `K=5` 的 imagined rollout
- actor-critic
- GAE
- reward clipping
- detached world-model update

也就是我们现在的 Stage 1 设定，见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:124) 和 [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:153)。

---

## 7. 为什么我们相信现在的结果是可信的

这部分很重要，因为外部合作者往往会问：

“这些结果是不是中间有 bug，或者 evaluator 定义不稳？”

答案是：

- 我们确实经历过一轮比较完整的 metric / loss / evaluator 修正
- 但也正因为这轮修正，当前主结论反而更可信

在 Stage 0 中，我们做过一系列关键修正：

- 把原先饱和的 `Rare Recall @ 5m` 改成更有区分度的 `Interaction Recall @ 1m`
- 把无意义的 collision label 定义修正成 relation-token TTC 派生标签
- 修掉 visibility 没有真实梯度的问题
- 给 `Holistic-16Slot` 加了 set-prediction loss，避免它退化成“全学 ego”
- 把 obs loss 改成 **type-aware obs loss**
- 在 evaluator 中只允许 dynamic-type slot 参与 dynamic nearest matching

关键点在于：

- 即便做完这些修正
- naive object+relation 仍然失败

这恰恰说明：

- 失败不是 evaluator artifact
- 而是真正的结构问题

这个诊断链条可以看 [stage0.md](/Users/hb40355/Desktop/期刊/stage0.md:237)。

---

## 8. 到目前为止，我们已经做过哪些实验

### 8.1 Stage 0：nuScenes 表示充分性主实验

设置：

- 数据集：nuScenes
- 规模：700 scenes，28,096 samples
- split：scene-level 80/20
- 预算：统一 16-slot
- seeds：7、42、2026
- 变体数：6 个公平 16-slot 变体 + 1 个 97-token reference

主结果见 [stage0.md](/Users/hb40355/Desktop/期刊/stage0.md:9) 和 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:234)。

最关键的数值如下：

| Variant | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|
| Object-only-16 | 3.7449 ± 1.0099 | 0.9463 ± 0.0041 | 1.0964 ± 0.1159 | 0.9009 ± 0.0335 |
| Object+Relation-16 naive | 40.2822 ± 29.5376 | 0.9803 ± 0.0125 | 7.5060 ± 5.4799 | 0.4295 ± 0.4074 |
| **Obj+Rel-Decoupled** | **2.1148 ± 0.1889** | 0.9285 ± 0.0389 | **0.4913 ± 0.1768** | **0.9842 ± 0.0135** |
| **Decoupled+Visibility** | **1.8761 ± 0.2271** | 0.9257 ± 0.0290 | 0.5197 ± 0.0495 | 0.9787 ± 0.0078 |

这一步的结论非常明确：

- naive mixing 失败
- decoupled 成功
- 而且是稳定成功

相对 `object-only`，`Obj+Rel-Decoupled`：

- DynRoll 降低约 `44%`
- Rare ADE 降低约 `55%`
- IntRec@1m 提升约 `8.3` 个点

此外，naive mixing 的 `IntRec` 标准差高达 `0.407`，而 decoupled 只有约 `0.014`，说明 decoupled 不只是均值更好，稳定性也明显更强。

### 8.2 Stage 1：nuScenes latent imagination RL 主验证

设置：

- 数据集：nuScenes
- val samples：5,622
- horizon：5
- epochs：10
- seeds：7、42、123
- 主条件：`wm_object`、`wm_decoupled`、`wm_decoupled_no_vis`

结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:329)。

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Ego Stability (cos) |
|---|---:|---:|---:|---:|
| **wm_object** | **31.79 ± 19.70** | **0.597 ± 0.048** | **0.610 ± 0.048** | 0.258 ± 0.058 |
| wm_decoupled | 4.34 ± 13.66 | 0.695 ± 0.283 | 0.676 ± 0.243 | 0.636 ± 0.108 |
| wm_decoupled_no_vis | 0.34 ± 15.97 | 0.820 ± 0.260 | 0.814 ± 0.167 | 0.223 ± 0.033 |

这一步告诉我们：

- Stage 0 表示更好，不代表 Stage 1 策略一定更好
- 在当前 nuScenes 的 imagination RL 设定下，object-only 仍然最稳

### 8.3 Stage 1：nuScenes 辅助 ablation

我们还做了两个重要辅助实验。

第一个是 `14+2` typed-budget ablation：

- 目的：测试 `12+4` 是否只是因为 relation slot 太多
- 结果：`14+2` 不能修复问题，Return 仍低，Collision 仍差

结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:350)。

第二个是 `rel-to-critic-only` fusion ablation：

- 目的：测试 relation 信息是否只该进入 critic，而不直接喂给 actor
- 结果：有一点改善，但远不足以救回与 object-only 的差距

结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:373)。

这两个 ablation 的意义是：

- nuScenes Stage 1 的问题不是简单把 relation slot 减少一点就能解决
- 也不是简单把 relation 从 actor 拿掉就能解决

### 8.4 Stage 1：nuPlan 5k cross-dataset pilot

设置：

- 数据集：nuPlan preprocessed NPZ
- 规模：5,000 samples
- split：4,000 train / 1,000 val
- seeds：7、42、123

结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:413)。

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ |
|---|---:|---:|---:|
| wm_object | -6.01 ± 3.10 | 0.348 ± 0.254 | 0.384 ± 0.150 |
| wm_decoupled | 9.38 ± 9.59 | **0.215 ± 0.095** | **0.281 ± 0.097** |
| **wm_decoupled_no_vis** | **12.91 ± 2.69** | 0.247 ± 0.029 | 0.325 ± 0.060 |

这一步非常关键，因为它第一次显示：

- decoupled 在 Stage 1 不是全局失败
- 在更 planning-oriented 的 nuPlan 设定里，排名开始反转

### 8.5 nuPlan 20k Stage 0 warm-start

为了验证 5k pilot 不是偶然，我们扩展到了 20k。

Stage 0 warm-start 结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:496)。

| Variant | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---:|---:|---:|
| object_only | 168.733 | 0.534 | 17.981 | 0.294 |
| object_relation_decoupled_visibility | 3.210 | 0.724 | 0.848 | 0.987 |
| object_relation_decoupled | 3.210 | 0.724 | 0.848 | 0.987 |

这说明：

- 在 nuPlan 20k 的表示层上，decoupled 也依然非常强

### 8.6 nuPlan 20k Stage 1 主扩展实验

这是当前最重要的 Stage 1 正结果之一。

设置：

- 数据集：nuPlan 20k balanced subset
- split：16k / 4k
- seeds：7、42、123
- 条件：`wm_object`、`wm_decoupled`、`wm_decoupled_no_vis`

结果见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:538)。

| Condition | Return ↑ | CollRate ↓ | CollMean ↓ | Ego Stability (cos) |
|---|---:|---:|---:|---:|
| wm_object | 4.74 ± 13.95 | 0.373 ± 0.083 | 0.395 ± 0.057 | 0.426 ± 0.099 |
| wm_decoupled | 13.48 ± 4.09 | 0.488 ± 0.217 | 0.509 ± 0.193 | 0.546 ± 0.046 |
| **wm_decoupled_no_vis** | **17.50 ± 1.37** | **0.226 ± 0.105** | **0.251 ± 0.101** | **0.136 ± 0.022** |

更重要的是 per-seed 结果：

- `wm_decoupled_no_vis` 三个 seed 都是正 return
- `wm_object` 有两个 seed 是负 return

这说明：

- `wm_decoupled_no_vis` 不是偶然高分
- 而是当前最稳定的强条件

### 8.7 我们还做过哪些历史实验

还有一批历史实验和调试实验已经做过，但**不应该作为论文主结论引用**。

例如：

- 早期单 seed `stage1_pilot`
- `stage1_pilot_v3`
- `stage1_nanfix`
- `stage1_sanity`
- 一系列 Stage 0 smoke/sanity runs

这些实验的价值在于：

- 帮助我们定位 bug
- 理清方法真正的 failure mode

但它们不应被当成最终 scientific evidence。

汇总见 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:581)。

---

## 9. 当前最可靠的结论是什么

### 9.1 在表示层面

我们现在最有把握的结论是：

**在固定 16-slot budget 下，decoupled typed-budget abstraction 是当前最可靠的表示设计。**

它解决了：

- naive relation mixing 的 slot competition

并且改善了：

- dynamic rollout
- rare-agent displacement
- interaction recall

### 9.2 在策略学习层面

我们同样有一个很重要但更克制的结论：

**表示层的优势不会自动变成策略层的优势。**

具体来说：

- 在 `nuScenes Stage 1` 上，object-only 仍然最稳
- 在 `nuPlan Stage 1` 上，尤其是 `20k`，`wm_decoupled_no_vis` 反而最强

这意味着：

- relation-aware abstraction 是否真正帮助 policy learning
- 取决于数据设定、任务形态、token 化方式，以及 relation 信息是否对 actor 真正“可用”

### 9.3 关于 visibility 的结论

visibility 的作用现在不能写成“普遍增强项”。

我们现在更稳妥的说法是：

- 在 Stage 0，它对 dynamic rollout 有一点帮助
- 在 nuScenes Stage 1，它像 stabilizer
- 在 nuPlan 20k Stage 1，它反而不是最优，`no_vis` 最强

所以 visibility 的作用也是 dataset-dependent 的。

---

## 10. 我们现在应该写一篇什么样的论文

### 10.1 这篇论文最合适的定位

最合适的定位是：

**一篇关于 decision-sufficient semantic abstraction for imagination-based driving policy learning 的方法与分析论文。**

更直白一点说：

**一篇研究“关系信息应该如何进入容量受限的驾驶 world model，以及这种设计何时会真正帮助策略学习”的论文。**

### 10.2 这篇论文不应该写成什么

不应该写成：

- 一个完整端到端 perception-to-control 系统论文
- 一个最终闭环 benchmark 全面刷榜论文
- 一个纯 forecasting 论文
- 一个单纯“大而全 simulator”论文

因为当前证据并不支持这些更大的 claim。

### 10.3 这篇论文应该写成什么

应该写成：

- 有明确问题定义
- 有结构性 failure mode 发现
- 有方法设计
- 有跨数据集的政策学习分析
- 有边界、有局限、有条件性的研究结论

### 10.4 当前最稳的三条贡献

如果要写给审稿人看，最稳的三条 contribution 可以是：

1. 我们识别出 **inter-type budget competition**：在固定 latent budget 下，naive object+relation mixing 会让 relation token 与 dynamic token 争抢容量，导致 dynamic-agent coverage 下降。
2. 我们提出 **typed-budget object-relational abstraction**：通过显式划分 `K_dyn` 和 `K_rel`，在不增加总 context budget 的前提下提升 decision-sufficient representation。
3. 我们表明这种表示改进的 policy-learning 收益是 **dataset-dependent**：在 nuScenes Stage 1 中 object-only 仍最稳，而在 nuPlan 20k 中 decoupled-no-vis 成为 robust winner。

### 10.5 当前合适的论文标题方向

当前主稿标题方向是：

**Typed-Budget Object-Relational Abstractions for Imagination-Based Driving Policy Learning**

见 [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:29)。

---

## 11. 现在这篇论文还不能声称什么

这部分非常关键，因为决定了我们投稿时该如何控制 claim。

当前还不能强说：

- 我们已经完成了 end-to-end 自动驾驶
- 我们已经在外部高保真闭环 benchmark 上全面领先
- relation-aware abstraction 在所有数据集和所有设定下都优于 object-only
- visibility 一定有帮助
- Stage 1 的收益已经完全解释清楚

这些都不是现在证据支持的结论。

---

## 12. 我们还有哪些没做

### 12.1 科学问题上还没做完的

还没有完成的关键科学部分包括：

- **reactive training vs replay** 的直接实验证据
- 真正的 external closed-loop benchmark
- 高保真迁移验证，例如 CARLA / NAVSIM / 更完整的 nuPlan closed-loop
- perception noise 条件下的验证
- 对 `nuScenes` 与 `nuPlan` 排名反转的机制解释
- 更系统的 typed-budget sensitivity 分析

其中最重要的是：

- 我们现在还没有真正完成 Stage 2 / Stage 3

在内部规划里：

- Stage 2 对应 reactive training / evaluation
- Stage 3 对应高保真外部迁移

这在 [experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:704) 已明确写出。

### 12.2 论文材料上还没做完的

当前论文还缺：

- 一张更强的方法总览图
- 一张更清晰的 cross-dataset ranking figure
- 两条占位参考文献的正式替换
- 作者姓名、单位、通讯作者、bio
- 更统一的 related work 和 limitations 表达

### 12.3 如果只补最有价值的新实验

当前最值得补的，不是再盲目扫所有条件，而是：

- `nuPlan` 更大规模验证
- 优先只扩 `wm_object` 和 `wm_decoupled_no_vis`

因为这两者最能回答当前主线问题：

- decoupled-no-vis 的优势是不是稳定真实

---

## 13. 为了投 T-IV，我们还需要做什么

### 13.1 从研究内容上

如果以“尽快形成可投稿版本”为目标，我建议的优先级是：

1. 固定现在的论文主线，不再改方向。
2. 补最必要的新证据，优先 `nuPlan` 更大规模确认。
3. 把方法图、cross-dataset 图和讨论部分写得更成熟。
4. 清楚写出 scope 和 limitation，避免过度 claim。

### 13.2 从写作和材料上

截至我在 **2026-04-25** 查阅的 T-IV 官方页面，当前最相关的投稿要求包括：

- 投稿类型最适合走 `Regular Paper`
- 建议长度为 `10` 页
- 审稿为 `single-blind`
- 稿件格式为 IEEE journal 双栏、单倍行距
- 摘要要求 `150–250` 词、单段、无公式表格参考文献
- 需要 `3–4` 个关键词
- `Regular Paper` 需要在文末附所有作者 short biography
- 投稿前要做 checklist，包括 COI、机构邮箱、作者信息不能有非英文字符等

官方页面还说明：

- 超页可以，但录用后每超一页要付 `175 USD`

需要注意的一点是：

- 我在 **2026-04-25** 查到的两个 T-IV 官方页面都显示最近更新于 **2025-06-30**
- 但它们对 OA 费用和部分稿件类别名称有轻微不一致

所以正式投稿当天，仍建议以 Author Portal 实际显示为准再核一次。

相关官方信息来源：

- [T-IV Author Information](https://ieee-itss.org/pub/t-iv/author/)
- [T-IV Overview](https://ieee-itss.org/pub/t-iv/)

---

## 14. 如果要向完全不了解项目的作者解释，我们最推荐的口头版本

可以直接这样讲：

“我们这篇论文研究的是，自动驾驶 world model 在表示容量受限时，relation 信息到底应该怎么进模型。我们发现，简单把 relation token 混进 shared top-k bottleneck 反而会失败，因为 relation 会挤占 dynamic agent 的稀缺 slot。为了解决这个问题，我们提出了一种 typed-budget 的 decoupled abstraction，把动态目标和关系分别分配预算，在不增加总 context 的前提下显著提升了表示层质量。更有意思的是，这种表示优势并不会在所有 Stage 1 策略学习任务里自动兑现：在 nuScenes 的短 horizon imagination RL 里，object-only 仍然更稳；但在更 planning-oriented 的 nuPlan 20k 设定里，decoupled-no-vis 成为了稳定最强条件。所以这篇论文的核心结论不是‘我们 everywhere wins’，而是 relation-aware abstraction 对 policy learning 的价值取决于下游规划设定和 token 化特征。这是一篇方法与分析论文，而不是一个已经完全闭环验证完的大系统论文。” 

---

## 15. 我们现在最推荐的论文主张

如果要压缩成一句英文 thesis，目前最推荐的是：

**Typed-budget object-relational abstraction reliably improves decision-sufficient representation under a fixed latent budget, but its policy-learning benefit is downstream-regime dependent: object-only remains the most stable baseline on nuScenes short-horizon imagination, whereas decoupled no-visibility becomes the robust winner on planning-oriented nuPlan 20k.**

---

## 16. 相关文件索引

当前最重要的本地文件如下：

- 当前主稿：[doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:40)
- 当前 PDF：[doorrl_tiv_main.pdf](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.pdf)
- Stage 0 详细报告：[stage0.md](/Users/hb40355/Desktop/期刊/stage0.md:1)
- 全部实验总报告：[experiment_report.md](/Users/hb40355/Desktop/期刊/experiment_report.md:1)
- 本文档：[coauthor_report_doorrl_tiv_2026-04-25.md](/Users/hb40355/Desktop/期刊/coauthor_report_doorrl_tiv_2026-04-25.md:1)

---

## 17. 最后结论

如果只保留一句话给合作者，我会建议保留这句：

**我们现在最有价值的发现，不是“relation 一定更好”，而是“relation 能否帮助自动驾驶 policy learning，取决于有限表示容量是否按语义角色被正确分配，以及下游规划设定是否真的让这些关系变得可行动”。**

