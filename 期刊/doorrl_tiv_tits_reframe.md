# DOOR-RL 面向 T-IV / T-ITS 的顶刊化重写方案

## 1. 先说结论

你现在的想法有顶刊潜力，但当前版本更像一个“大型研究计划”，还不是一篇已经收口的期刊论文。  
如果目标是 `IEEE Transactions on Intelligent Vehicles (T-IV)` 和 `IEEE Transactions on Intelligent Transportation Systems (T-ITS)`，最关键的不是继续加模块，而是把论文压缩成一个非常清楚的核心命题：

**我们要证明的不是“我们搭了一个很大的系统”，而是：**

1. 日志回放式闭环训练会系统性伤害交互决策学习。
2. 对象-关系世界模型可以在保持交互性的同时，显著提高训练效率。
3. 在 latent imagination 中学到的策略，能够迁移到高保真闭环环境中，而不是“只在梦里有效”。

这三点足够撑起一篇强论文，也最符合你现有材料里最强的主线。

## 2. 官方 Scope 对位

截至 `2026-04-16`，我参考了 IEEE ITS Society 官方页面。

### T-IV 更适合什么

官方 scope 强调：

- intelligent vehicles
- automated vehicles
- vehicle environment perception
- vehicle control
- collision avoidance
- cooperative vehicle systems

这说明 `T-IV` 更偏“车本体”的智能决策、感知、控制和自动驾驶闭环能力。

### T-ITS 更适合什么

官方 scope 强调：

- modern transportation systems
- modeling and simulation
- experimentation and evaluation
- coordinated multiple vehicles
- other road users and their interactions
- data-based approaches, reinforcement learning, multi-agent systems

这说明 `T-ITS` 更偏“交通系统/多参与者/交互建模/仿真评估”的问题设定。

## 3. 你的题目更适合投哪本

### 第一推荐：T-IV

如果你的主叙事是：

- 学习自动驾驶策略
- 解决 ego policy 在伪闭环中的学习失真
- 用高保真闭环环境验证驾驶性能

那这更像 `T-IV`。因为中心问题仍然是自动驾驶车辆的决策与控制。

### 第二推荐：T-ITS

如果你的主叙事改成：

- 研究 reactive vs log-replay 对多智能体交互学习的影响
- 强调多交通参与者互动建模
- 把贡献写成一种 ITS-oriented simulation-and-learning framework

那它也可以投 `T-ITS`，但需要把“系统层交互建模与评估”讲得更重。

### 当前版本的建议

对你现在这套 DOOR-RL，我建议：

1. 主投 `T-IV`
2. 按 `T-IV` 口味写正文
3. 同时保留一份 `T-ITS` 版本摘要和导言重心，作为备选转投版本

## 4. 顶刊化之后的论文内核

### 一句话定位

**DOOR-RL learns reactive closed-loop driving policies in an object-relational world model and validates them in high-fidelity simulators, addressing the pseudo-interaction and render-in-the-loop bottlenecks of 3DGS-based driving RL.**

### 必须收掉的内容

下面这些内容可以存在，但不能全部抢“主贡献位”：

- DGGT / VGGT4D 4D 先验
- 不确定性集成
- 专家蒸馏
- 决策完备性抽象
- CARLA + 3DGS 双评估
- 跨数据集迁移

如果全都写成贡献，审稿人会看不清真正的 novelty。

### 建议保留的 3 个核心贡献

#### 贡献 1：问题定义与训练范式

指出日志回放式“闭环”并不构成真实 reactive interaction，并系统性分析其对 RL 学习的负面影响。

#### 贡献 2：方法主体

提出对象-关系 world model，在对象 token 与关系 token 的 latent imagination 中训练策略，替代 render-in-the-loop 训练。

#### 贡献 3：验证协议

提出并执行 “train in imagination, test in high-fidelity closed-loop simulators” 的验证协议，在 `CARLA + 3DGS` 中验证梦境到现实的迁移。

### 建议降级为“增强项”而非主贡献

- 4D 先验：增强特征，不是主创新
- 专家蒸馏：稳定训练的课程学习技巧
- 不确定性：附加扩展，不是第一版必须项

## 5. 题目建议

### T-IV 版本

1. **DOOR-RL: Reactive Object-Relational World Models for Sample-Efficient Closed-Loop Autonomous Driving**
2. **Learning Reactive Driving Policies in Object-Relational World Models with High-Fidelity Closed-Loop Evaluation**
3. **Beyond Log-Replay Closed Loops: Object-Relational World Models for Autonomous Driving Reinforcement Learning**

### T-ITS 版本

1. **Reactive Closed-Loop Driving via Object-Relational World Models: Learning in Imagination, Evaluation in High-Fidelity Traffic Simulators**
2. **From Pseudo-Interaction to Reactive Learning: Object-Relational World Models for Closed-Loop Driving**
3. **Object-Relational World Models for Interactive Driving Policy Learning and High-Fidelity Transfer Evaluation**

## 6. 推荐摘要

### 6.1 T-IV 版本摘要

High-fidelity simulators are increasingly used for autonomous driving reinforcement learning, yet many recent three-dimensional reconstruction based closed-loop frameworks still rely on log-replay traffic agents and render-in-the-loop training. This combination creates two key limitations: other road users do not react to the ego vehicle, and per-step rendering severely limits training throughput. We present DOOR-RL, a reactive driving framework that learns policies in an object-relational world model instead of optimizing directly in a rendering loop. DOOR-RL encodes each scene into ego, agent, map, and interaction tokens, learns latent dynamics for multi-agent traffic evolution, and trains an actor-critic policy through imagined rollouts. To avoid evaluating the policy only in its own learned latent space, we adopt a train-in-imagination, test-in-high-fidelity protocol and assess transfer in CARLA and a closed-loop simulator based on three-dimensional Gaussian Splatting. Experiments are designed to test three claims: reactive training outperforms log-replay training in social driving behaviors, object-relational modeling preserves decision-critical interaction structure better than holistic latent baselines, and imagination-trained policies retain performance in high-fidelity closed-loop evaluation. The results position DOOR-RL as a scalable alternative to render-in-the-loop driving reinforcement learning for automated vehicles.

### 6.2 T-ITS 版本摘要

Closed-loop learning for autonomous driving increasingly depends on simulation, but many existing high-fidelity training environments still use non-reactive log replay for surrounding traffic and therefore fail to model real multi-agent interaction. This mismatch can bias reinforcement learning toward overly conservative or brittle behaviors while also incurring substantial computational cost when rendering is kept inside the training loop. This paper proposes DOOR-RL, an object-relational world-model framework for reactive closed-loop driving. DOOR-RL represents traffic scenes as ego, dynamic-agent, map, and relation tokens, learns latent transition dynamics from interactive rollouts, and optimizes the driving policy through imagined trajectories rather than expensive image rendering. To address the evaluation paradox of world-model based learning, the policy is trained in latent imagination but tested in high-fidelity closed-loop simulators, including CARLA and an environment based on three-dimensional Gaussian Splatting reconstructed from driving data. The study evaluates whether reactive training improves interaction quality over log-replay baselines, whether object-relational abstractions outperform holistic latent representations, and whether latent-world training transfers to realistic traffic simulators. The proposed framework targets a practical bridge between scalable learning and realistic intelligent transportation system evaluation.

## 7. 建议导言结构

### 第一段：背景

高保真重建环境为自动驾驶 RL 提供了更接近真实传感器分布的测试平台，但并没有自动解决交互真实性与训练效率问题。

### 第二段：核心缺陷

现有 3DGS-based driving RL 往往同时存在两类问题：

1. 他车由 log replay 驱动，缺乏 reactive interaction。
2. 渲染在环导致每步训练成本过高。

### 第三段：现有 world model 也不够

整体式 latent world model 虽然高效，但容易压缩掉决策关键的对象关系结构，因此不能直接解决驾驶交互学习。

### 第四段：本文方法

本文提出 DOOR-RL，在对象-关系 world model 中进行 latent imagination policy learning，并用高保真闭环环境检验迁移效果。

### 第五段：贡献

建议只写三条，且每条都能被实验直接验证：

1. 证明伪交互训练会显著削弱多智能体驾驶决策学习。
2. 提出对象-关系 latent world model，用于 reactive closed-loop driving RL。
3. 提出梦境训练、高保真测试的验证协议，并在 CARLA 与 3DGS 环境中验证。

## 8. 论文里必须说清楚的设定

这是你现在最容易被审稿人抓住的地方。

### 8.1 不要写成“端到端”

如果输入是 `GT objects + map tokens + relation tokens`，那这不是严格意义上的 end-to-end autonomous driving。  
更准确的说法应该是：

- structured-perception driving policy learning
- decision-making under object-centric state abstraction
- closed-loop policy learning with structured scene tokens

### 8.2 必须定义测试时 token 从哪里来

这是全稿最关键的问题之一。

你必须明确说明下列两种设定中的哪一种：

1. **Oracle-token evaluation**
   在 CARLA 和 3DGS 测试时，使用 simulator / annotation 提供的对象状态生成 token。  
   这时论文研究的是“决策学习”，不是“感知到控制的端到端学习”。

2. **Online perception-token evaluation**
   在测试时通过感知模块从图像中提取 token。  
   这时论文更完整，但工程难度会暴涨。

如果你的目标是尽快形成顶刊稿，我建议第一版明确采用：

**Oracle-token evaluation for decision-learning validation.**

然后把限制写清楚，而不是模糊带过。

## 9. 最小可发表版本

### 必做项

1. 用 `GT object/map tokens` 建立统一 schema。
2. 实现对象-关系 world model。
3. 在 `SMARTS` 或等价 reactive 环境中训练。
4. 在 `CARLA` 做闭环测试。
5. 加一个高保真外部验证环境，优先 `3DGS`。
6. 做清楚的四类对比：
   - BC teacher
   - model-free RL
   - holistic world model baseline
   - full DOOR-RL

### 非必须但加分项

1. DGGT / VGGT4D 先验
2. uncertainty ensemble
3. 反事实压力测试
4. 跨数据集泛化

### 第一版建议删除的内容

如果算力或时间不够，先删：

1. Waymo + nuScenes 双数据集同时做
2. DGGT 和 VGGT4D 同时做
3. 不确定性集成
4. 过多的可解释性可视化

## 10. 实验主表应该怎么搭

### 主表 1：闭环驾驶性能

列建议：

- collision rate
- route completion
- success rate
- off-road rate
- traffic violation score
- comfort

行建议：

- BC teacher
- model-free RL
- DreamerV3-style holistic WM
- DOOR-RL w/o relations
- DOOR-RL full

### 主表 2：交互真实性影响

目标是直接打中你的主命题。

列建议：

- reactive-train + reactive-test
- replay-train + reactive-test
- replay-train + replay-test
- merge negotiation success
- yielding compliance
- near-collision rate

这张表必须让审稿人一眼看出：

**不是高保真渲染本身在帮你，而是 reactive learning 在帮你。**

### 主表 3：训练效率

列建议：

- env steps per second
- wall-clock to threshold
- GPU hours
- training frames

这张表用来证明你确实解决了 render-in-the-loop 的算力问题。

### 主表 4：梦境到现实迁移

列建议：

- latent validation score
- CARLA score
- 3DGS score
- transfer gap

这张表用来防止审稿人说：你的 world model 只是“在自己学到的环境里赢”。

## 11. 最关键的消融实验

只保留最有杀伤力的 5 个：

1. `reactive training` vs `log-replay training`
2. `object-relational WM` vs `holistic WM`
3. `with relations` vs `without relations`
4. `with teacher warm start` vs `without teacher warm start`
5. `latent-only evaluation` vs `high-fidelity transfer evaluation`

如果篇幅紧张，4D 先验和不确定性都可以放附录或后续版本。

## 12. 审稿人最可能问的 8 个问题

### Q1. 你是不是只是把很多已有模块拼起来？

回答策略：强调主创新不在模块堆叠，而在于“reactive object-relational learning + high-fidelity transfer evaluation”的统一问题定义和验证闭环。

### Q2. 如果测试时用 oracle token，这和真实自动驾驶有多大关系？

回答策略：明确本文聚焦决策学习，而非感知误差建模；高保真环境用于验证闭环交互与控制，而不是做完整感知链条对比。

### Q3. 提升来自对象关系建模，还是来自教师蒸馏？

回答策略：必须有 `without teacher` 消融。

### Q4. 提升来自 reactive data，还是来自 world model？

回答策略：必须把 `reactive vs replay` 和 `world model vs model-free` 两条轴拆开做正交实验。

### Q5. 为什么一定要 3DGS 评测？

回答策略：CARLA 给可控性，3DGS 给更接近真实日志的视觉闭环特性；两者互补。

### Q6. 你的方法是不是只适合 token-level 输入？

回答策略：是，第一版就承认这一点，并把它定位成“decision-learning paper”。

### Q7. world model 误差累积怎么办？

回答策略：给多步预测结果、短视程到长视程课程学习、真实 rollout refresh。

### Q8. 为什么不是直接在 CARLA 里做 RL？

回答策略：CARLA 交互真实但训练吞吐低；本文目标是用 latent imagination 提升效率，并保留高保真测试。

## 13. 推荐论文结构

1. Introduction
2. Related Work
3. Problem Formulation
4. DOOR-RL
5. Experimental Setup
6. Main Results
7. Ablations and Analysis
8. Limitations
9. Conclusion

## 14. 建议你在正文里主动承认的限制

主动写限制，反而更像成熟顶刊稿。

### 限制 1

本文主要研究结构化对象表示下的决策学习，不覆盖端到端视觉感知误差。

### 限制 2

高保真 3DGS 环境主要承担外部验证角色，而不是训练主环境。

### 限制 3

方法目前面向局部交互决策，不直接优化全局交通网络效率指标。

## 15. 我对你这篇稿子的最终建议

### 如果你要冲 T-IV

请把主标题、导言、结果讨论全部围绕：

- autonomous driving
- closed-loop policy learning
- vehicle interaction
- high-fidelity evaluation

### 如果你要冲 T-ITS

请把主标题、导言、结果讨论更多围绕：

- multi-agent interaction
- transportation-system simulation
- coordinated road-user behavior
- modeling, simulation, and evaluation

### 实操上怎么选

如果第一版实验主要是：

- ego vehicle policy
- CARLA leaderboard
- 3DGS closed-loop scenes
- safety / completion / comfort 指标

那我建议先写成 `T-IV` 版本。

如果后续你补强了：

- multi-agent interaction metrics
- reactive vs replay system analysis
- 交通参与者协同与交互建模

那 `T-ITS` 会更强。

## 16. 可直接放进论文里的三条贡献表述

你可以直接拿下面这版改。

1. **We identify pseudo-interaction as a core limitation of recent high-fidelity driving RL frameworks and show why log-replay closed loops are insufficient for learning socially compliant driving policies.**
2. **We propose DOOR-RL, an object-relational world-model framework that learns reactive driving policies through latent imagination rather than render-in-the-loop optimization.**
3. **We establish a train-in-imagination, test-in-high-fidelity evaluation protocol and demonstrate transfer in CARLA and a 3DGS-based closed-loop simulator.**

## 17. 投稿前的硬性检查清单

在你真正准备投之前，至少满足下面这些条件：

1. 有一张表直接证明 `reactive > replay`
2. 有一张表直接证明 `object-relational > holistic`
3. 有一张表直接证明 `imagination training is faster`
4. 有一张表直接证明 `latent success transfers to high-fidelity simulators`
5. 明确写清楚测试时 token 来源
6. 不把 4D 先验写成主要 novelty
7. 不把论文写成“大系统说明书”

## 18. 参考的官方来源

- T-IV official page: https://ieee-itss.org/pub/t-iv/
- T-IV author information: https://ieee-itss.org/pub/t-iv/author/
- T-ITS official page: https://ieee-itss.org/pub/t-its/

这些官方页面显示：

- `T-IV` 的 scope 明显更偏智能车辆与自动驾驶本体问题
- `T-ITS` 的 scope 更偏智能交通系统、建模仿真、实验评估和多参与者交互
- 两本期刊的 regular paper 建议长度都是 `10 pages`
- 摘要都要求大致 `150-250 words`

---

如果继续推进，下一步最值得做的不是再扩想法，而是把这份重写稿继续落成：

1. `摘要 + Introduction + Related Work` 初稿
2. `实验主表模板`
3. `LaTeX 论文骨架`
4. `给 T-IV 的第一版投稿文案`
