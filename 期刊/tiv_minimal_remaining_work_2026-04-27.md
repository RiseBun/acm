# DOOR-RL 投 IEEE T-IV 的最小剩余工作清单

更新时间：2026-04-27

这份清单的目的不是把所有可能工作都列出来，而是回答一个更务实的问题：

**基于我们现在已经做完的实验，离形成一篇可投的 T-IV 稿件，最少还差什么？**

我把建议分成两条路线：

- `Route A`：保守投稿路径  
  目标是尽快形成一篇“方法 + 分析”导向、证据闭环已经基本成立的投稿版本。
- `Route B`：增强投稿路径  
  目标是在 `Route A` 基础上，再补一层更强的下游或闭环证据，但代价更高、风险也更高。

---

## 1. 当前状态快照

从证据链上看，我们已经具备这些核心结果：

### 已经足够作为主结果的部分

- `Stage 0 / nuScenes / 3 seeds`  
  明确证明 `typed-budget decoupling` 提升表示充分性，并解决 naive relation mixing 的 slot competition。
- `Stage 1 / nuScenes / 3 seeds`  
  明确证明“更好的表示不自动意味着更好的 policy learning”，object-only 在当前设定下更稳。
- `Stage 1 / nuPlan 20k / 3 seeds`  
  明确显示 ranking reversal，`wm_decoupled_no_vis` 成为最强条件。
- `Stage 1 / nuPlan 50k / 3 seeds`  
  对 20k 的结论进行了 scale-up 确认，说明趋势不是小样本偶然。
- `nuPlan 50k offline planner-like sanity check`  
  进一步说明 `wm_decoupled_no_vis` 的优势不只体现在 latent reward 上，也体现在更接近 planner-like 行为的 offline probe 上。
- `dataset statistics`  
  为 `nuScenes` 与 `nuPlan` 的 ranking reversal 提供了解释性支持。

### 已经完成但目前不适合当主结果的部分

- `official nuPlan closed-loop MVP / sanity / wrapper development`

这些结果很有价值，但当前更像：

- 链路已经跑通
- wrapper 已经从“完全不能用”走到“可以稳定 sanity”
- 但 wrapper 质量仍显著影响最终指标

所以它们目前更适合写成：

- discussion / limitation / supplementary evidence

而不适合作为论文主结果。

---

## 2. 我们现在已经“够写”的论文是什么

如果今天就冻结主线，这篇最适合写成：

**一篇关于 typed-budget object-relational abstraction 的方法与分析论文。**

更准确说，是：

**在固定 latent budget 下，relation-aware abstraction 何时有效、为何有效，以及何时它的表示收益能够兑现为 policy-learning 收益。**

这篇论文当前最稳的 claim 是：

1. `inter-type budget competition` 是真实存在的 failure mode。
2. `typed-budget decoupling` 在表示层上稳定有效。
3. 这种表示收益对 policy learning 的兑现是 `dataset / planning-regime dependent`。
4. `nuPlan 20k/50k + planner-like sanity` 已经足够支持 “planning-oriented downstream regime favors decoupled-no-vis”。

---

## 3. Route A：保守投稿路径

这条路径的目标是：

**不再扩大论文主张，只把现有结果整理成一篇结构完整、表述克制、可提交的 T-IV regular paper。**

### 3.1 必须完成

#### 1. 方法总览图

这是当前最缺的一项。

必须补一张图，至少包含：

- 完整 97-token scene decomposition
- naive shared top-`K` bottleneck
- typed-budget dyn/rel decoupling
- shared world model
- latent imagination actor-critic loop

原因：

- 当前文字已经够，但没有这张图，审稿人第一遍阅读会更费力。

主稿里我已经加了图位：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:128)

#### 2. 4 组 case study 可视化

最值得补的不是继续扫新条件，而是把已经有的结论“看得见”。

建议至少做 4 组：

1. `naive object+relation` 漏掉关键 dynamic agent  
   证明 shared top-`K` 的 slot competition
2. `decoupled` 在 Stage 0 恢复关键 dynamic context  
   证明 typed-budget 的表示层价值
3. `nuScenes Stage 1` 里 object-only 更稳的代表例子  
   证明“表示提升不自动兑现为策略提升”
4. `nuPlan 50k` 里 `wm_decoupled_no_vis` 更优的代表例子  
   证明 planning-oriented regime 下 decoupled 的优势兑现

主稿里我已经加了 cross-dataset 图位：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:324)

#### 3. 替换占位参考文献

当前至少还缺两条正式 citation：

- `rad_todo`
- `recondreamer_todo`

位置在：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:375)

这一步必须完成，否则不能正式投稿。

#### 4. 填作者与投稿元信息

包括：

- 作者姓名
- 单位
- 通讯作者邮箱
- short bio

位置在：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:31)
- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:383)

#### 5. 做一次全文术语统一

必须统一这些表述：

- `object-only` / `WM-Object`
- `decoupled` / `WM-Decoupled`
- `decoupled-no-vis` / `WM-Decoupled-NoVis`
- `planner-like sanity`
- `oracle-token evaluation`
- `representation sufficiency`
- `planning-oriented`

现在主稿已经基本统一了，但投稿前还需要通读一遍。

#### 6. 把 supplementary / appendix 材料整理出来

不是说必须所有东西都放在主文，而是要把附件结构准备好。

我已经在主稿里放了 appendix skeleton：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:341)

建议 supplement 至少放：

- Stage 1 per-seed full tables
- dataset statistics full tables
- closed-loop sanity wrapper evolution简表

### 3.2 强烈建议完成

#### 7. 再压一轮摘要和引言

当前主稿已经是正确主线，但投稿版还可以再“硬一点”：

- 少一点 narrative
- 多一点 thesis clarity
- 更直接地告诉审稿人：  
  “这不是一个 final closed-loop benchmark paper，而是一篇 representation-to-policy interface paper”

#### 8. 把 closed-loop 描述降到合适位置

当前最好的做法不是删除 closed-loop，而是：

- 在正文 discussion 里简要说“official loop 已接通，但当前 wrapper quality dominates”
- 在 supplement 中完整说明各个 wrapper 版本

这样既保住工作量，也不让主 claim 被它拖累。

### 3.3 Route A 下不建议再做

如果走保守投稿路径，以下工作**不建议继续投入**：

- 再扫更多 budget 组合
- 再扫更多 actor/critic fusion 变体
- 再做大规模 closed-loop wrapper engineering，但没有明确 stop rule
- 再扩展很多新 benchmark

原因很简单：

- 这些工作很容易消耗大量时间
- 但未必提升主论文的可信度
- 反而可能继续发散主线

---

## 4. Route B：增强投稿路径

这条路径的目标是：

**在 Route A 基础上，再补一层更强的“更像车辆规划论文”的证据。**

### 4.1 最值得做的增强项

#### 1. 一个可站住脚的 official nuPlan closed-loop 小规模主表

注意这里说的是：

- **可站住脚**
- **小规模也可以**

不要求一上来就 100-scenario。

更合理的目标是：

- 固定一个你愿意公开承认的 wrapper
- 固定一组 sanity scenarios
- 给出 `wm_object` vs `wm_decoupled_no_vis` 的 official metrics

但前提必须是：

- wrapper 不再频繁切换
- 结论不是完全由 wrapper heuristic 决定

如果做不到这一点，就不要把 closed-loop 拉进主结果。

#### 2. 更正式的 downstream planner behavior figure

例如画出：

- teacher action
- `wm_object` policy action
- `wm_decoupled_no_vis` policy action
- imagined collision proxy

这类图比再多一张表更容易打动 reviewer。

### 4.2 Route B 的风险

最大的风险是：

- 你会花很多时间在 planner wrapper 上
- 最后得到的不是“方法更强”
- 而是“某个 wrapper 工程更强”

如果出现这种趋势，应该立刻止损，把 closed-loop 留在 supplement。

---

## 5. 最小剩余工作结论

如果只问：

**为了形成一篇可投版本，最低限度还差什么？**

答案是这 6 件：

1. 方法总览图
2. 4 组 case-study 可视化
3. 替换占位参考文献
4. 填作者/单位/bio
5. 统一术语并收紧摘要引言
6. 整理 supplement / appendix

如果这 6 件完成了，**即使不再新增任何主实验**，这篇论文也已经有希望作为一篇“方法 + 分析”导向的 T-IV 稿件提交。

---

## 6. 如果还有额外时间，最值得加的一件事

如果你还有额外时间，只建议再做一件事：

**固定一个 closed-loop wrapper，然后做一个小规模、可复现、可解释的 official nuPlan closed-loop sanity table。**

但这件事必须满足两个条件：

- 结论稳定
- wrapper 不再是 moving target

如果不满足，就不要硬塞进主文。

---

## 7. 当前已经填进主稿的表

为了方便协作，这里列一下目前已经正式落进论文的主表：

- `Stage 0 / nuScenes representation table`
- `Stage 1 / cross-dataset imagination table`
  - `nuScenes`
  - `nuPlan 20k`
  - `nuPlan 50k`
- `nuPlan 50k offline planner-like sanity table`

对应位置：

- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:230)
- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:271)
- [doorrl_tiv_main.tex](/Users/hb40355/Desktop/期刊/IEEE-Transactions-LaTeX2e-templates-and-instructions/doorrl_tiv_main.tex:307)

---

## 8. 一句话建议

**现在最正确的动作不是继续把实验空间无限扩张，而是把已经很强的证据链收成一篇聚焦、克制、可解释的论文。**

