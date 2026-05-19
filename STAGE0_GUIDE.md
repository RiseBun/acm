# Stage 0: Table 3 表示充分性消融 - 执行指南

## 🎯 目标

**证明 object-relational representation matters**

这是整个论文的基础。如果这一步看不到明显收益，后续做RL也很危险。

---

## 📊 输出：论文Table 3

```
Table 3: Representation Sufficiency Ablation

Variant                 | Rollout Error ↓    | Reward Error ↓     | Collision Pred. ↑ | Rare-Agent Recall ↑
------------------------|--------------------|--------------------|--------------------|---------------------
Holistic                |                    |                    |                    |
Object-only             |                    |                    |                    |
Object + Relation       |                    |                    |                    |
Obj+Rel+Visibility      |                    |                    |                    |
```

**假设**:
- Object + Relation 应该显著优于 Holistic 和 Object-only
- 如果看不到差异 → 重新思考表示设计

---

## 🚀 执行步骤

### 方式1: 运行所有变体（推荐）

```bash
cd /mnt/volumes/cpfs-ares-root/prediction/lipeinan/code

# 一键运行所有4个变体
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant all \
    --num-scenes 20 \
    --epochs 30 \
    --batch-size 8
```

### 方式2: 单独运行每个变体

```bash
# 1. Holistic baseline
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant holistic \
    --num-scenes 20 \
    --epochs 30

# 2. Object-only
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_only \
    --num-scenes 20 \
    --epochs 30

# 3. Object + Relation (核心)
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation \
    --num-scenes 20 \
    --epochs 30

# 4. Object + Relation + Visibility
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation_visibility \
    --num-scenes 20 \
    --epochs 30
```

---

## 📁 输出文件

```
experiments/table3_representation_sufficiency/
├── holistic/
│   ├── model.pt              # 训练好的模型
│   └── table3_results.json   # 评估指标
├── object_only/
│   ├── model.pt
│   └── table3_results.json
├── object_relation/
│   ├── model.pt
│   └── table3_results.json
├── object_relation_visibility/
│   ├── model.pt
│   └── table3_results.json
└── table3_complete.json      # 完整结果 (用于生成表格)
```

---

## 📈 结果分析

### 自动生成的LaTeX表格

脚本会自动输出LaTeX代码，可以直接复制到论文中：

```latex
\begin{table}[t]
\centering
\caption{Representation Sufficiency Ablation Study}
\label{tab:representation_sufficiency}
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Rollout Error} $\downarrow$ & ... \\
\midrule
Holistic & ... \\
Object-only & ... \\
Object + Relation & ... \\
Obj+Rel+Visibility & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### Go/No-Go 决策点

**成功标准**:
- ✅ Object + Relation 的 Rollout Error 比 Holistic 低至少10%
- ✅ Object + Relation 的 Collision Accuracy 比 Object-only 高至少5%
- ✅ Rare-Agent Recall 有明显提升

**失败处理**:
- ❌ 如果所有变体差不多 → 关系特征设计有问题
- ❌ 如果 Holistic 最好 → 对象分离可能不必要
- ❌ 如果 Object-only 最好 → 关系特征可能有噪声

---

## ⏱️ 预计时间

| 配置 | GPU | 时间 |
|------|-----|------|
| 20场景, 30 epochs | 单GPU (RTX 3090) | ~2小时/变体 |
| 20场景, 30 epochs | 单GPU (A100) | ~1小时/变体 |
| 50场景, 50 epochs | 单GPU | ~8小时/变体 |

**建议**:
- 先用 5场景, 10 epochs 验证代码 (30分钟)
- 然后跑正式实验

---

## 🔧 调试技巧

### 1. 快速验证代码

```bash
# 最小实验
python3 run_stage0_table3.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant holistic \
    --num-scenes 5 \
    --epochs 2 \
    --batch-size 4
```

### 2. 只评估不训练

```bash
# 如果已经有训练好的模型
python3 run_stage0_table3.py \
    --variant object_relation \
    --evaluate-only
```

### 3. 检查结果

```bash
# 查看JSON结果
cat experiments/table3_representation_sufficiency/object_relation/table3_results.json

# 查看完整对比
cat experiments/table3_representation_sufficiency/table3_complete.json
```

---

## 📝 论文写作提示

### 如何解释Table 3

**如果成功**:
> "As shown in Table 3, our object-relational representation significantly outperforms both holistic and object-only baselines across all metrics. The 15% reduction in rollout error suggests that explicit relation modeling enables more accurate world model predictions. The improved collision prediction accuracy (+8%) demonstrates that relation tokens capture critical safety-relevant information."

**如果失败**:
> 需要重新设计关系特征，或考虑其他表示方式

### 图表建议

- **Figure 1**: 训练曲线对比 (4条线)
- **Table 3**: 消融结果 (自动生成)
- **Figure 2**: Token attention可视化 (可选)

---

## ✅ 完成后

### 如果Stage 0成功 → 继续Stage 1

```bash
# Stage 1: Multi-step latent rollout
# (需要先实现 multi-step prediction)
```

### 如果Stage 0失败 → 重新思考

可能的原因:
1. 关系特征设计不当 (TTC, risk等不够好)
2. Token化策略有问题
3. 模型容量不足/过大
4. 训练不充分

---

## 🎯 关键提醒

**这是论文的基础实验！**

- 不要跳过Stage 0直接做RL
- 如果Table 3没有明显差异，后续实验风险很大
- 先证明表示有价值，再证明RL有效

**下一步行动**:
1. 运行小实验验证代码 (30分钟)
2. 运行正式实验 (2-8小时)
3. 分析Table 3结果
4. 决定是否继续Stage 1

---

**祝实验顺利！**
