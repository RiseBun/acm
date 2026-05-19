# DOOR-RL 论文实验使用指南

## 📋 更新内容

本项目已经从MVP框架升级为**可用于论文实验的完整代码**，包含：

✅ **真实数据Pipeline**
- nuScenes真实action/reward提取
- 地图元素tokenization
- 序列采样支持

✅ **消融实验系统**
- 4种模型变体 (holistic/object_only/object_relation/object_relation_visibility)
- 一键运行所有实验
- 自动结果分析和可视化

✅ **评估系统**
- 世界模型预测精度评估
- 策略性能评估
- 训练/验证集自动划分

---

## 🚀 快速开始

### 1. 测试环境

```bash
cd /mnt/volumes/cpfs-ares-root/prediction/lipeinan/code
conda activate find_physics_zone

# 运行基础测试
python3 -m pytest tests/ -v
```

### 2. 合成数据调试 (验证代码)

```bash
# 快速测试 (2 epochs)
python3 train_debug.py \
    --config configs/debug_mvp.json \
    --epochs 2
```

### 3. 真实nuScenes数据训练

```bash
# 小规模测试 (5个场景, 10 epochs)
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 5 \
    --epochs 10

# 完整训练 (20个场景, 30 epochs)
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 20 \
    --epochs 30 \
    --batch-size 16
```

---

## 🔬 消融实验 (论文核心)

### 运行单个变体

```bash
# Holistic baseline
python3 train_ablation.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant holistic \
    --num-scenes 20 \
    --epochs 30

# Object-only
python3 train_ablation.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_only \
    --num-scenes 20 \
    --epochs 30

# Object-relation (DOOR-RL核心)
python3 train_ablation.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation \
    --num-scenes 20 \
    --epochs 30

# Object-relation + visibility
python3 train_ablation.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation_visibility \
    --num-scenes 20 \
    --epochs 30
```

### 一键运行所有实验

```bash
# 给脚本添加执行权限
chmod +x run_ablation_study.sh

# 运行所有变体 (自动顺序执行)
./run_ablation_study.sh
```

实验结果会保存到 `experiments/ablation/` 目录：
```
experiments/ablation/
├── holistic_20260420_123456/
│   ├── config.json
│   ├── history.json
│   └── model.pt
├── object_only_20260420_134567/
│   ├── config.json
│   ├── history.json
│   └── model.pt
├── object_relation_20260420_145678/
│   └── ...
└── object_relation_visibility_20260420_156789/
    └── ...
```

### 分析结果

```bash
# 打印对比表格
python3 analyze_ablation.py \
    --exp-dir experiments/ablation \
    --table

# 绘制训练曲线
python3 analyze_ablation.py \
    --exp-dir experiments/ablation \
    --plot \
    --save-plot ablation_results.png

# 生成LaTeX表格 (用于论文)
python3 analyze_ablation.py \
    --exp-dir experiments/ablation \
    --latex

# 所有功能
python3 analyze_ablation.py \
    --exp-dir experiments/ablation \
    --table --plot --latex \
    --save-plot paper_figure.png
```

---

## 📊 模型变体说明

### 1. Holistic (Baseline)
- **表示方式**: 所有token全局池化，不区分对象/关系
- **目的**: 证明对象分离表示的必要性
- **预期结果**: 性能最差

### 2. Object-Only
- **表示方式**: 只使用对象token，过滤掉关系token
- **目的**: 证明关系token的贡献
- **预期结果**: 比holistic好，比object_relation差

### 3. Object-Relation (DOOR-RL)
- **表示方式**: 对象token + 关系token (TTC, risk, lane_conflict等)
- **目的**: 论文核心贡献
- **预期结果**: 性能最好

### 4. Object-Relation + Visibility
- **表示方式**: 在object-relation基础上添加可见性加权
- **目的**: 证明可见性先验的额外价值
- **预期结果**: 可能略好于object_relation

---

## 🎯 论文章节对应

### Table 1: 消融实验结果
```bash
python3 analyze_ablation.py --exp-dir experiments/ablation --latex
```

### Figure 1: 训练曲线对比
```bash
python3 analyze_ablation.py --exp-dir experiments/ablation --plot --save-plot training_curves.png
```

### Section 4.2: 实验设置
参考 `experiments/ablation/*/config.json`

### Section 4.3: 结果分析
参考 `experiments/ablation/*/history.json`

---

## 💡 实验建议

### 实验规模

| 阶段 | 场景数 | Epochs | 目的 |
|------|--------|--------|------|
| 代码验证 | 5 | 10 | 确保代码正常运行 |
| 初步结果 | 10 | 20 | 验证假设 |
| 论文数据 | 20-50 | 30-50 | 正式实验 |

### 超参数调优

```bash
# 更大的模型
python3 train_ablation.py \
    --config configs/experiment_safe.json \
    --variant object_relation \
    --num-scenes 50 \
    --epochs 50 \
    --batch-size 16

# 修改configs/experiment_safe.json中的:
# - model_dim: 128 -> 256
# - num_layers: 2 -> 4
# - learning_rate: 0.001 -> 0.0005
```

### 多次运行取平均

```bash
# 运行3次不同种子
for SEED in 7 42 123; do
    python3 train_ablation.py \
        --variant object_relation \
        --num-scenes 20 \
        --epochs 30 \
        --seed $SEED \
        --output-dir experiments/ablation_seed${SEED}
done
```

---

## 📈 预期结果

### 训练损失趋势
```
Epoch 1:  Total Loss ~ 2.5
Epoch 10: Total Loss ~ 1.2
Epoch 30: Total Loss ~ 0.8
```

### 变体性能排序
```
object_relation < object_relation_visibility < object_only < holistic
(损失越低越好)
```

### 关键指标
- Observation MSE: 0.5-1.0
- Reward MSE: 0.2-0.5
- Action MSE (BC): 0.1-0.3
- Collision Accuracy: > 85%

---

## 🔧 常见问题

### Q1: 训练出现NaN
```bash
# 降低学习率
python3 train_ablation.py --variant object_relation --learning-rate 0.0001

# 增加梯度裁剪
# (已在trainer.py中实现，max_norm=1.0)
```

### Q2: 数据加载慢
```bash
# 增加workers
# 修改train_ablation.py中的 num_workers=4 -> 8

# 使用pin_memory
# (已启用)
```

### Q3: 内存不足
```bash
# 减小batch size
python3 train_ablation.py --batch-size 4

# 减小模型
# 修改config: model_dim=128 -> 64, num_layers=2 -> 1
```

### Q4: CAN总线数据不可用
代码会自动fallback到从位姿差异计算action，不影响训练。

---

## 📝 下一步工作

### 已完成 ✅
- [x] nuScenes真实数据Pipeline
- [x] 消融实验系统
- [x] 评估指标系统
- [x] 训练/验证集划分

### 待实现 (可选)
- [ ] nuPlan完整Adapter (用于闭环评估)
- [ ] NAVSIM迁移实验
- [ ] 反应式训练循环
- [ ] 3DGS集成

---

## 📚 相关文档

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) - 完整项目说明
- [TOKENIZATION_SPEC.md](docs/TOKENIZATION_SPEC.md) - Token化规范
- [REAL_DATA_PIPELINE.md](docs/REAL_DATA_PIPELINE.md) - 真实数据使用指南

---

## 🎓 论文写作提示

### 贡献点强调
1. **对象关系表示**: 显式建模TTC、碰撞风险、车道冲突等关系特征
2. **决策充分抽象**: 通过注意力机制选择top-k关键token
3. **系统性消融**: 4种表示方式的公平对比

### 实验设计
- **Fair comparison**: 所有变体使用相同数据、超参数、训练轮数
- **Multiple seeds**: 建议至少3个随机种子
- **Statistical significance**: 报告mean ± std

### 可视化建议
- Figure 1: 训练曲线 (4条线对比)
- Figure 2: Token attention可视化
- Figure 3: 案例分析 (成功/失败场景)
- Table 1: 消融实验结果

---

**祝实验顺利！如有问题，查看代码注释或联系开发团队。**
