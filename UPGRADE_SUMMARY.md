# DOOR-RL 项目升级总结

## 📊 从MVP到论文代码的改造

### 改造前 (MVP框架)
- ❌ 只有合成数据训练
- ❌ 真实数据Adapter有大量TODO
- ❌ action/reward都是硬编码
- ❌ 没有消融实验支持
- ❌ 没有评估系统
- ❌ 训练/验证集不分

### 改造后 (论文代码)
- ✅ 完整的nuScenes真实数据Pipeline
- ✅ 真实action提取 (CAN总线 + 位姿计算)
- ✅ 复合reward函数 (安全+舒适+效率+规则)
- ✅ 4种模型变体支持消融实验
- ✅ 完整的评估指标系统
- ✅ 自动训练/验证集划分
- ✅ 结果分析和可视化工具

---

## 🔧 新增/修改的文件

### 1. 核心功能文件 (新增)

| 文件 | 功能 | 行数 |
|------|------|------|
| `src/doorrl/adapters/nuscenes_action_extractor.py` | 真实action/reward提取 | 326 |
| `src/doorrl/models/doorrl_variant.py` | 4种消融实验变体 | 265 |
| `src/doorrl/evaluation/metrics.py` | 评估指标系统 | 214 |
| `src/doorrl/evaluation/__init__.py` | 评估模块导出 | 14 |
| `train_ablation.py` | 消融实验训练脚本 | 247 |
| `analyze_ablation.py` | 结果分析工具 | 257 |
| `run_ablation_study.sh` | 批量实验脚本 | 62 |
| `PAPER_EXPERIMENT_GUIDE.md` | 论文实验指南 | 344 |

**总计新增**: ~1,729行代码和文档

### 2. 核心功能文件 (修改)

| 文件 | 修改内容 |
|------|---------|
| `src/doorrl/adapters/nuscenes_real_adapter.py` | +128行: 集成action extractor, 添加地图元素, 实现优先级计算 |
| `src/doorrl/data/real_dataset.py` | +23行: 支持序列采样, 正确传递next_sample |
| `train_real_nuscenes.py` | +16行: 添加训练/验证集划分 |
| `src/doorrl/models/__init__.py` | +17行: 导出新模型变体 |

---

## 🎯 关键功能实现

### 1. 真实Action提取

**策略**:
```python
# 优先使用CAN总线
action = action_extractor.extract_action_from_can(scene_name, sample)

# 备选：从位姿差异计算
if action is None:
    action = action_extractor.extract_action_from_pose(sample, next_sample)
```

**输出**:
- `action[0]`: 纵向加速度 (m/s²), 范围 [-4, 4]
- `action[1]`: 转向角 (rad), 范围 [-0.5, 0.5]

### 2. 复合Reward函数

**设计**:
```python
total_reward = (
    0.4 * safety_reward +        # 安全性 (TTC + 距离)
    0.3 * comfort_reward +       # 舒适性 (动作平滑)
    0.2 * efficiency_reward +    # 效率 (速度跟踪)
    0.1 * rule_reward            # 规则 (车道保持)
)
```

**安全性细节**:
- TTC < 2s: -10.0 (严重惩罚)
- TTC < 5s: -2.0 (中等惩罚)
- 距离 < 5m: -5.0
- 距离 < 10m: -1.0

### 3. 消融实验变体

**4种表示方式**:

```python
class ModelVariant(Enum):
    HOLISTIC = "holistic"                    # 全局池化 baseline
    OBJECT_ONLY = "object_only"              # 仅对象token
    OBJECT_RELATION = "object_relation"      # 对象+关系 (核心)
    OBJECT_RELATION_VISIBILITY = "object_relation_visibility"  # +可见性
```

**实现机制**:
- `Holistic`: 直接全局平均池化，跳过abstraction
- `Object-only`: 过滤掉RELATION类型的token
- `Object-relation`: 标准DOOR-RL流程
- `Visibility`: 在abstraction前添加可见性加权

### 4. 评估指标系统

**指标分类**:
```
World Model:
  - Observation MSE
  - Reward MSE
  - Continue Accuracy
  - Collision Accuracy

Policy:
  - Action MSE (BC loss)
  - Value Loss

Advanced:
  - TTC Accuracy
  - Relation Prediction Accuracy
```

---

## 📈 实验流程

### 标准实验流程

```
1. 代码验证
   └─ python3 train_debug.py --epochs 2
   
2. 真实数据测试
   └─ python3 train_real_nuscenes.py --num-scenes 5 --epochs 10
   
3. 消融实验
   └─ ./run_ablation_study.sh
   
4. 结果分析
   └─ python3 analyze_ablation.py --table --plot --latex
   
5. 论文写作
   └─ 使用生成的表格和图表
```

### 预期时间

| 阶段 | 场景数 | Epochs | 时间 (单GPU) |
|------|--------|--------|-------------|
| 代码验证 | 5 | 10 | ~30分钟 |
| 初步实验 | 20 | 30 | ~2小时 |
| 正式实验 | 50 | 50 | ~8小时 |
| 所有变体 | 50×4 | 50 | ~32小时 |

---

## 🔬 实验设计

### 消融实验矩阵

| 实验 | 场景数 | 变体 | 种子 | 目的 |
|------|--------|------|------|------|
| Exp-1 | 20 | 全部4种 | 7 | 初步对比 |
| Exp-2 | 50 | 全部4种 | 7,42,123 | 正式实验 |
| Exp-3 | 50 | object_relation | 7,42,123 | 主方法稳定性 |

### 控制变量

所有实验保持相同:
- ✅ 训练/验证数据划分
- ✅ 超参数 (lr, batch_size, model_dim)
- ✅ 训练轮数
- ✅ 随机种子 (同组内)
- ✅ 硬件环境

---

## 📊 预期论文图表

### Table 1: 消融实验结果
```latex
\begin{table}[t]
\centering
\caption{Ablation Study on Object-Relational Representations}
\begin{tabular}{lcccc}
\toprule
Model Variant & Val Loss & Obs Loss & Reward Loss & Collision Loss \\
\midrule
Holistic (Baseline) & 1.2345 & 0.8901 & 0.4567 & 0.1234 \\
Object-Only & 1.0123 & 0.7654 & 0.3456 & 0.1123 \\
Object-Relation (Ours) & \textbf{0.8901} & \textbf{0.6543} & \textbf{0.2345} & \textbf{0.0987} \\
Object-Relation + Visibility & 0.9012 & 0.6678 & 0.2456 & 0.1001 \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure 1: 训练曲线
- 4条曲线对比 (holistic, object_only, object_relation, object_relation_visibility)
- X轴: Epoch (1-30)
- Y轴: Validation Loss
- 实线=train, 虚线=val

---

## ✨ 代码质量

### 工程改进

| 方面 | MVP | 现在 |
|------|-----|------|
| 类型注解 | 部分 | 完整 |
| 文档字符串 | 简单 | 详细 (含Args/Returns) |
| 错误处理 | 最小 | 完善 (try-except) |
| 日志输出 | 基础 | 详细 (进度+指标) |
| 配置管理 | JSON | JSON + dataclass |

### 可维护性

- ✅ 模块化设计 (adapter/model/evaluation分离)
- ✅ 工厂模式 (create_model_variant)
- ✅ 策略模式 (4种变体切换)
- ✅ 配置驱动 (所有超参数可配置)
- ✅ 实验可复现 (随机种子控制)

---

## 🚀 使用示例

### 最快验证路径 (30分钟)

```bash
# 1. 合成数据测试
python3 train_debug.py --epochs 2

# 2. 真实数据测试 (5场景)
python3 train_real_nuscenes.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 5 \
    --epochs 10

# 3. 单个消融变体
python3 train_ablation.py \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation \
    --num-scenes 5 \
    --epochs 10
```

### 正式实验路径 (2天)

```bash
# Day 1: 运行所有消融实验
./run_ablation_study.sh

# Day 2: 分析结果 + 多次运行
python3 analyze_ablation.py --table --plot --latex

# 验证统计显著性
for SEED in 7 42 123; do
    python3 train_ablation.py --variant object_relation --seed $SEED
done
```

---

## 📝 论文写作检查清单

### 实验部分
- [x] 消融实验设计
- [x] 基线方法 (holistic)
- [x] 评估指标定义
- [x] 数据集描述
- [x] 超参数设置
- [x] 训练/验证划分

### 结果部分
- [ ] 训练曲线图 (待运行实验)
- [ ] 消融结果表 (待运行实验)
- [ ] 案例分析 (待实现可视化)
- [ ] 统计显著性 (待多次运行)

### 附录
- [ ] 实现细节
- [ ] 硬件配置
- [ ] 训练时间
- [ ] 失败案例分析

---

## 🎓 与原始MVP的对比

### 研究能力

| 能力 | MVP | 现在 |
|------|-----|------|
| 合成数据训练 | ✅ | ✅ |
| 真实数据训练 | ❌ | ✅ |
| 消融实验 | ❌ | ✅ (4种变体) |
| 自动评估 | ❌ | ✅ |
| 结果分析 | ❌ | ✅ |
| 论文图表 | ❌ | ✅ (LaTeX) |

### 数据Pipeline

| 组件 | MVP | 现在 |
|------|-----|------|
| nuScenes action | 硬编码[0,0] | CAN总线+位姿计算 |
| nuScenes reward | 固定0.0 | 复合reward函数 |
| nuScenes map | 空列表 | 交通锥+护栏 |
| nuScenes relations | TODO | 完整实现 |
| 序列采样 | 不支持 | 支持next_sample |

### 实验支持

| 功能 | MVP | 现在 |
|------|-----|------|
| 变体切换 | 手动改代码 | 命令行参数 |
| 批量实验 | 无 | shell脚本 |
| 结果保存 | 无 | JSON+模型权重 |
| 可视化 | 无 | matplotlib+LaTeX |
| 多次运行 | 手动 | 循环脚本 |

---

## 🔮 未来工作 (可选)

### 短期 (1-2周)
1. 完成nuPlan Adapter (闭环评估)
2. 添加NAVSIM迁移实验
3. 实现反应式训练循环

### 中期 (1个月)
1. 集成3DGS评估
2. 添加更多基线方法
3. 实现分布式训练

### 长期 (论文扩展)
1. 跨城市迁移实验
2. 长尾场景分析
3. 实时部署优化

---

## 📞 技术支持

### 常见问题
1. **CAN总线数据不可用**: 代码自动fallback到位姿计算
2. **训练出现NaN**: 降低学习率或检查数据
3. **内存不足**: 减小batch size或模型维度

### 调试建议
```bash
# 启用详细日志
export PYTHONVERBOSE=1

# 单步调试
python3 -m pdb train_ablation.py --variant object_relation

# 检查数据
python3 -c "
from doorrl.data.real_dataset import NuScenesSceneDataset
dataset = NuScenesSceneDataset(...)
print(dataset[0])
"
```

---

**总结**: 本项目已从MVP框架升级为**完整的论文实验代码**，支持真实数据训练、消融实验、自动评估和结果分析。可以立即开始跑实验！

**下一步**: 按照 `PAPER_EXPERIMENT_GUIDE.md` 开始实验。
