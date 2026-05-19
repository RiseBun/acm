# DOOR-RL 项目完整说明

## 📖 项目概述

**DOOR-RL** (Decision-sufficient Object-relational Representation for Reinforcement Learning) 是一个用于**自动驾驶决策学习**的研究框架。

### 核心研究问题

本项目围绕三个核心假设展开研究：

1. **反应式训练的重要性** (reactive training matters)
   - 在训练时考虑其他交通参与者的反应，比假设他们行为固定更有效
   
2. **对象关系表示的重要性** (object-relational representation matters)
   - 显式建模车辆、行人之间的关系（距离、速度、碰撞风险），比学习全局隐表示更有效
   
3. **高保真迁移的重要性** (high-fidelity transfer matters)
   - 在仿真中训练的模型需要能够迁移到真实场景或其他仿真器

### 项目定位

这是一个 **MVP (Minimum Viable Product) 研究框架**，而非最终的生产系统。它的设计目标是：

- ✅ 提供稳定的场景token化schema
- ✅ 实现决策充分的抽象模块
- ✅ 实现对象关系世界模型
- ✅ 支持从合成数据到真实数据的平滑过渡
- ✅ 便于进行消融实验和对比研究

---

## 🎯 这个项目有什么用？

### 1. **学术研究价值**

#### 论文贡献方向
本项目支持发表自动驾驶/强化学习领域的研究论文，核心贡献点：

- **表示学习**: 提出对象关系token化的新方法
- **世界模型**: 构建反应式的对象关系世界模型
- **训练策略**: 对比reactive vs non-reactive训练
- **迁移能力**: 验证模型在不同benchmark间的迁移能力

#### 可进行的实验
- ✅ 表示消融实验（holistic vs object-only vs object+relation）
- ✅ 训练策略对比（replay vs reactive）
- ✅ 跨benchmark迁移实验（nuScenes → nuPlan → NAVSIM）
- ✅ 基线对比（BC, model-free RL, world model variants）

### 2. **技术能力**

#### 数据处理能力
- **多数据集支持**: nuScenes, nuPlan, NAVSIM
- **统一token schema**: 将不同格式的数据统一为标准token表示
- **关系特征计算**: 自动计算TTC、碰撞风险、车道冲突等
- **场景序列提取**: 支持时序建模和序列学习

#### 模型能力
- **Token编码**: 将原始传感器/标注数据编码为隐向量
- **决策抽象**: 从大量token中选择最关键的top-k个
- **世界模型**: 预测下一时刻的状态、奖励、碰撞风险
- **策略学习**: 输出连续控制动作（加速度、转向）

#### 训练能力
- **多目标损失**: observation + reward + collision + BC
- **梯度裁剪**: 稳定训练过程
- **合成+真实数据**: 支持从调试到真实训练的过渡

### 3. **工程价值**

#### 可扩展架构
- **Adapter模式**: 轻松接入新的数据集
- **配置驱动**: JSON配置控制所有超参数
- **模块化设计**: encoder/abstraction/world_model/policy可独立替换

#### 开发友好
- **合成数据调试**: 无需真实数据即可验证代码
- **完整测试**: 单元测试覆盖核心功能
- **详细文档**: 使用指南、API文档、性能参考

---

## 🏗️ 项目架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        数据层 (Data)                         │
├─────────────────────────────────────────────────────────────┤
│  nuScenes Adapter │ nuPlan Adapter │ NAVSIM Adapter         │
│  ↓                │               │                        │
│  └───────────────→│  Token Schema │←───────────────────────┘│
│                   │  (统一表示)    │                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      模型层 (Model)                          │
├─────────────────────────────────────────────────────────────┤
│  TokenEncoder → DecisionSufficientAbstraction               │
│       ↓                    ↓                                │
│  ReactiveObjectRelationalWorldModel ←→ ActorCriticHead      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     训练层 (Training)                        │
├─────────────────────────────────────────────────────────────┤
│  多目标Loss │ 梯度裁剪 │ AdamW优化器 │ 训练/验证循环         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     评估层 (Evaluation)                      │
├─────────────────────────────────────────────────────────────┤
│  nuPlan Closed-loop │ NAVSIM Transfer │ 3DGS (optional)    │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

```
原始数据 (nuScenes/nuPlan)
    ↓
[Adapter转换]
    ↓
Token Schema (ego, objects, map, relations)
    ↓
[TokenEncoder]
    ↓
隐向量表示 [B, S, D]
    ↓
[DecisionSufficientAbstraction]
    ↓
Top-K 关键token [B, K, D]
    ↓
[WorldModel + Policy]
    ↓
动作预测 + 未来预测
```

---

## 📁 项目结构 (完整版)

```
code/
│
├── 📄 README.md                          # 项目说明
├── 📄 IMPLEMENTATION_SUMMARY.md          # 实现总结
├── 📄 pyproject.toml                     # Python项目配置
│
├── ⚙️ configs/
│   ├── debug_mvp.json                    # 调试配置 (小模型, 少数据)
│   └── nuplan_stack_template.json        # nuPlan实验配置模板
│
├── 📖 docs/
│   ├── SERVER_HANDOFF.md                 # 服务器交接文档
│   ├── TOKENIZATION_SPEC.md              # Token化规范
│   └── REAL_DATA_PIPELINE.md             # 真实数据使用指南 [NEW]
│
├── 🧠 src/doorrl/
│   ├── __init__.py
│   │
│   ├── 📊 schema.py                      # Token Schema定义
│   │   ├── TokenType (枚举): EGO, VEHICLE, PEDESTRIAN, etc.
│   │   └── SceneBatch (数据类): tokens, mask, actions, etc.
│   │
│   ├── ⚙️ config.py                      # 配置管理
│   │   ├── ModelConfig: 模型超参数
│   │   ├── TrainingConfig: 训练超参数
│   │   ├── DataConfig: 数据配置
│   │   ├── BenchmarkConfig: benchmark配置
│   │   └── DoorRLConfig: 总配置类
│   │
│   ├── 🛠️ utils.py                       # 工具函数
│   │   ├── set_seed(): 随机种子
│   │   ├── masked_mean(): 掩码平均
│   │   └── batched_index_select(): 批量索引
│   │
│   ├── 🔌 adapters/                      # 数据集适配器
│   │   ├── base.py                       # 基础类和转换器
│   │   │   ├── TokenizationSpec
│   │   │   ├── NormalizedSceneConverter
│   │   │   └── BenchmarkMode
│   │   │
│   │   ├── nuscenes_adapter.py           # nuScenes基础Adapter
│   │   ├── nuplan_adapter.py             # nuPlan基础Adapter
│   │   ├── navsim_adapter.py             # NAVSIM基础Adapter
│   │   │
│   │   ├── nuscenes_real_adapter.py      # [NEW] nuScenes真实数据Adapter
│   │   │   ├── NuScenesRealDataAdapter
│   │   │   ├── 场景加载
│   │   │   ├── Ego状态提取
│   │   │   ├── 对象提取
│   │   │   └── 关系特征计算 (TTC, risk, etc.)
│   │   │
│   │   └── nuplan_real_adapter.py        # [NEW] nuPlan真实数据Adapter
│   │       ├── NuPlanRealDataAdapter
│   │       ├── Reactive/Non-reactive模式
│   │       └── 场景转换
│   │
│   ├── 📦 data/                          # 数据集
│   │   ├── synthetic.py                  # 合成数据生成器
│   │   │   └── SyntheticDrivingDataset
│   │   │
│   │   └── real_dataset.py               # [NEW] 真实数据集
│   │       ├── RealDrivingDataset
│   │       └── NuScenesSceneDataset
│   │
│   ├── 🧠 models/                        # 模型定义
│   │   ├── encoder.py                    # Token编码器
│   │   │   └── TokenEncoder
│   │   │       ├── 输入投影
│   │   │       ├── 类型嵌入
│   │   │       └── MLP + LayerNorm
│   │   │
│   │   ├── abstraction.py                # 决策充分抽象
│   │   │   └── DecisionSufficientAbstraction
│   │   │       ├── Ego-query注意力
│   │   │       ├── Top-K选择
│   │   │       └── 全局latent池化
│   │   │
│   │   ├── world_model.py                # 世界模型
│   │   │   └── ReactiveObjectRelationalWorldModel
│   │   │       ├── Action-token注入
│   │   │       ├── Transformer编码
│   │   │       └── 预测头 (next-token, reward, collision)
│   │   │
│   │   ├── policy.py                     # 策略头
│   │   │   └── ActorCriticHead
│   │   │       ├── 动作均值预测
│   │   │       ├── 动作log-std
│   │   │       └── 状态价值预测
│   │   │
│   │   └── doorrl.py                     # 主模型
│   │       └── DoorRLModel
│   │           └── Encoder + Abstraction + WorldModel + Policy
│   │
│   └── 🎓 training/                      # 训练逻辑
│       ├── losses.py                     # 损失函数
│       │   └── compute_losses()
│       │       ├── Observation loss (MSE)
│       │       ├── Reward loss (MSE)
│       │       ├── Continue loss (BCE)
│       │       ├── Collision loss (BCE)
│       │       └── BC loss (MSE)
│       │
│       └── trainer.py                    # 训练器
│           └── DoorRLTrainer
│               ├── fit(): 训练循环
│               └── run_epoch(): 单轮训练
│
├── 🧪 tests/                             # 测试
│   ├── test_forward.py                   # 前向传播测试
│   ├── test_adapters.py                  # Adapter测试
│   └── test_real_data.py                 # [NEW] 真实数据测试
│
├── 🚀 训练脚本
│   ├── train_debug.py                    # 合成数据调试训练
│   └── train_real_nuscenes.py            # [NEW] 真实nuScenes训练
│
└── 🔍 explore_nuscenes.py                # [NEW] nuScenes数据探索
```

---

## 🔬 核心组件详解

### 1. Token Schema

**作用**: 统一不同数据源的表示

```python
# 8种Token类型
TokenType.EGO        # 自车
TokenType.VEHICLE    # 其他车辆
TokenType.PEDESTRIAN # 行人
TokenType.CYCLIST    # 骑行者
TokenType.MAP        # 地图元素
TokenType.SIGNAL     # 交通信号
TokenType.RELATION   # 关系token
TokenType.PAD        # 填充

# 场景批次
SceneBatch:
  - tokens: [B, S, D]         # token特征
  - token_mask: [B, S]         # 有效掩码
  - token_types: [B, S]        # token类型
  - actions: [B, A]            # 动作
  - next_tokens: [B, S, D]     # 下一时刻token
  - rewards: [B]               # 奖励
  - continues: [B]             # 继续标志
```

### 2. 模型架构

```
DoorRLModel
│
├── TokenEncoder
│   Input: tokens [B, S, 40]
│   Output: latent [B, S, 128]
│
├── DecisionSufficientAbstraction
│   Input: latent [B, S, 128]
│   Output: 
│     - selected_tokens [B, 16, 128]  # Top-K关键token
│     - global_latent [B, 128]         # 全局表示
│
├── ReactiveObjectRelationalWorldModel
│   Input: selected_tokens + actions
│   Output:
│     - predicted_next_tokens [B, 16, 40]
│     - predicted_reward [B]
│     - predicted_collision [B]
│     - predicted_continue [B]
│
└── ActorCriticHead
    Input: global_latent
    Output:
      - action_mean [B, 2]     # 加速度, 转向
      - action_log_std [B, 2]  # 探索噪声
      - value [B]              # 状态价值
```

### 3. 关系特征

**计算的关系特征** (在nuscenes_real_adapter.py中):

```python
Relation Features:
  - dx, dy: 相对位置
  - rel_vx, rel_vy: 相对速度
  - distance: 欧氏距离
  - ttc: 碰撞时间 (Time To Collision)
  - risk: 碰撞风险 (1/distance)
  - lane_conflict: 车道冲突 (同车道=1)
  - visibility: 可见性
  - priority: 优先级
```

### 4. 损失函数

**多目标损失**:

```python
Total Loss = 
  1.0 × Observation_Loss (MSE)        # 世界模型预测
  0.5 × Reward_Loss (MSE)             # 奖励预测
  0.25 × Continue_Loss (BCE)          # 继续预测
  0.25 × Collision_Loss (BCE)         # 碰撞预测
  0.1 × BC_Loss (MSE)                 # 行为克隆
```

---

## 🚀 使用指南

### 快速开始

#### 1. 测试环境
```bash
cd /mnt/cpfs/prediction/lipeinan/code
conda activate find_physics_zone

# 运行基础测试
python3 -m pytest tests/ -v

# 运行真实数据测试
python3 test_real_data.py
```

#### 2. 合成数据训练 (调试)
```bash
python3 train_debug.py \
    --config configs/debug_mvp.json \
    --epochs 10
```

#### 3. 真实数据训练
```bash
# 使用nuScenes真实数据
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 5 \
    --epochs 20
```

### 典型工作流

```
1. 代码调试阶段
   ├─ 使用合成数据 (train_debug.py)
   ├─ 验证模型形状和训练循环
   └─ 快速迭代

2. 真实数据验证
   ├─ 使用少量真实场景 (5-10个)
   ├─ 验证Adapter和数据pipeline
   └─ 调整超参数

3. 大规模训练
   ├─ 使用50-100个场景
   ├─ 增大模型 (model_dim=256, num_layers=4)
   └─ 训练50+ epochs

4. 消融实验
   ├─ 对比不同表示 (object vs object+relation)
   ├─ 对比训练策略 (replay vs reactive)
   └─ 记录结果

5. 评估迁移
   ├─ nuPlan闭环评估
   ├─ NAVSIM外部评估
   └─ 撰写论文
```

---

## 📊 支持的Datasets

| 数据集 | 路径 | 用途 | 状态 |
|-------|------|------|------|
| **nuScenes** | `/mnt/datasets/e2e-nuscenes/20260302/` | 离线token化, 世界模型预训练 | ✅ 完全集成 |
| **nuPlan** | `/mnt/datasets/e2e-nuplan-lon/` | 主闭环benchmark | ⚠️ 框架就绪, 待连接API |
| **NAVSIM** | `/mnt/datasets/navsim/v1.0.0/` | 外部迁移评估 | ⚠️ 待集成 |

**数据规模**:
- nuScenes: 850场景, 34,149样本, 1,166,187标注
- nuPlan: 完整训练/验证/测试集 (约2000+场景)

---

## 🎓 研究应用示例

### 实验1: 表示消融

```python
# 对比不同表示
variants = {
    'holistic': '全局latent, 无对象分离',
    'object_only': '仅对象token, 无关系',
    'object_relation': '对象+关系token',
    'object_relation_visibility': '对象+关系+可见性',
}

# 评估指标
metrics = [
    'imagined_rollout_error',
    'collision_prediction_accuracy',
    'interactive_success_rate',
    'rare_agent_recall',
]
```

### 实验2: 训练策略对比

```python
experiments = {
    'replay_train_replay_test': '非反应式训练+测试',
    'replay_train_reactive_test': '非反应式训练+反应式测试',
    'reactive_train_reactive_test': '反应式训练+测试',
}
```

### 实验3: 跨Benchmark迁移

```python
transfer_path = {
    'train': 'nuScenes (离线数据)',
    'finetune': 'nuPlan (闭环)',
    'evaluate': 'NAVSIM (外部)',
}
```

---

## 💡 项目优势

### 1. **研究友好**
- ✅ 从合成数据开始，无需真实数据即可调试
- ✅ 模块化设计，易于替换组件
- ✅ 配置驱动，实验管理方便

### 2. **工程规范**
- ✅ 类型注解完整
- ✅ 单元测试覆盖
- ✅ 文档完善

### 3. **可扩展性**
- ✅ Adapter模式接入新数据集
- ✅ 支持多种模型变体
- ✅ 灵活的训练策略

### 4. **资源就绪**
- ✅ 真实数据已下载 (nuScenes, nuPlan)
- ✅ 环境已配置 (所有依赖安装)
- ✅ 代码已测试 (全部通过)

---

## 🔧 技术栈

- **深度学习**: PyTorch 2.1.0+cu121
- **数据集**: nuScenes-devkit, nuplan-devkit, navsim
- **Python**: 3.10.20
- **环境**: Conda (find_physics_zone)
- **配置**: JSON + dataclass

---

## 📈 性能参考

**训练速度** (基于测试):
- 合成数据: ~1秒/epoch (64样本)
- 真实数据: ~0.1秒/样本转换
- 内存: ~2GB (小批量)

**数据加载**:
- nuScenes初始化: ~30秒
- 单场景加载: ~0.1秒

---

## 📝 相关文档

- [README.md](file:///mnt/cpfs/prediction/lipeinan/code/README.md) - 原始项目说明
- [IMPLEMENTATION_SUMMARY.md](file:///mnt/cpfs/prediction/lipeinan/code/IMPLEMENTATION_SUMMARY.md) - 实现总结
- [REAL_DATA_PIPELINE.md](file:///mnt/cpfs/prediction/lipeinan/code/docs/REAL_DATA_PIPELINE.md) - 真实数据使用指南
- [TOKENIZATION_SPEC.md](file:///mnt/cpfs/prediction/lipeinan/code/docs/TOKENIZATION_SPEC.md) - Token化规范

---

## 🎯 总结

**DOOR-RL是一个用于自动驾驶决策学习的研究框架**，它的核心价值在于：

1. **学术价值**: 支持发表高质量研究论文
2. **技术价值**: 提供完整的表示学习+世界模型+RL pipeline
3. **工程价值**: 模块化、可扩展、易调试

**适用场景**:
- ✅ 自动驾驶决策算法研究
- ✅ 世界模型表示学习
- ✅ 强化学习训练策略对比
- ✅ 跨domain迁移学习

**当前状态**: 
- ✅ 环境配置完成
- ✅ 真实数据Adapter实现
- ✅ 测试全部通过
- ✅ 可以开始训练

**下一步**:
- 扩大训练规模 (50-100场景)
- 进行消融实验
- 集成nuPlan闭环评估

---

**项目位置**: `/mnt/cpfs/prediction/lipeinan/code/`  
**环境**: `conda activate find_physics_zone`  
**数据**: `/mnt/datasets/e2e-nuscenes/20260302/`
