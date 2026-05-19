# DOOR-RL 论文实验目录结构

## 按论文章节组织的实验

```
experiments/
│
├── table3_representation_sufficiency/     # Stage 0: 表示充分性消融 [当前实现]
│   ├── holistic/                          #   - Holistic baseline
│   ├── object_only/                       #   - Object-only
│   ├── object_relation/                   #   - Object + Relation (核心)
│   └── object_relation_visibility/        #   - Object + Relation + Visibility
│
├── table1_multi_step_rollout/             # Stage 1: Multi-step latent rollout [下一步]
│   ├── horizon_1/
│   ├── horizon_5/
│   └── horizon_10/
│
├── table2_latent_imagination_rl/          # Stage 2: Latent imagination RL [之后]
│   ├── bc/
│   ├── model_free_rl/
│   ├── holistic_dreamer/
│   └── door_rl/
│
├── table4_reactive_closed_loop/           # Stage 3: Reactive closed-loop [之后]
│   ├── replay_train_replay_test/
│   ├── replay_train_reactive_test/
│   └── reactive_train_reactive_test/
│
└── table5_high_fidelity_eval/             # Stage 4: 高保真评估 [最后]
    ├── carla/
    └── 3dgs/
```

## 代码组织结构

```
src/doorrl/
│
├── data/                                  # 数据层
│   ├── nuscenes_adapter.py
│   ├── nuplan_adapter.py
│   └── real_dataset.py
│
├── models/                                # 模型层
│   ├── encoder.py
│   ├── abstraction.py
│   ├── world_model.py                    # one-step world model
│   ├── policy.py
│   └── doorrl_variant.py                 # 4种表示变体
│
├── imagination/                           # Stage 1+: Latent imagination [TODO]
│   ├── rollout.py                        #   - Multi-step rollout
│   ├── imagination_trainer.py            #   - Imagination training
│   └── lambda_return.py                  #   - Lambda return
│
├── rl/                                    # Stage 2+: RL训练 [TODO]
│   ├── actor_critic.py
│   ├── dreamer.py
│   └── ppo.py
│
├── envs/                                  # Stage 3+: 环境接口 [TODO]
│   ├── base_env.py
│   ├── smart_env.py
│   ├── carla_env.py
│   └── waymax_env.py
│
├── evaluation/                            # 评估层
│   ├── metrics.py                        # 当前实现: Table 3指标
│   ├── rollout_evaluator.py              # Stage 1: Rollout评估
│   └── closed_loop_evaluator.py          # Stage 3: 闭环评估
│
└── training/
    ├── losses.py
    ├── trainer.py
    └── ablation_trainer.py               # Stage 0: 消融实验训练
```

---

## 当前阶段：Stage 0

### 目标
证明 **object-relational representation matters**

### 实验设计
- **输入**: 相同nuScenes数据
- **变量**: 4种表示方式
- **指标**: 
  1. Rollout Error (想象 rollout 误差)
  2. Reward Error (奖励预测误差)
  3. Collision Prediction (碰撞预测准确率)
  4. Rare-Agent Recall (稀有智能体召回率)

### 论文表格 (Table 3)

```
Table 3: Representation Sufficiency Ablation

Variant              | Rollout Error ↓ | Reward Error ↓ | Collision Pred. ↑ | Rare-Agent Recall ↑
---------------------|-----------------|----------------|-------------------|---------------------
Holistic             |       -         |       -        |        -          |          -
Object-only          |       -         |       -        |        -          |          -
Object + Relation    |       -         |       -        |        -          |          -
Obj+Rel+Visibility   |       -         |       -        |        -          |          -
```

### 实现状态
- [x] 4种模型变体
- [x] nuScenes真实数据Pipeline
- [x] 训练/验证集划分
- [ ] Rollout Error 指标
- [ ] Reward Error 指标 (已有，需验证)
- [ ] Collision Prediction 指标
- [ ] Rare-Agent Recall 指标
- [ ] 自动化实验脚本

### 下一步
1. 实现4个核心评估指标
2. 运行实验填充Table 3
3. 分析结果：如果Object+Relation显著优于baseline，继续Stage 1

---

## Stage 1-4 规划

### Stage 1: Multi-step Latent Rollout
**核心问题**: World model能否滚多步？

**实验**:
- Open-loop multi-step prediction
- Horizon: H=1, 5, 10
- 对比4种表示

**论文表格 (Table 1)**:
```
Horizon | Holistic | Object-only | Object+Relation | Full
--------|----------|-------------|-----------------|------
H=1     |    -     |      -      |        -        |   -
H=5     |    -     |      -      |        -        |   -
H=10    |    -     |      -      |        -        |   -
```

### Stage 2: Latent Imagination RL
**核心问题**: 能否在latent imagination中训练policy？

**实验**:
- BC baseline
- Model-free RL
- Holistic Dreamer-style
- DOOR-RL (ours)

**论文表格 (Table 2)**:
```
Method          | Latent Return ↑ | Collision Risk ↓ | Comfort ↑ | Training Time
----------------|-----------------|------------------|-----------|--------------
BC              |        -        |        -         |     -     |      -
Model-free RL   |        -        |        -         |     -     |      -
Holistic Dreamer|        -        |        -         |     -     |      -
DOOR-RL (ours)  |        -        |        -         |     -     |      -
```

### Stage 3: Reactive Closed-Loop
**核心问题**: Reactive training matters?

**实验**:
- Replay Train + Replay Test
- Replay Train + Reactive Test
- Reactive Train + Reactive Test

**论文表格 (Table 4)** - **最有攻击力的表**:
```
Setting                 | Merge ↑ | Yield ↑ | Cut-in ↑ | Pedestrian ↑ | Near Collision ↓
------------------------|---------|---------|----------|--------------|-----------------
Replay+Replay           |    -    |    -    |     -    |      -       |        -
Replay+Reactive         |    -    |    -    |     -    |      -       |        -
Reactive+Reactive (ours)|    -    |    -    |     -    |      -       |        -
```

### Stage 4: High-Fidelity Evaluation
**核心问题**: 不是只在梦里赢

**实验**:
- CARLA closed-loop
- 3DGS evaluation (可选)

**论文表格 (Table 5)**:
```
Method          | CARLA Score ↑ | Success Rate ↑ | Infraction ↓
----------------|---------------|----------------|-------------
Baseline        |       -       |       -        |      -
DOOR-RL (ours)  |       -       |       -        |      -
```

---

## 执行原则

### ✅ 正确做法
1. **论文驱动**: 每个实验对应论文的一张表
2. **分阶段验证**: Stage 0成功 → Stage 1 → Stage 2...
3. **Go/No-Go决策点**: 
   - Stage 0无明显收益 → 重新思考表示设计
   - Stage 1多步rollout失败 → world model有问题
   - Stage 2 latent RL不如BC → imagination训练有问题

### ❌ 错误做法
1. 无限工程化，不知道哪张表能撑论文
2. 先包装成论文样子，实际内容空缺
3. 跳过Stage 0直接做Stage 3 (没有表示优势证明)

---

## 当前行动

**立即执行**:
1. ✅ 创建论文式目录结构
2. 🔄 实现Table 3的4个指标
3. ⏳ 运行Stage 0实验
4. ⏳ 分析结果，决定是否继续

**代码改动范围**:
- 只修改 `evaluation/metrics.py` (添加4个指标)
- 只修改 `train_ablation.py` (输出Table 3格式)
- 不改动模型架构
- 不添加新模块

**时间目标**: 1-2天内完成Stage 0实验

---

**总结**: 当前只做Stage 0，证明object-relation表示有价值，再决定后续。
