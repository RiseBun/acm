# DOOR-RL 正确实施路线

## ⚠️ 之前的错误

❌ **无限工程化** - 添加一堆代码，但不知道哪张表能撑论文  
❌ **包装成论文样子** - 目录好看但内容空缺  
❌ **跳步实施** - 没有证明表示优势就直接做RL  

---

## ✅ 正确路线：论文驱动的分阶段实验

**核心原则**:
1. **按论文主张组织实验** - 每个阶段对应一张论文表格
2. **按工程阶段逐步实现** - 不跳步，不超前
3. **Go/No-Go决策点** - 每阶段验证假设，失败就重新思考

---

## 📋 论文表格与阶段对应

| 阶段 | 论文表格 | 核心主张 | 验证内容 | 状态 |
|------|---------|---------|---------|------|
| **Stage 0** | Table 3 | object-relational representation matters | 表示充分性消融 | ✅ 已实现 |
| **Stage 1** | Table 1 | world model能滚多步 | multi-step rollout | ⏳ 待实现 |
| **Stage 2** | Table 2 | latent imagination中训练policy | actor-critic RL | ⏳ 待实现 |
| **Stage 3** | Table 4 | reactive training matters | closed-loop交互 | ⏳ 待实现 |
| **Stage 4** | Table 5 | 不是只在梦里赢 | 高保真评估 | ⏳ 待实现 |

---

## 🎯 当前阶段：Stage 0 (已完成)

### 实现内容

✅ **论文式目录结构**
```
experiments/
├── table3_representation_sufficiency/  ← 当前实现
├── table1_multi_step_rollout/
├── table2_latent_imagination_rl/
├── table4_reactive_closed_loop/
└── table5_high_fidelity_eval/
```

✅ **Table 3的4个核心指标**
- Rollout Error (one-step prediction error)
- Reward Error (reward prediction MSE)
- Collision Prediction Accuracy (基于TTC)
- Rare-Agent Recall (行人/骑行者检测)

✅ **Stage 0实验脚本**
```bash
# 运行所有变体
python3 run_stage0_table3.py --variant all --num-scenes 20 --epochs 30

# 运行单个变体
python3 run_stage0_table3.py --variant object_relation --num-scenes 20 --epochs 30
```

✅ **自动生成LaTeX表格**
```latex
\begin{table}[t]
\caption{Representation Sufficiency Ablation Study}
...
\end{table}
```

### 输出文件

```
experiments/table3_representation_sufficiency/
├── holistic/table3_results.json
├── object_only/table3_results.json
├── object_relation/table3_results.json
├── object_relation_visibility/table3_results.json
└── table3_complete.json
```

### 下一步行动

1. **运行实验** (1-2天)
   ```bash
   python3 run_stage0_table3.py --variant all --num-scenes 20 --epochs 30
   ```

2. **分析Table 3结果**
   - Object + Relation 是否显著优于baseline?
   - 如果是 → 继续Stage 1
   - 如果否 → 重新设计关系特征

---

## 🔮 后续阶段规划

### Stage 1: Multi-step Latent Rollout

**目标**: 证明world model能滚多步

**需要实现**:
```python
# imagination/rollout.py
def latent_rollout(model, z_t, actions, horizon):
    """
    z_t → z_{t+1} → z_{t+2} → ... → z_{t+H}
    """
    trajectory = [z_t]
    z_current = z_t
    
    for t in range(horizon):
        z_next = model.world_model.predict_next(z_current, actions[t])
        trajectory.append(z_next)
        z_current = z_next
    
    return trajectory
```

**论文表格 (Table 1)**:
```
Horizon | Holistic | Object-only | Object+Relation | Full
--------|----------|-------------|-----------------|------
H=1     |          |             |                 |
H=5     |          |             |                 |
H=10    |          |             |                 |
```

**Go/No-Go**: 如果H=5误差就爆炸 → world model架构有问题

---

### Stage 2: Latent Imagination RL

**目标**: 在latent imagination中训练policy

**需要实现**:
```python
# rl/dreamer.py
class DreamerRL:
    def train(self):
        # 1. 在latent空间rollout
        trajectory = self.world_model.rollout(z_t, policy, horizon=H)
        
        # 2. 计算lambda return
        returns = compute_lambda_return(trajectory)
        
        # 3. Actor loss
        actor_loss = -returns.mean()
        
        # 4. Critic loss
        critic_loss = MSE(value, returns)
        
        # 5. 更新policy
        (actor_loss + critic_loss).backward()
```

**论文表格 (Table 2)**:
```
Method          | Latent Return ↑ | Collision Risk ↓ | Comfort ↑
----------------|-----------------|------------------|----------
BC              |                 |                  |
Model-free RL   |                 |                  |
Holistic Dreamer|                 |                  |
DOOR-RL (ours)  |                 |                  |
```

**Go/No-Go**: 如果不如BC → imagination训练有问题

---

### Stage 3: Reactive Closed-Loop

**目标**: 证明reactive training matters

**需要实现**:
```python
# envs/smart_env.py (或CARLA/Waymax)
class ReactiveEnv:
    def step(self, action):
        # 其他智能体也根据ego动作做出反应
        other_agents.react(action)
        next_state = self.simulate()
        return next_state, reward, done
```

**论文表格 (Table 4) - 最有攻击力的表**:
```
Setting                 | Merge ↑ | Yield ↑ | Cut-in ↑ | Pedestrian ↑
------------------------|---------|---------|----------|-------------
Replay+Replay           |         |         |          |
Replay+Reactive         |         |         |          |
Reactive+Reactive(ours) |         |         |          |
```

**Go/No-Go**: 如果Reactive训练没有优势 → 需要重新设计reward或环境

---

### Stage 4: High-Fidelity Evaluation

**目标**: 证明不是只在梦里赢

**需要实现**:
- CARLA closed-loop
- 或 3DGS evaluation

**论文表格 (Table 5)**:
```
Method          | CARLA Score ↑ | Success Rate ↑ | Infraction ↓
----------------|---------------|----------------|-------------
Baseline        |               |                |
DOOR-RL (ours)  |               |                |
```

---

## 📂 代码组织

### 当前已实现

```
src/doorrl/
├── data/                          # ✅ 完成
│   ├── nuscenes_adapter.py
│   └── real_dataset.py
├── models/                        # ✅ 完成
│   ├── encoder.py
│   ├── abstraction.py
│   ├── world_model.py            # one-step
│   ├── policy.py
│   └── doorrl_variant.py         # 4种变体
├── evaluation/                    # ✅ Stage 0完成
│   ├── metrics.py
│   └── table3_metrics.py         # Table 3指标
└── training/
    ├── losses.py
    └── trainer.py
```

### 后续需要添加

```
src/doorrl/
├── imagination/                   # ⏳ Stage 1
│   ├── rollout.py
│   └── imagination_trainer.py
├── rl/                            # ⏳ Stage 2
│   ├── dreamer.py
│   └── actor_critic.py
├── envs/                          # ⏳ Stage 3
│   ├── smart_env.py
│   └── carla_env.py
└── evaluation/
    └── closed_loop_evaluator.py   # ⏳ Stage 3
```

---

## 🎓 论文写作时间线

### Week 1-2: Stage 0
- [ ] 运行Table 3实验
- [ ] 分析结果
- [ ] 如果成功 → 写Section 4.2 (Representation Sufficiency)

### Week 3-4: Stage 1
- [ ] 实现multi-step rollout
- [ ] 运行Table 1实验
- [ ] 写Section 4.3 (World Model Capability)

### Week 5-7: Stage 2
- [ ] 实现latent imagination RL
- [ ] 运行Table 2实验
- [ ] 写Section 4.4 (Latent RL Training)

### Week 8-10: Stage 3
- [ ] 接入reactive environment
- [ ] 运行Table 4实验
- [ ] 写Section 4.5 (Reactive Closed-Loop)

### Week 11-12: Stage 4
- [ ] CARLA评估
- [ ] 运行Table 5实验
- [ ] 写完整论文

---

## 💡 关键决策点

### Stage 0 → Stage 1
**问题**: Object-relation表示是否有优势？

**判断标准**:
- ✅ Rollout Error降低≥10% → 继续
- ❌ 无显著差异 → 重新设计关系特征

### Stage 1 → Stage 2
**问题**: World model能否滚多步？

**判断标准**:
- ✅ H=5误差可接受 → 继续
- ❌ H=5就爆炸 → 改进world model架构

### Stage 2 → Stage 3
**问题**: Latent RL是否优于BC？

**判断标准**:
- ✅ Latent Return显著高于BC → 继续
- ❌ 不如BC → 检查imagination训练

### Stage 3 → Stage 4
**问题**: Reactive训练是否有效？

**判断标准**:
- ✅ Reactive+Reactive最好 → 继续
- ❌ 无优势 → 检查reward设计或环境

---

## 🚨 常见陷阱

### ❌ 错误做法
1. 跳过Stage 0直接做Stage 3
2. Stage 0失败但继续做后续实验
3. 无限优化Stage 0，不敢开始Stage 1
4. 同时做多个stage，结果混乱

### ✅ 正确做法
1. 严格按顺序执行
2. 每阶段验证假设
3. 失败就停下来重新思考
4. 一阶段完成再开始下一阶段

---

## 📝 总结

**当前状态**: Stage 0代码已完成，等待运行实验

**下一步**: 
```bash
# 1. 验证代码
python3 run_stage0_table3.py --variant holistic --num-scenes 5 --epochs 2

# 2. 正式实验
python3 run_stage0_table3.py --variant all --num-scenes 20 --epochs 30

# 3. 分析结果
cat experiments/table3_representation_sufficiency/table3_complete.json

# 4. 决定是否继续Stage 1
```

**核心原则**: 
- **论文驱动** - 每个实验对应一张表
- **分阶段验证** - 不跳步
- **Go/No-Go** - 失败就重新思考

---

**按照这个路线，不会迷路，每步都有明确目标！**
