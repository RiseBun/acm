# 🚀 如何开始实验 - 快速回答

## 你现在还差什么？

### ✅ 不差的（已就绪）
- ✅ 环境配置
- ✅ 数据集 (nuScenes, nuPlan)
- ✅ 代码框架 (Adapter, Model, Trainer)
- ✅ 测试验证

### ⚠️ 刚才补充的（现在有了）
- ✅ 实验配置文件: `configs/experiment_baseline.json`
- ✅ 增强训练脚本: `train_experiment.py`
- ✅ 快速启动脚本: `run_experiment.sh`
- ✅ 实验目录结构: `experiments/`

### ❌ 真正缺的（可选，不影响开始）
- ❌ TensorBoard/WandB日志系统（可以用打印代替）
- ❌ 完整的实验管理平台（手动管理也可以）
- ❌ 自动化评估脚本（可以手动评估）
- ❌ 可视化工具（后期再加）

---

## 🎯 现在就可以开始实验！

### 方法1: 一键启动（最简单）⭐⭐⭐

```bash
cd /mnt/cpfs/prediction/lipeinan/code
bash run_experiment.sh
```

然后选择:
- **选项1**: 快速测试（2分钟）
- **选项2**: 小实验（10分钟）
- **选项3**: 基线实验（20分钟）

### 方法2: 命令行启动

```bash
cd /mnt/cpfs/prediction/lipeinan/code
conda activate find_physics_zone

# 快速测试
python3 train_experiment.py \
    --config configs/debug_mvp.json \
    --num-scenes 3 \
    --epochs 2

# 或基线实验
python3 train_experiment.py \
    --config configs/experiment_baseline.json \
    --num-scenes 10 \
    --epochs 20
```

### 方法3: 使用旧脚本

```bash
python3 train_real_nuscenes.py \
    --scenes scene-0001 scene-0002 scene-0003 \
    --epochs 5
```

---

## 📊 实验输出

运行后会自动创建:

```
experiments/runs/
└── experiment_baseline_20260416_204500/
    ├── config.json              # 实验配置
    ├── args.json                # 命令行参数
    ├── checkpoints/
    │   ├── latest.pth           # 最新模型
    │   └── best.pth             # 最佳模型
    └── logs/
        └── training.log         # 训练日志
```

---

## 🎓 推荐实验流程

### 第1步: 快速验证（现在，2分钟）

```bash
python3 train_experiment.py \
    --config configs/debug_mvp.json \
    --num-scenes 3 \
    --epochs 2
```

**目标**: 确认代码和数据都正常

### 第2步: 小规模实验（今天，20分钟）

```bash
python3 train_experiment.py \
    --config configs/experiment_baseline.json \
    --num-scenes 10 \
    --epochs 20
```

**目标**: 观察训练曲线，验证loss下降

### 第3步: 正式实验（本周，数小时）

```bash
# 创建50场景配置
cp configs/experiment_baseline.json configs/experiment_50scenes.json
# 编辑: num_scenes=50, epochs=50

python3 train_experiment.py \
    --config configs/experiment_50scenes.json \
    --num-scenes 50 \
    --epochs 50
```

**目标**: 训练一个可用的世界模型

### 第4步: 消融实验（下周）

```bash
# 实验1: 无关系token
# 实验2: 有关系token
# 实验3: 完整关系
# 对比结果
```

---

## 💡 关键建议

### ✅ 现在就做
1. **立即运行快速测试** - 不要等
2. **观察loss是否下降** - 验证训练正常
3. **记录实验结果** - 建立实验笔记

### ⏰ 今天完成
1. 跑完一个小实验（10-20场景）
2. 确认训练曲线合理
3. 保存检查点

### 📅 本周完成
1. 训练正式模型（50场景，50 epochs）
2. 尝试不同超参数
3. 开始消融实验

### 🔮 后期优化
1. 添加TensorBoard日志
2. 添加验证集评估
3. 集成nuPlan闭环测试
4. 完善实验管理

---

## 🔍 如何判断实验是否正常？

### 正常现象
```
epoch=1  train_total=150.23  val_total=145.67
epoch=5  train_total=120.45  val_total=118.23  ← loss下降
epoch=10 train_total=95.67   val_total=98.45
epoch=15 train_total=78.23   val_total=82.56   ← 继续下降
epoch=20 train_total=65.89   val_total=71.34
```

### 异常现象
```
epoch=1  train_total=150.23  val_total=145.67
epoch=5  train_total=150.45  val_total=148.23  ← loss不变
epoch=10 train_total=NaN     val_total=NaN     ← 出现NaN
```

### 处理方法
- **Loss不下降**: 降低learning_rate (0.0003 → 0.0001)
- **出现NaN**: 降低learning_rate，检查数据
- **过拟合**: train↓但val↑，添加dropout或正则化

---

## 📋 实验检查清单

开始实验前:
- [x] 环境配置
- [x] 数据下载
- [x] 代码测试
- [x] 实验配置
- [x] 训练脚本

运行实验中:
- [ ] 监控loss
- [ ] 检查GPU使用
- [ ] 保存检查点

实验完成后:
- [ ] 记录最终指标
- [ ] 保存模型
- [ ] 分析结果

---

## 🎯 总结

### 你还差什么？

**答案是: 什么都不差！** 

你现在就可以开始实验，所有必要的都已经就绪。

### 最小启动命令

```bash
cd /mnt/cpfs/prediction/lipeinan/code
python3 train_experiment.py --num-scenes 3 --epochs 2
```

**就这一行命令，实验就开始跑了！** 🚀

### 下一步

1. **现在**: 跑快速测试（2分钟）
2. **今天**: 跑小实验（20分钟）
3. **本周**: 跑正式实验（数小时）

**不要追求完美，先跑起来再说！**

实验结果比实验工具重要100倍。等你有了结果，再优化工具也不迟。

---

## 📞 需要帮助？

如果遇到问题:
1. 查看输出日志
2. 检查GPU: `nvidia-smi`
3. 验证数据: `python3 test_real_data.py`
4. 查看详细文档: `docs/EXPERIMENT_STARTUP_GUIDE.md`

---

**准备好了吗？现在就运行吧！** 

```bash
bash run_experiment.sh
```
