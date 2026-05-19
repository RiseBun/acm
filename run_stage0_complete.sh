#!/bin/bash
# Stage 0 实验完整运行脚本

echo "=========================================="
echo " Stage 0: Table 3 表示充分性消融实验"
echo "=========================================="
echo ""

# 设置路径
CODE_DIR="/mnt/cpfs/prediction/lipeinan/code"
cd "$CODE_DIR"

# 激活conda环境
echo "🔧 激活conda环境..."
source /opt/conda/etc/profile.d/conda.sh
conda activate find_physics_zone

# 检查数据
NUSCENES_ROOT="/mnt/datasets/e2e-nuscenes/20260302"
if [ ! -d "$NUSCENES_ROOT" ]; then
    echo "❌ nuScenes数据未找到: $NUSCENES_ROOT"
    exit 1
fi
echo "✓ nuScenes数据已找到"

# 创建输出目录
echo "📁 创建实验目录..."
mkdir -p experiments/table3_representation_sufficiency
echo "✓ 目录已创建"

echo ""
echo "=========================================="
echo " 开始运行实验"
echo "=========================================="
echo ""

# 运行所有4个变体
echo "🚀 运行Stage 0实验 (所有4个变体)..."
echo "   - 场景数: 3 (快速测试)"
echo "   - Epochs: 2 (快速测试)"
echo "   - Batch size: 8"
echo ""

python3 run_stage0_table3.py \
    --config configs/debug_mvp.json \
    --nuscenes-root "$NUSCENES_ROOT" \
    --variant all \
    --num-scenes 3 \
    --epochs 2 \
    --batch-size 8 \
    --output-dir experiments/table3_representation_sufficiency \
    --seed 7

echo ""
echo "=========================================="
echo " 实验完成!"
echo "=========================================="
echo ""
echo "📊 结果保存在: experiments/table3_representation_sufficiency/"
echo "📈 查看完整结果: cat experiments/table3_representation_sufficiency/table3_complete.json"
echo ""
