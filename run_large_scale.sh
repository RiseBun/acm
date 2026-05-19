#!/bin/bash
# 大规模训练启动脚本

set -e

echo "=========================================="
echo "DOOR-RL 大规模训练启动器"
echo "=========================================="

# 默认值
NUM_SCENES=100
EPOCHS=100
BATCH_SIZE=32
VARIANT="object_relation"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes)
            NUM_SCENES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --scenes N       场景数量 (默认: 100)"
            echo "  --epochs N       训练轮数 (默认: 100)"
            echo "  --batch-size N   批次大小 (默认: 32)"
            echo "  --variant TYPE   模型变体 (默认: object_relation)"
            echo "                   可选: holistic, object_only, object_relation, full"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 激活conda环境
echo "激活conda环境..."
source /opt/conda/etc/profile.d/conda.sh
conda activate find_physics_zone

# 设置路径
CODE_DIR="/mnt/volumes/cpfs-ares-root/prediction/lipeinan/code"
NUSCENES_ROOT="/mnt/datasets/e2e-nuscenes/20260302"

cd "$CODE_DIR"

# 创建输出目录
mkdir -p experiments/large_runs

# 打印配置
echo ""
echo "训练配置:"
echo "  场景数量: $NUM_SCENES"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  模型变体: $VARIANT"
echo "  数据路径: $NUSCENES_ROOT"
echo ""

# 运行训练
echo "开始训练..."
nohup python3 train_large_scale.py \
    --config configs/experiment_large_scale.json \
    --nuscenes-root "$NUSCENES_ROOT" \
    --num-scenes "$NUM_SCENES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --variant "$VARIANT" \
    --num-workers 4 \
    > experiments/large_runs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo "训练进程已启动 (PID: $TRAIN_PID)"
echo "日志文件: experiments/large_runs/training_*.log"
echo ""
echo "查看训练进度: tail -f experiments/large_runs/training_*.log"
echo "停止训练: kill $TRAIN_PID"