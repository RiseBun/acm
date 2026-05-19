#!/bin/bash
# 批量运行消融实验 - 对比所有表示变体

set -e

# 配置
NUSCENES_ROOT="/mnt/datasets/e2e-nuscenes/20260302"
CONFIG="configs/debug_mvp.json"
NUM_SCENES=20
EPOCHS=30
BATCH_SIZE=8
OUTPUT_DIR="experiments/ablation"

echo "=========================================="
echo "DOOR-RL Ablation Study - All Variants"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  NuScenes root: $NUSCENES_ROOT"
echo "  Num scenes:    $NUM_SCENES"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Output dir:    $OUTPUT_DIR"
echo ""

# 定义所有变体
VARIANTS=("holistic" "object_only" "object_relation" "object_relation_visibility")

# 运行每个变体
for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running variant: $VARIANT"
    echo "=========================================="
    
    python3 train_ablation.py \
        --config "$CONFIG" \
        --nuscenes-root "$NUSCENES_ROOT" \
        --variant "$VARIANT" \
        --num-scenes "$NUM_SCENES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --seed 7
    
    echo ""
    echo "✓ Completed: $VARIANT"
    echo ""
done

echo ""
echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -la $OUTPUT_DIR"
echo "  cat $OUTPUT_DIR/*/config.json"
echo ""
