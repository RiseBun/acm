#!/bin/bash
# DOOR-RL 实验快速启动脚本

echo "=========================================="
echo " DOOR-RL Experiment Launcher"
echo "=========================================="
echo ""

# 检查环境
echo "📋 Checking environment..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    exit 1
fi

# 激活环境
echo "🔧 Activating conda environment..."
conda activate find_physics_zone

# 检查数据
echo "📊 Checking data..."
NUSCENES_ROOT="/mnt/datasets/e2e-nuscenes/20260302"
if [ ! -d "$NUSCENES_ROOT" ]; then
    echo "❌ nuScenes data not found at $NUSCENES_ROOT"
    exit 1
fi
echo "✓ nuScenes data found"

# 创建输出目录
echo "📁 Creating experiment directories..."
mkdir -p experiments/runs
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
echo "✓ Directories created"

echo ""
echo "=========================================="
echo " Choose an experiment to run:"
echo "=========================================="
echo ""
echo "1) Quick test (3 scenes, 2 epochs) - 2 min"
echo "2) Small experiment (10 scenes, 10 epochs) - 10 min"
echo "3) Baseline experiment (10 scenes, 20 epochs) - 20 min"
echo "4) Custom experiment"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running quick test..."
        python3 train_experiment.py \
            --config configs/debug_mvp.json \
            --nuscenes-root $NUSCENES_ROOT \
            --num-scenes 3 \
            --epochs 2 \
            --output-dir experiments/runs
        ;;
    2)
        echo ""
        echo "🚀 Running small experiment..."
        python3 train_experiment.py \
            --config configs/experiment_baseline.json \
            --nuscenes-root $NUSCENES_ROOT \
            --num-scenes 10 \
            --epochs 10 \
            --output-dir experiments/runs
        ;;
    3)
        echo ""
        echo "🚀 Running baseline experiment..."
        python3 train_experiment.py \
            --config configs/experiment_baseline.json \
            --nuscenes-root $NUSCENES_ROOT \
            --num-scenes 10 \
            --epochs 20 \
            --output-dir experiments/runs
        ;;
    4)
        echo ""
        read -p "Config file path: " config_path
        read -p "Number of scenes: " num_scenes
        read -p "Number of epochs: " epochs
        
        python3 train_experiment.py \
            --config $config_path \
            --nuscenes-root $NUSCENES_ROOT \
            --num-scenes $num_scenes \
            --epochs $epochs \
            --output-dir experiments/runs
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " Experiment completed!"
echo "=========================================="
echo ""
echo "📊 Results saved in: experiments/runs/<experiment_name>/"
echo "📈 View training logs: ls experiments/runs/*/logs/"
echo "💾 View checkpoints: ls experiments/runs/*/checkpoints/"
echo ""
