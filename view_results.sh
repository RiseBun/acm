#!/bin/bash
# 快速查看训练结果

echo "=========================================="
echo " DOOR-RL Training Results"
echo "=========================================="
echo ""

# 查找最新的训练日志
LATEST_LOG=$(find /tmp -name "train_*.log" -type f 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training logs found!"
    exit 1
fi

echo "📊 Latest training log: $LATEST_LOG"
echo ""

# 提取epoch信息
echo "📈 Training Progress:"
echo "------------------------------------------"
grep "epoch=" "$LATEST_LOG" | tail -10
echo ""

# 统计信息
echo "📋 Training Statistics:"
echo "------------------------------------------"
python3 -c "
import re

log_file = '$LATEST_LOG'
epochs = []
trains = []
vals = []

with open(log_file) as f:
    for line in f:
        m = re.search(r'epoch=(\d+).*?train_total=([\d.]+).*?val_total=([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            trains.append(float(m.group(2)))
            vals.append(float(m.group(3)))

if epochs:
    print(f'Total epochs: {len(epochs)}')
    print(f'Initial train: {trains[0]:.2f}')
    print(f'Final train: {trains[-1]:.2f}')
    print(f'Best val: {min(vals):.2f} @ epoch {epochs[vals.index(min(vals))]}')
    print(f'Train reduction: {(1-trains[-1]/trains[0])*100:.1f}%')
    print(f'Val reduction: {(1-vals[-1]/vals[0])*100:.1f}%')
"

echo ""
echo "📊 Training curve:"
echo "------------------------------------------"
python3 plot_training_curve.py "$LATEST_LOG" 2>/dev/null

echo ""
echo "=========================================="
echo " Done!"
echo "=========================================="
