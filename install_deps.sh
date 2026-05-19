#!/bin/bash
# 安装 DOOR-RL 依赖到 doorrl 环境

echo "========================================"
echo "安装 DOOR-RL 依赖包"
echo "========================================"

# 激活 conda
source /mnt/volumes/cpfs-ares-root/prediction/lipeinan/environments/miniforge3/etc/profile.d/conda.sh
conda activate doorrl

echo ""
echo "Python 版本:"
python --version
echo ""

echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "安装其他依赖..."
pip install numpy matplotlib tqdm pyyaml shapely pyquaternion nuscenes-devkit tensorboard pytest black flake8

echo ""
echo "========================================"
echo "验证安装..."
echo "========================================"

python -c "
import torch
import numpy
import matplotlib
import yaml
import tqdm
print('✅ 核心依赖安装成功')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {numpy.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
"

echo ""
echo "安装完成！"
