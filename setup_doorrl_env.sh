#!/bin/bash
# DOOR-RL 环境安装脚本

echo "========================================"
echo "DOOR-RL Conda 环境安装"
echo "========================================"

# 设置conda路径
CONDA_ROOT="/mnt/volumes/cpfs-ares-root/prediction/lipeinan/environments/miniforge3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# 创建环境（如果不存在）
if ! conda env list | grep -q "doorrl"; then
    echo "创建 doorrl 环境..."
    conda create -n doorrl python=3.10 -y
fi

# 激活环境
conda activate doorrl

echo ""
echo "安装 PyTorch (CUDA 12.1)..."
echo "这可能需要 5-10 分钟，请耐心等待..."
echo ""

# 安装 PyTorch (CPU版本，下载更快)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "安装其他依赖..."
pip install numpy matplotlib tqdm pyyaml shapely pyquaternion

echo ""
echo "安装 nuScenes-devkit..."
pip install nuscenes-devkit

echo ""
echo "========================================"
echo "安装完成！验证环境..."
echo "========================================"
python -c "import torch; print(f'torch: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}')"
python -c "import nuscenes; print('nuScenes OK')"

echo ""
echo "激活环境命令:"
echo "  source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate doorrl"
echo ""
echo "启动训练命令:"
echo "  cd /mnt/volumes/cpfs-ares-root/prediction/lipeinan/code"
echo "  python train_large_scale.py --num-scenes 100 --epochs 100"