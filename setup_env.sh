#!/bin/bash
# DOOR-RL 环境设置和验证脚本

echo "=========================================="
echo "DOOR-RL 环境设置"
echo "=========================================="

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate doorrl

echo ""
echo "1. 检查Python环境..."
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "2. 检查数据集..."
if [ -d "/mnt/datasets/e2e-nuscenes/20260302" ]; then
    echo "nuScenes数据集: ✓ 存在"
else
    echo "nuScenes数据集: ✗ 不存在"
fi

echo ""
echo "3. 运行快速测试..."
python -c "
import sys
sys.path.insert(0, 'src')
from doorrl.config import DoorRLConfig
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
import torch

print('Testing model creation...')
config = DoorRLConfig()
model = DoorRLModelVariant(config.model, ModelVariant.OBJECT_RELATION)
print(f'Model created: ✓')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

print('Testing forward pass...')
batch_size = 2
tokens = torch.randn(batch_size, 97, 40).cuda()
token_mask = torch.ones(batch_size, 97, dtype=torch.bool).cuda()
token_types = torch.zeros(batch_size, 97, dtype=torch.long).cuda()
actions = torch.randn(batch_size, 2).cuda()
next_tokens = torch.randn_like(tokens)
rewards = torch.randn(batch_size)
continues = torch.ones(batch_size)

from doorrl.schema import SceneBatch
batch = SceneBatch(
    tokens=tokens,
    token_mask=token_mask,
    token_types=token_types,
    actions=actions,
    next_tokens=next_tokens,
    rewards=rewards,
    continues=continues,
)

output = model(batch)
print(f'Forward pass: ✓')
print(f'Output shape: obs={output.world_model.predicted_next_tokens.shape}')
"

echo ""
echo "=========================================="
echo "环境验证完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 运行Stage 0实验:"
echo "   bash run_stage0_table3.sh"
echo ""
echo "2. 或运行快速测试:"
echo "   python run_stage0_table3.py --variant object_relation --num-scenes 5 --epochs 5"
echo ""
echo "3. 查看实验指南:"
echo "   cat STAGE0_GUIDE.md"
