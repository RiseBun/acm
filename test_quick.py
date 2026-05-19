#!/usr/bin/env python3
"""快速测试脚本 - 验证数据加载和模型"""
import sys
from pathlib import Path

# 添加src到路径
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig

print("✓ 导入doorrl成功")

# 加载配置
config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
print(f"✓ 配置加载成功: {config.model.model_dim}d model")

# 尝试导入数据集
try:
    from doorrl.data.real_dataset import NuScenesSceneDataset
    print("✓ 数据集模块导入成功")
    
    # 尝试加载数据
    print("\n尝试加载nuScenes数据...")
    ds = NuScenesSceneDataset(
        config=config,
        nuscenes_root='/mnt/datasets/e2e-nuscenes/20260302',
        scenes=None,
        version='v1.0-trainval',
    )
    print(f"✓ 数据集加载成功: {len(ds)} samples")
    
except Exception as e:
    print(f"✗ 数据集加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
