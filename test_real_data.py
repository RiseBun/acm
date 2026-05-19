"""测试真实nuScenes数据pipeline"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch.utils.data import DataLoader

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models import DoorRLModel
from doorrl.schema import SceneBatch


def test_nuscenes_real_data_pipeline() -> None:
    """测试nuScenes真实数据pipeline"""
    print("=" * 80)
    print("Testing NuScenes Real Data Pipeline")
    print("=" * 80)
    
    # 加载配置
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    
    # nuScenes数据路径
    nuscenes_root = "/mnt/datasets/e2e-nuscenes/20260302"
    
    print(f"\n1. Loading dataset from {nuscenes_root}...")
    dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=nuscenes_root,
        scenes=["scene-0001", "scene-0002"],  # 使用前2个场景测试
        version='v1.0-trainval',
    )
    
    print(f"   Dataset size: {len(dataset)} samples")
    assert len(dataset) > 0, "Dataset should not be empty"
    
    print(f"\n2. Testing data loading...")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=SceneBatch.collate,
    )
    
    batch = next(iter(loader))
    print(f"   Batch tokens shape: {batch.tokens.shape}")
    print(f"   Batch token_mask shape: {batch.token_mask.shape}")
    print(f"   Batch token_types shape: {batch.token_types.shape}")
    print(f"   Batch actions shape: {batch.actions.shape}")
    
    # 验证形状
    assert batch.tokens.ndim == 3, "tokens must have shape [B, S, D]"
    assert batch.token_mask.shape == batch.tokens.shape[:2], "token_mask shape mismatch"
    assert batch.actions.ndim == 2, "actions must have shape [B, A]"
    
    print(f"\n3. Testing model forward pass...")
    model = DoorRLModel(config.model)
    output = model(batch)
    
    print(f"   Abstraction selected_tokens shape: {output.abstraction.selected_tokens.shape}")
    print(f"   World model predicted_next_tokens shape: {output.world_model.predicted_next_tokens.shape}")
    print(f"   Policy action_mean shape: {output.policy.action_mean.shape}")
    
    # 验证输出形状
    assert output.abstraction.selected_tokens.shape[:2] == (2, config.model.top_k)
    assert output.world_model.predicted_next_tokens.shape[0] == 2
    assert output.policy.action_mean.shape == (2, config.model.action_dim)
    
    print(f"\n4. Testing scene sequence extraction...")
    sequence = dataset.get_scene_sequence("scene-0001")
    print(f"   Scene-0001 sequence length: {len(sequence)} frames")
    assert len(sequence) > 0, "Scene sequence should not be empty"
    
    print(f"\n{'=' * 80}")
    print("✓ All tests passed!")
    print(f"{'=' * 80}")


def test_nuscenes_adapter_directly() -> None:
    """直接测试nuScenes Adapter"""
    print("\n" + "=" * 80)
    print("Testing NuScenes Adapter Directly")
    print("=" * 80)
    
    from doorrl.adapters.base import TokenizationSpec
    from doorrl.adapters.nuscenes_real_adapter import NuScenesRealDataAdapter
    
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    
    spec = TokenizationSpec(
        raw_dim=config.model.raw_dim,
        max_tokens=config.model.max_tokens,
        max_dynamic_objects=config.data.max_dynamic_objects,
        max_map_tokens=config.data.max_map_tokens,
        max_relation_tokens=config.data.max_relation_tokens,
        action_dim=config.model.action_dim,
    )
    
    nuscenes_root = "/mnt/datasets/e2e-nuscenes/20260302"
    
    print(f"\n1. Initializing adapter...")
    adapter = NuScenesRealDataAdapter(
        spec=spec,
        nuscenes_root=nuscenes_root,
        version='v1.0-trainval',
        use_can_bus=False,  # 暂时不使用CAN总线
    )
    
    print(f"\n2. Loading scene samples...")
    samples = adapter.get_scene_samples("scene-0001")
    print(f"   Scene-0001 has {len(samples)} samples")
    assert len(samples) > 0, "Should have samples"
    
    print(f"\n3. Converting first sample...")
    sample = samples[0]
    scene_item = adapter.convert_sample_to_scene_item(sample, compute_relations=True)
    
    print(f"   Tokens shape: {scene_item['tokens'].shape}")
    print(f"   Token mask sum: {scene_item['token_mask'].sum()}")
    print(f"   Token types unique: {scene_item['token_types'].unique()}")
    print(f"   Actions: {scene_item['actions']}")
    
    # 验证
    assert scene_item['tokens'].shape == (config.model.max_tokens, config.model.raw_dim)
    assert scene_item['token_mask'].sum() > 0, "Should have some valid tokens"
    
    print(f"\n{'=' * 80}")
    print("✓ Adapter test passed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # 先测试adapter
    test_nuscenes_adapter_directly()
    
    # 再测试完整pipeline
    test_nuscenes_real_data_pipeline()
