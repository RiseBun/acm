"""诊断NaN问题"""
import torch
from torch.utils.data import DataLoader
from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models import DoorRLModel
from doorrl.schema import SceneBatch
from doorrl.training.losses import compute_losses

config = DoorRLConfig.from_json("configs/experiment_baseline.json")

# 加载数据
dataset = NuScenesSceneDataset(
    config=config,
    nuscenes_root="/mnt/datasets/e2e-nuscenes/20260302",
    scenes=["scene-0001", "scene-0002"],
    version='v1.0-trainval',
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=SceneBatch.collate,
)

batch = next(iter(loader))

print("=" * 80)
print("诊断NaN问题")
print("=" * 80)

print(f"\n1. 检查输入数据")
print(f"   tokens has NaN: {torch.isnan(batch.tokens).any()}")
print(f"   tokens has Inf: {torch.isinf(batch.tokens).any()}")
print(f"   tokens range: [{batch.tokens[batch.tokens != 0].min():.4f}, {batch.tokens[batch.tokens != 0].max():.4f}]")
print(f"   next_tokens has NaN: {torch.isnan(batch.next_tokens).any()}")
print(f"   actions has NaN: {torch.isnan(batch.actions).any()}")
print(f"   rewards: {batch.rewards}")
print(f"   continues: {batch.continues}")

print(f"\n2. 创建模型")
model = DoorRLModel(config.model)
model.train()

print(f"\n3. 前向传播")
with torch.no_grad():
    output = model(batch)
    
    print(f"   abstraction.selected_tokens has NaN: {torch.isnan(output.abstraction.selected_tokens).any()}")
    print(f"   world_model.predicted_next_tokens has NaN: {torch.isnan(output.world_model.predicted_next_tokens).any()}")
    print(f"   world_model.predicted_reward has NaN: {torch.isnan(output.world_model.predicted_reward).any()}")
    print(f"   world_model.predicted_continue has NaN: {torch.isnan(output.world_model.predicted_continue).any()}")
    print(f"   world_model.predicted_collision has NaN: {torch.isnan(output.world_model.predicted_collision).any()}")
    print(f"   policy.action_mean has NaN: {torch.isnan(output.policy.action_mean).any()}")
    
    print(f"\n   selected_tokens range: [{output.abstraction.selected_tokens.min():.4f}, {output.abstraction.selected_tokens.max():.4f}]")
    print(f"   predicted_next_tokens range: [{output.world_model.predicted_next_tokens.min():.4f}, {output.world_model.predicted_next_tokens.max():.4f}]")
    print(f"   predicted_reward: {output.world_model.predicted_reward}")
    print(f"   predicted_continue: {output.world_model.predicted_continue}")
    print(f"   predicted_collision: {output.world_model.predicted_collision}")

print(f"\n4. 计算损失")
try:
    loss, stats = compute_losses(batch, output, config.training)
    print(f"   Loss computed successfully")
    print(f"   total: {stats['total']}")
    print(f"   obs: {stats['obs']}")
    print(f"   reward: {stats['reward']}")
    print(f"   continue: {stats['continue']}")
    print(f"   collision: {stats['collision']}")
    print(f"   bc: {stats['bc']}")
except Exception as e:
    print(f"   Error computing loss: {e}")

print(f"\n5. 检查selected_mask")
print(f"   selected_mask: {output.abstraction.selected_mask}")
print(f"   selected_mask sum: {output.abstraction.selected_mask.sum(dim=1)}")
print(f"   selected_mask has all False: {(~output.abstraction.selected_mask).all(dim=1).any()}")

print(f"\n6. 检查token_mask")
print(f"   token_mask sum: {batch.token_mask.sum(dim=1)}")
print(f"   token_types: {batch.token_types[0]}")
