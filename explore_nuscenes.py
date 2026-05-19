"""探索nuScenes数据结构"""
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import json

# 初始化nuScenes (使用mini版本进行探索)
nuscenes_root = "/mnt/datasets/e2e-nuscenes/20260302"
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)

print(f"\n=== nuScenes 数据集信息 ===")
print(f"场景数量: {len(nusc.scene)}")
print(f"样本数量: {len(nusc.sample)}")
print(f"样本数据数量: {len(nusc.sample_data)}")
print(f"标注数量: {len(nusc.sample_annotation)}")

# 查看第一个场景
print(f"\n=== 第一个场景 ===")
scene = nusc.scene[0]
print(json.dumps(scene, indent=2, default=str))

# 查看第一个样本
print(f"\n=== 第一个样本 ===")
sample = nusc.sample[0]
print(f"Sample token: {sample['token']}")
print(f"场景token: {sample['scene_token']}")
print(f"时间戳: {sample['timestamp']}")
print(f"数据键: {list(sample['data'].keys())}")

# 查看标注
print(f"\n=== 第一个样本的标注 ===")
sample_ann_token = sample['anns']
print(f"标注数量: {len(sample_ann_token)}")
if len(sample_ann_token) > 0:
    # 找到对应的annotation
    for ann in nusc.sample_annotation:
        if ann['token'] == sample_ann_token[0]:
            print(f"标注token: {ann['token']}")
            print(f"类别: {ann['category_name']}")
            print(f"位置: {ann['translation']}")
            print(f"速度: {ann.get('velocity', 'N/A')}")
            print(f"尺寸: {ann['size']}")
            break

# 查看CAN总线数据
print(f"\n=== CAN总线数据 ===")
try:
    can_bus = NuScenesCanBus(dataroot=nuscenes_root)
    print(f"CAN场景数量: {len(can_bus.can_blacklist)}")
    print("CAN总线可用")
except Exception as e:
    print(f"CAN总线不可用: {e}")

# 查看ego_pose
print(f"\n=== Ego Pose 示例 ===")
sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
print(f"Ego位置: {ego_pose['translation']}")
print(f"Ego旋转: {ego_pose['rotation']}")
