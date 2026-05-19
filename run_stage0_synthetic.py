#!/usr/bin/env python3
"""
Stage 0 快速验证 - 使用合成数据
验证训练流程是否正常，无需等待真实数据加载
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.synthetic import SyntheticDrivingDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch
from doorrl.training import DoorRLTrainer
from doorrl.utils import set_seed
from doorrl.evaluation.table3_metrics import Table3Metrics, evaluate_stage0


def main():
    print("="*80)
    print("Stage 0: 快速验证 (合成数据)")
    print("="*80)
    
    # 1. 加载配置
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    config.training.epochs = 2  # 快速测试
    config.training.batch_size = 8
    config.seed = 7
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n设备: {device}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    
    # 2. 创建合成数据集
    print("\n创建合成数据集...")
    train_dataset = SyntheticDrivingDataset(config, size=64, seed=7)
    val_dataset = SyntheticDrivingDataset(config, size=16, seed=42)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=SceneBatch.collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=0,
    )
    
    print(f"✓ 训练集: {len(train_dataset)} samples")
    print(f"✓ 验证集: {len(val_dataset)} samples")
    
    # 3. 测试所有4个变体
    variants = [
        ("holistic", "Holistic"),
        ("object_only", "Object-only"),
        ("object_relation", "Object + Relation"),
        ("object_relation_visibility", "Obj+Rel+Visibility"),
    ]
    
    results = {}
    
    for variant_key, variant_name in variants:
        print(f"\n{'='*80}")
        print(f"训练变体: {variant_name}")
        print(f"{'='*80}")
        
        # 创建模型
        variant = ModelVariant(variant_key)
        model = DoorRLModelVariant(config.model, variant)
        model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数: {num_params:,}")
        
        # 训练
        trainer = DoorRLTrainer(
            model=model,
            config=config.training,
            device=device,
        )
        
        print(f"\n开始训练...")
        trainer.fit(train_loader, val_loader=val_loader)
        
        # 评估
        print(f"\n评估Table 3指标...")
        metrics = evaluate_stage0(
            model=model,
            data_loader=val_loader,
            variant_name=variant_key,
            device=device,
            verbose=True,
        )
        
        table3_results = metrics.compute_table3()
        results[variant_key] = table3_results
        
        print(f"\n✓ {variant_name} 完成")
        print(f"  Rollout Error: {table3_results['rollout_error']:.4f}")
        print(f"  Reward Error: {table3_results['reward_error']:.4f}")
        print(f"  Collision Acc: {table3_results['collision_accuracy']:.4f}")
        print(f"  Rare Recall: {table3_results['rare_agent_recall']:.4f}")
    
    # 4. 打印完整Table 3
    print(f"\n{'='*80}")
    print("Table 3: Representation Sufficiency Ablation (Complete)")
    print(f"{'='*80}\n")
    
    print(f"{'Variant':<25} | {'Rollout Error ↓':<18} | {'Reward Error ↓':<18} | {'Collision Acc. ↑':<18} | {'Rare Recall ↑':<15}")
    print("-"*100)
    
    for variant_key, variant_name in variants:
        if variant_key in results:
            r = results[variant_key]
            print(
                f"{variant_name:<25} | "
                f"{r['rollout_error']:.4f} ± {r['rollout_error_std']:.4f}  | "
                f"{r['reward_error']:.4f} ± {r['reward_error_std']:.4f}  | "
                f"{r['collision_accuracy']:.4f}             | "
                f"{r['rare_agent_recall']:.4f}"
            )
    
    print("-"*100)
    
    # 5. 生成LaTeX表格
    print("\nLaTeX Table:")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Representation Sufficiency Ablation Study (Synthetic Data)}")
    print(r"\label{tab:representation_sufficiency_synthetic}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Variant} & \textbf{Rollout Error} $\downarrow$ & \textbf{Reward Error} $\downarrow$ & \textbf{Collision Acc.} $\uparrow$ & \textbf{Rare Recall} $\uparrow$ \\")
    print(r"\midrule")
    
    for variant_key, variant_name in variants:
        if variant_key in results:
            r = results[variant_key]
            print(
                f"{variant_name} & "
                f"{r['rollout_error']:.4f} $\\pm$ {r['rollout_error_std']:.4f} & "
                f"{r['reward_error']:.4f} $\\pm$ {r['reward_error_std']:.4f} & "
                f"{r['collision_accuracy']:.4f} & "
                f"{r['rare_agent_recall']:.4f} \\\\"
            )
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    print(f"\n{'='*80}")
    print("✓ 快速验证完成!")
    print(f"{'='*80}")
    print("\n下一步:")
    print("1. 如果训练流程正常，可以运行真实数据实验")
    print("2. 运行命令: bash run_stage0_complete.sh")
    print("3. 或在后台运行: nohup python3 run_stage0_table3.py --variant all --num-scenes 20 --epochs 30 > stage0.log 2>&1 &")


if __name__ == "__main__":
    main()
