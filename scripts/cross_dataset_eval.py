"""Cross-dataset Stage-1 evaluation for ranking-reversal diagnostics.

The script loads Stage-1 checkpoints trained on one dataset and evaluates their
K-step imagination metrics on the other dataset's validation split. It is
intentionally eval-only and writes a JSON audit trail.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.evaluation.stage1_metrics import evaluate_stage1
from doorrl.imagination.task_reward import TaskRewardCfg
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch
from doorrl.utils import set_seed


_VARIANT_BY_COND = {
    "wm_object": "object_only",
    "wm_decoupled": "object_relation_decoupled_visibility",
    "wm_decoupled_no_vis": "object_relation_decoupled",
}


def _build_loader(args: argparse.Namespace, config: DoorRLConfig, dataset_name: str):
    if dataset_name == "nuscenes":
        full = NuScenesSceneDataset(
            config=config,
            nuscenes_root=args.nuscenes_root,
            num_scenes=args.num_scenes,
            version="v1.0-trainval",
            cache_dir=args.token_cache_dir or None,
        )
    elif dataset_name == "nuplan":
        full = NuPlanPreprocessedDataset(
            config=config,
            data_root=args.nuplan_root,
            num_samples=args.nuplan_num_samples,
            index_json=args.nuplan_index_json,
            seed=args.seed,
            num_workers=args.nuplan_workers,
            materialize_cache=not args.nuplan_lazy,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    names = sorted(set(full.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * len(shuffled)))) if len(shuffled) > 1 else 0
    val_scenes = set(shuffled[len(shuffled) - n_val:])
    val_idx = full.indices_for_scenes(val_scenes)

    if args.max_val_samples > 0:
        val_idx = val_idx[:args.max_val_samples]

    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(full, val_idx), shuffle=False, **loader_kwargs), len(val_idx)


def _load_model(condition: str, ckpt: Path, config: DoorRLConfig, device: torch.device):
    variant = _VARIANT_BY_COND[condition]
    model = DoorRLModelVariant(config.model, ModelVariant(variant))
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _pairs(args: argparse.Namespace) -> Iterable[Tuple[str, str, int, str, Path]]:
    for seed in args.seeds:
        for condition in args.nuscenes_conditions:
            ckpt = Path(args.nuscenes_stage1_root) / f"seed{seed}" / condition / "model.pt"
            yield "nuscenes", "nuplan", seed, condition, ckpt
        for condition in args.nuplan_conditions:
            ckpt = Path(args.nuplan_stage1_root) / f"seed{seed}" / condition / "model.pt"
            yield "nuplan", "nuscenes", seed, condition, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "cross_dataset_eval"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--nuscenes-root", default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument("--num-scenes", type=int, default=700)
    parser.add_argument("--token-cache-dir", default=str(ROOT / "experiments" / "_token_cache"))
    parser.add_argument("--nuscenes-stage1-root", default=str(ROOT / "experiments" / "stage1_pilot_x"))
    parser.add_argument("--nuscenes-conditions", nargs="+", default=["wm_object", "wm_decoupled"])

    parser.add_argument("--nuplan-root", default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument("--nuplan-index-json", default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"))
    parser.add_argument("--nuplan-workers", type=int, default=32)
    parser.add_argument("--nuplan-lazy", action="store_true")
    parser.add_argument("--loader-workers", type=int, default=32)
    parser.add_argument("--nuplan-stage1-root", default=str(ROOT / "experiments" / "nuplan_stage1_50k"))
    parser.add_argument("--nuplan-conditions", nargs="+", default=["wm_object", "wm_decoupled_no_vis"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    config.seed = args.seed

    loaders: Dict[str, Tuple[DataLoader, int]] = {}
    results: List[Dict] = []
    for source, target, seed, condition, ckpt in _pairs(args):
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}", flush=True)
            continue
        if target not in loaders:
            print(f"[loader] building {target} val loader", flush=True)
            loaders[target] = _build_loader(args, config, target)
        val_loader, n_val = loaders[target]
        print(f"[eval] {source}-> {target} seed={seed} cond={condition}", flush=True)
        model = _load_model(condition, ckpt, config, device)
        metrics = evaluate_stage1(
            model, val_loader, device=device,
            horizon=args.horizon, reward_cfg=TaskRewardCfg(),
        )
        row = {
            "source_dataset": source,
            "target_dataset": target,
            "seed": seed,
            "condition": condition,
            "checkpoint": str(ckpt),
            "target_val_samples": n_val,
            "horizon": args.horizon,
            "metrics": metrics.to_dict(),
        }
        results.append(row)
        out_path = out_dir / f"{source}_to_{target}_seed{seed}_{condition}.json"
        out_path.write_text(json.dumps(row, indent=2))
        print(f"  -> {out_path}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = out_dir / "cross_dataset_eval_all.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
