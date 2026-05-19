"""Interaction-conditioned offline sanity analysis on nuPlan NPZ.

This reuses the Stage-1 checkpoints and validation split from
offline_planner_sanity.py, but reports metrics on interaction-heavy subsets.
It is not a closed-loop evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import TaskRewardCfg
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_CONDITIONS = {
    "wm_object": "object_only",
    "wm_decoupled_no_vis": "object_relation_decoupled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument(
        "--nuplan-root",
        default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split",
    )
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument(
        "--nuplan-index-json",
        default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"),
    )
    parser.add_argument(
        "--stage1-root",
        default=str(ROOT / "experiments" / "nuplan_stage1_50k"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "nuplan_interaction_subset_50k"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["wm_object", "wm_decoupled_no_vis"],
        choices=list(_CONDITIONS),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--low-ttc-threshold", type=float, default=5.0)
    parser.add_argument("--dense-agent-threshold", type=int, default=12)
    parser.add_argument("--rare-dense-threshold", type=int, default=8)
    return parser.parse_args()


def _build_val_loader(args: argparse.Namespace, config: DoorRLConfig, seed: int):
    dataset = NuPlanPreprocessedDataset(
        config=config,
        data_root=args.nuplan_root,
        num_samples=args.nuplan_num_samples,
        index_json=args.nuplan_index_json,
        seed=seed,
        materialize_cache=False,
    )
    names = sorted(set(dataset.cache_scene_names))
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled)))) if len(shuffled) > 1 else 0
    val_scenes = set(shuffled[len(shuffled) - n_val:])
    val_idx = dataset.indices_for_scenes(val_scenes)
    if args.max_val_samples and args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]
    print(
        f"[loader] seed={seed} val_scenes={len(val_scenes)} "
        f"val_samples={len(val_idx)}",
        flush=True,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)


def _subset_masks(batch: SceneBatch, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    mask = batch.token_mask
    types = batch.token_types
    dynamic = (
        (types == int(TokenType.VEHICLE))
        | (types == int(TokenType.PEDESTRIAN))
        | (types == int(TokenType.CYCLIST))
    ) & mask
    rare = ((types == int(TokenType.PEDESTRIAN)) | (types == int(TokenType.CYCLIST))) & mask
    relation = (types == int(TokenType.RELATION)) & mask

    dyn_count = dynamic.sum(dim=1)
    rare_count = rare.sum(dim=1)
    relation_ttc = torch.where(
        relation,
        batch.tokens[:, :, 8],
        torch.full_like(batch.tokens[:, :, 8], 1e6),
    )
    min_ttc = relation_ttc.min(dim=1).values
    lane_conflict = ((batch.tokens[:, :, 9] > 0.5) & relation).any(dim=1)
    low_ttc = min_ttc <= args.low_ttc_threshold
    dense_agents = dyn_count >= args.dense_agent_threshold
    rare_agent_dense = (rare_count >= 1) & (dyn_count >= args.rare_dense_threshold)
    high_interaction = low_ttc | lane_conflict | dense_agents | rare_agent_dense
    return {
        "all_val": torch.ones_like(low_ttc, dtype=torch.bool),
        "low_ttc_proxy": low_ttc,
        "lane_conflict": lane_conflict,
        "dense_agents": dense_agents,
        "rare_agent_dense": rare_agent_dense,
        "high_interaction_union": high_interaction,
    }


def _new_bucket() -> Dict[str, float]:
    return {
        "n": 0.0,
        "teacher_action_mse_sum": 0.0,
        "latent_return_sum": 0.0,
        "imagined_collision_sum": 0.0,
    }


def _add(bucket: Dict[str, float], sample_mask: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> None:
    n = int(sample_mask.sum().item())
    if n <= 0:
        return
    bucket["n"] += n
    bucket["teacher_action_mse_sum"] += float(metrics["action_mse"][sample_mask].sum().item())
    bucket["latent_return_sum"] += float(metrics["return"][sample_mask].sum().item())
    bucket["imagined_collision_sum"] += float(metrics["collision"][sample_mask].sum().item())


@torch.no_grad()
def evaluate_condition(
    args: argparse.Namespace,
    config: DoorRLConfig,
    val_loader,
    seed: int,
    condition: str,
    device: torch.device,
) -> Dict:
    variant = _CONDITIONS[condition]
    model = DoorRLModelVariant(config.model, ModelVariant(variant))
    ckpt = Path(args.stage1_root) / f"seed{seed}" / condition / "model.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    buckets: Dict[str, Dict[str, float]] = defaultdict(_new_bucket)
    for batch in val_loader:
        batch = batch.to(device)
        traj = imagine_trajectory(
            model,
            batch,
            horizon=args.horizon,
            deterministic=True,
            reward_cfg=TaskRewardCfg(),
            detach_world_model=True,
            action_sample_clip=5.0,
        )
        diff = traj.action_means[:, 0, :] - batch.actions
        metrics = {
            "action_mse": diff.pow(2).mean(dim=1).detach().cpu(),
            "return": traj.rewards.sum(dim=1).detach().cpu(),
            "collision": (traj.collisions.max(dim=1).values > 0.5).float().detach().cpu(),
        }
        cpu_batch = SceneBatch(
            tokens=batch.tokens.detach().cpu(),
            token_mask=batch.token_mask.detach().cpu(),
            token_types=batch.token_types.detach().cpu(),
            actions=batch.actions.detach().cpu(),
            next_tokens=batch.next_tokens.detach().cpu(),
            rewards=batch.rewards.detach().cpu(),
            continues=batch.continues.detach().cpu(),
        )
        for name, sample_mask in _subset_masks(cpu_batch, args).items():
            _add(buckets[name], sample_mask, metrics)

    subsets = {}
    for name, bucket in sorted(buckets.items()):
        n = max(bucket["n"], 1.0)
        subsets[name] = {
            "n_samples": int(bucket["n"]),
            "teacher_action_mse": bucket["teacher_action_mse_sum"] / n,
            "latent_return_mean": bucket["latent_return_sum"] / n,
            "imagined_collision_rate": bucket["imagined_collision_sum"] / n,
        }
    return {
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "subsets": subsets,
    }


def _summarize(results: Dict[str, Dict[str, Dict]]) -> Dict:
    summary: Dict[str, Dict] = {}
    metrics = ["teacher_action_mse", "latent_return_mean", "imagined_collision_rate"]
    for condition, by_seed in results.items():
        subset_names = sorted({s for row in by_seed.values() for s in row["subsets"]})
        summary[condition] = {}
        for subset in subset_names:
            rows = [row["subsets"][subset] for row in by_seed.values() if subset in row["subsets"]]
            out = {"n_samples_mean": float(torch.tensor([r["n_samples"] for r in rows]).float().mean().item())}
            for metric in metrics:
                vals = torch.tensor([r[metric] for r in rows], dtype=torch.float32)
                out[f"{metric}_mean"] = float(vals.mean().item())
                out[f"{metric}_std"] = float(vals.std().item()) if len(vals) > 1 else 0.0
            summary[condition][subset] = out
    return summary


def _write_markdown(out_dir: Path, summary: Dict, args: argparse.Namespace) -> None:
    lines = [
        "# nuPlan Interaction-Conditioned Subset Analysis",
        "",
        "Offline validation analysis using existing nuPlan 50k Stage-1 checkpoints.",
        "This is not a closed-loop evaluation.",
        "",
        (
            f"Subsets: low_ttc_proxy <= {args.low_ttc_threshold}s, "
            f"dense_agents >= {args.dense_agent_threshold}, "
            f"rare_agent_dense rare>=1 and dyn>={args.rare_dense_threshold}."
        ),
        "",
    ]
    subset_names = sorted({s for by_subset in summary.values() for s in by_subset})
    for subset in subset_names:
        lines.extend([
            f"## {subset}",
            "",
            "| condition | n/seed | Action MSE ↓ | Return ↑ | Collision ↓ |",
            "|---|---:|---:|---:|---:|",
        ])
        for condition in args.conditions:
            row = summary[condition][subset]
            lines.append(
                f"| {condition} | {row['n_samples_mean']:.0f} | "
                f"{row['teacher_action_mse_mean']:.3f} ± {row['teacher_action_mse_std']:.3f} | "
                f"{row['latent_return_mean_mean']:.3f} ± {row['latent_return_mean_std']:.3f} | "
                f"{row['imagined_collision_rate_mean']:.3f} ± {row['imagined_collision_rate_std']:.3f} |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Dict[str, Dict]] = {condition: {} for condition in args.conditions}

    for seed in args.seeds:
        set_seed(seed)
        val_loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            print(f"[eval] seed={seed} condition={condition}", flush=True)
            result = evaluate_condition(args, config, val_loader, seed, condition, device)
            results[condition][str(seed)] = result
            cond_dir = out_dir / f"seed{seed}" / condition
            cond_dir.mkdir(parents=True, exist_ok=True)
            (cond_dir / "subset_metrics.json").write_text(json.dumps(result, indent=2))

    summary = _summarize(results)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    _write_markdown(out_dir, summary, args)


if __name__ == "__main__":
    main()
