"""Offline planner-like sanity check on nuPlan preprocessed NPZ.

This is deliberately *not* a closed-loop evaluation. It reuses the held-out
nuPlan NPZ split and Stage-1 checkpoints to ask whether the learned policy
looks more planner-like under offline supervision and short imagined rollouts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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
from doorrl.schema import SceneBatch
from doorrl.utils import set_seed


_CONDITIONS = {
    "bc": "object_only",
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
        default=str(ROOT / "experiments" / "nuplan_planner_sanity_50k"),
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
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Optional cap for quick debugging. 0 evaluates the full val split.",
    )
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
        batch_size=config.training.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)


def _stability_score(latents: torch.Tensor) -> torch.Tensor:
    a = latents[:, :-1, :]
    b = latents[:, 1:, :]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-3)
    return (1.0 - cos).mean(dim=1)


def _cat_mean(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    if not xs:
        return float("nan")
    return float(torch.cat(xs).mean().item())


def _cat_std(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    if not xs:
        return float("nan")
    return float(torch.cat(xs).std().item())


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

    action_mse: List[torch.Tensor] = []
    action_mae: List[torch.Tensor] = []
    action_l2: List[torch.Tensor] = []
    teacher_l2: List[torch.Tensor] = []
    action_delta_l2: List[torch.Tensor] = []
    returns: List[torch.Tensor] = []
    coll_max: List[torch.Tensor] = []
    coll_mean_step: List[torch.Tensor] = []
    stability: List[torch.Tensor] = []
    progress_proxy: List[torch.Tensor] = []
    n = 0

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
        mean0 = traj.action_means[:, 0, :]
        teacher = batch.actions
        diff = mean0 - teacher

        action_mse.append(diff.pow(2).mean(dim=1))
        action_mae.append(diff.abs().mean(dim=1))
        action_l2.append(mean0.norm(dim=1))
        teacher_l2.append(teacher.norm(dim=1))
        action_delta_l2.append(diff.norm(dim=1))
        returns.append(traj.rewards.sum(dim=1))
        cmax = traj.collisions.max(dim=1).values
        coll_max.append(cmax)
        coll_mean_step.append(traj.collisions.mean(dim=1))
        stability.append(_stability_score(traj.ego_latents))
        # Planner-like progress proxy: mean deterministic longitudinal action.
        progress_proxy.append(traj.action_means[:, :, 0].mean(dim=1))
        n += batch.tokens.size(0)

    result = {
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "n_samples": n,
        "metrics": {
            "teacher_action_mse": _cat_mean(action_mse),
            "teacher_action_mae": _cat_mean(action_mae),
            "teacher_action_delta_l2": _cat_mean(action_delta_l2),
            "policy_action_l2": _cat_mean(action_l2),
            "teacher_action_l2": _cat_mean(teacher_l2),
            "latent_return_mean": _cat_mean(returns),
            "latent_return_std": _cat_std(returns),
            "imagined_collision_rate": _cat_mean([(torch.cat(coll_max) > 0.5).float()]),
            "collision_mean": _cat_mean(coll_max),
            "collision_step_mean": _cat_mean(coll_mean_step),
            "rollout_stability": _cat_mean(stability),
            "ego_progress_proxy": _cat_mean(progress_proxy),
            "horizon": args.horizon,
        },
    }
    return result


def _summarize(results: Dict[str, Dict[str, Dict]]) -> Dict:
    summary: Dict[str, Dict] = {}
    metric_names = [
        "teacher_action_mse",
        "teacher_action_mae",
        "teacher_action_delta_l2",
        "policy_action_l2",
        "teacher_action_l2",
        "latent_return_mean",
        "imagined_collision_rate",
        "collision_mean",
        "collision_step_mean",
        "rollout_stability",
        "ego_progress_proxy",
    ]
    for condition, by_seed in results.items():
        rows = list(by_seed.values())
        summary[condition] = {
            "seeds": rows,
            "mean": {},
            "std_across_seeds": {},
        }
        for name in metric_names:
            vals = torch.tensor([r["metrics"][name] for r in rows], dtype=torch.float32)
            summary[condition]["mean"][name] = float(vals.mean().item())
            summary[condition]["std_across_seeds"][name] = (
                float(vals.std().item()) if len(vals) > 1 else 0.0
            )
    return summary


def _write_markdown(out_dir: Path, summary: Dict, args: argparse.Namespace) -> None:
    lines = [
        "# nuPlan 50k Offline Planner-Like Sanity Check",
        "",
        "这是一个更下游的 offline planner-like sanity check，不是正式 external closed-loop evaluation 的替代品。",
        "",
        (
            f"设置：nuPlan 50k val split，seeds {args.seeds}，"
            f"conditions {args.conditions}，horizon={args.horizon}，"
            f"lazy loading + {args.loader_workers} DataLoader workers。"
        ),
        "",
        "| condition | Teacher action MSE ↓ | Action ΔL2 ↓ | Policy action L2 | Return ↑ | CollRate ↓ | CollMean ↓ | Stability | Progress proxy ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, block in summary.items():
        mean = block["mean"]
        std = block["std_across_seeds"]
        lines.append(
            f"| {condition} | "
            f"{mean['teacher_action_mse']:.3f} ± {std['teacher_action_mse']:.3f} | "
            f"{mean['teacher_action_delta_l2']:.3f} ± {std['teacher_action_delta_l2']:.3f} | "
            f"{mean['policy_action_l2']:.3f} | "
            f"{mean['latent_return_mean']:.3f} ± {std['latent_return_mean']:.3f} | "
            f"{mean['imagined_collision_rate']:.3f} ± {std['imagined_collision_rate']:.3f} | "
            f"{mean['collision_mean']:.3f} ± {std['collision_mean']:.3f} | "
            f"{mean['rollout_stability']:.3f} | "
            f"{mean['ego_progress_proxy']:.3f} |"
        )
    lines.extend([
        "",
        "Per-seed raw view:",
        "",
    ])
    for condition, block in summary.items():
        lines.append(f"## {condition}")
        lines.append("")
        lines.append("| seed | action MSE | Return | CollRate | CollMean | Stability | Progress |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in block["seeds"]:
            m = row["metrics"]
            lines.append(
                f"| {row['seed']} | {m['teacher_action_mse']:.3f} | "
                f"{m['latent_return_mean']:.3f} | "
                f"{m['imagined_collision_rate']:.3f} | "
                f"{m['collision_mean']:.3f} | "
                f"{m['rollout_stability']:.3f} | "
                f"{m['ego_progress_proxy']:.3f} |"
            )
        lines.append("")
    lines.extend([
        "## Reading",
        "",
        "- 这个实验只能作为 downstream offline evidence；它没有 reactive agents，也没有 simulator。",
        "- 主 planner-like 指标是 teacher-derived action MSE；安全性仍通过 imagined collision 近似观察。",
        "- 如果结果支持某个条件，只能说明它在 offline planner-like probes 下更合理，不能声称已经通过正式闭环验证。",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Dict[str, Dict]] = {c: {} for c in args.conditions}
    for seed in args.seeds:
        print("\n" + "=" * 90, flush=True)
        print(f"[sanity] seed={seed}", flush=True)
        set_seed(seed)
        config.seed = seed
        val_loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            print(f"[sanity] seed={seed} condition={condition}", flush=True)
            result = evaluate_condition(args, config, val_loader, seed, condition, device)
            results[condition][str(seed)] = result
            exp_dir = out_dir / f"seed{seed}" / condition
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "planner_sanity.json").write_text(json.dumps(result, indent=2))
            print(f"[sanity] {condition}: {result['metrics']}", flush=True)

    summary = {
        "note": "offline planner-like sanity check; not closed-loop evaluation",
        "setup": vars(args),
        "summary": _summarize(results),
        "raw": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out_dir, summary["summary"], args)
    print(f"\nwrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
