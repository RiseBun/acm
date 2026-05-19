"""Evaluation-time relation feature-group ablation on nuPlan Stage1 checkpoints.

This is a lightweight appendix probe: it does not retrain models. Instead it
loads trained relation-aware policies and masks selected relation-token feature
groups on the held-out nuPlan split before deterministic imagination.
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

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import TaskRewardCfg
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_CONDITIONS = {
    "wm_decoupled_no_vis": "object_relation_decoupled",
    "wm_decoupled": "object_relation_decoupled_visibility",
}

_ABLATIONS = {
    "none": (),
    "no_ttc_risk": (6, 8),          # risk, TTC
    "no_lane_priority": (9, 10),    # lane_conflict, priority
    "no_relation_semantics": (6, 8, 9, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuplan-root", default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument("--nuplan-index-json", default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"))
    parser.add_argument("--stage1-root", default=str(ROOT / "experiments" / "nuplan_stage1_50k"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "nuplan_relation_feature_ablation_50k"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--conditions", nargs="+", default=["wm_decoupled_no_vis"], choices=list(_CONDITIONS))
    parser.add_argument("--ablations", nargs="+", default=list(_ABLATIONS), choices=list(_ABLATIONS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-val-samples", type=int, default=0)
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
    if args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]
    print(f"[loader] seed={seed} val_samples={len(val_idx)}", flush=True)
    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)


def _mask_batch(batch: SceneBatch, ablation: str) -> SceneBatch:
    indices = _ABLATIONS[ablation]
    if not indices:
        return batch
    rel_mask = batch.token_types == int(TokenType.RELATION)
    tokens = batch.tokens.clone()
    next_tokens = batch.next_tokens.clone()
    for idx in indices:
        tokens[..., idx] = torch.where(rel_mask, torch.zeros_like(tokens[..., idx]), tokens[..., idx])
        next_tokens[..., idx] = torch.where(rel_mask, torch.zeros_like(next_tokens[..., idx]), next_tokens[..., idx])
    return SceneBatch(
        tokens=tokens,
        token_mask=batch.token_mask,
        token_types=batch.token_types,
        actions=batch.actions,
        next_tokens=next_tokens,
        rewards=batch.rewards,
        continues=batch.continues,
    )


def _cat_mean(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    return float(torch.cat(xs).mean().item()) if xs else float("nan")


def _cat_std(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    return float(torch.cat(xs).std().item()) if xs else float("nan")


def _stability_score(latents: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(latents[:, :-1], latents[:, 1:], dim=-1, eps=1e-3)
    return (1.0 - cos).mean(dim=1)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, args: argparse.Namespace, ablation: str) -> Dict[str, float]:
    returns: List[torch.Tensor] = []
    coll_max: List[torch.Tensor] = []
    coll_step: List[torch.Tensor] = []
    action_mse: List[torch.Tensor] = []
    action_delta_l2: List[torch.Tensor] = []
    stability: List[torch.Tensor] = []
    n = 0
    for batch in loader:
        batch = _mask_batch(batch, ablation).to(device)
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
        action_mse.append(diff.pow(2).mean(dim=1))
        action_delta_l2.append(diff.norm(dim=1))
        returns.append(traj.rewards.sum(dim=1))
        cmax = traj.collisions.max(dim=1).values
        coll_max.append(cmax)
        coll_step.append(traj.collisions.mean(dim=1))
        stability.append(_stability_score(traj.ego_latents))
        n += batch.tokens.size(0)
    return {
        "n_samples": n,
        "teacher_action_mse": _cat_mean(action_mse),
        "teacher_action_delta_l2": _cat_mean(action_delta_l2),
        "latent_return_mean": _cat_mean(returns),
        "latent_return_std": _cat_std(returns),
        "imagined_collision_rate": _cat_mean([(torch.cat(coll_max) > 0.5).float()]),
        "collision_mean": _cat_mean(coll_max),
        "collision_step_mean": _cat_mean(coll_step),
        "rollout_stability": _cat_mean(stability),
    }


def _summarize(raw: Dict[str, Dict[str, Dict[str, Dict]]]) -> Dict:
    summary: Dict[str, Dict] = {}
    metric_names = [
        "teacher_action_mse",
        "teacher_action_delta_l2",
        "latent_return_mean",
        "imagined_collision_rate",
        "collision_mean",
        "collision_step_mean",
        "rollout_stability",
    ]
    for condition, by_ablation in raw.items():
        summary[condition] = {}
        for ablation, by_seed in by_ablation.items():
            rows = list(by_seed.values())
            summary[condition][ablation] = {"mean": {}, "std_across_seeds": {}, "seeds": rows}
            for name in metric_names:
                vals = torch.tensor([row[name] for row in rows], dtype=torch.float32)
                summary[condition][ablation]["mean"][name] = float(vals.mean().item())
                summary[condition][ablation]["std_across_seeds"][name] = (
                    float(vals.std().item()) if len(vals) > 1 else 0.0
                )
    return summary


def _write_markdown(out_dir: Path, summary: Dict, args: argparse.Namespace) -> None:
    lines = [
        "# nuPlan relation feature-group ablation",
        "",
        "Evaluation-time ablation on trained Stage1 checkpoints; no retraining.",
        "",
        f"Setup: seeds {args.seeds}, conditions {args.conditions}, horizon={args.horizon}, val split from nuPlan 50k.",
        "",
        "| condition | ablation | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, by_ablation in summary.items():
        for ablation, block in by_ablation.items():
            mean = block["mean"]
            std = block["std_across_seeds"]
            lines.append(
                f"| {condition} | {ablation} | "
                f"{mean['teacher_action_mse']:.3f} ± {std['teacher_action_mse']:.3f} | "
                f"{mean['teacher_action_delta_l2']:.3f} ± {std['teacher_action_delta_l2']:.3f} | "
                f"{mean['latent_return_mean']:.3f} ± {std['latent_return_mean']:.3f} | "
                f"{mean['imagined_collision_rate']:.3f} ± {std['imagined_collision_rate']:.3f} | "
                f"{mean['collision_mean']:.3f} ± {std['collision_mean']:.3f} | "
                f"{mean['rollout_stability']:.3f} |"
            )
    lines.extend([
        "",
        "Reading:",
        "",
        "- `no_ttc_risk` zeros relation-token risk and TTC features.",
        "- `no_lane_priority` zeros relation-token lane-conflict and priority features.",
        "- This is an interpretation probe, not a new trained model condition.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw: Dict[str, Dict[str, Dict[str, Dict]]] = {
        condition: {ablation: {} for ablation in args.ablations}
        for condition in args.conditions
    }
    for seed in args.seeds:
        print("\n" + "=" * 90, flush=True)
        print(f"[ablation] seed={seed}", flush=True)
        set_seed(seed)
        config.seed = seed
        loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            variant = _CONDITIONS[condition]
            model = DoorRLModelVariant(config.model, ModelVariant(variant))
            ckpt = Path(args.stage1_root) / f"seed{seed}" / condition / "model.pt"
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            model.to(device)
            model.eval()
            for ablation in args.ablations:
                print(f"[ablation] seed={seed} condition={condition} ablation={ablation}", flush=True)
                metrics = evaluate(model, loader, device, args, ablation)
                raw[condition][ablation][str(seed)] = {"seed": seed, **metrics}
                exp_dir = out_dir / f"seed{seed}" / condition / ablation
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "metrics.json").write_text(json.dumps(raw[condition][ablation][str(seed)], indent=2))
                print(metrics, flush=True)
    summary = {
        "note": "evaluation-time relation feature ablation; no retraining",
        "setup": vars(args),
        "summary": _summarize(raw),
        "raw": raw,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out_dir, summary["summary"], args)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
