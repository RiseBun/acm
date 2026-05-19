"""Evaluation-time perception-noise robustness probe for nuPlan Stage1 checkpoints.

The script does not retrain models. It corrupts the held-out input tokens before
deterministic latent imagination and compares degradation across object-only,
shared object-relation, and typed-budget object-relation policies.
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
    "wm_object": "object_only",
    "wm_naive": "object_relation",
    "wm_decoupled_no_vis": "object_relation_decoupled",
    "wm_doorplus_uncertainty": "object_relation_decoupled_uncertainty",
}

_DYNAMIC_TYPES = {
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuplan-root", default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument("--nuplan-index-json", default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"))
    parser.add_argument("--stage1-root", default=str(ROOT / "experiments" / "nuplan_stage1_50k"))
    parser.add_argument("--shared-stage1-root", default=str(ROOT / "experiments" / "nuplan_stage1_shared_relation_50k"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "tpami_perception_noise_robustness"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--conditions", nargs="+", default=list(_CONDITIONS), choices=list(_CONDITIONS))
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["clean", "loc0.5", "loc1.5", "miss0.2", "relfp0.2"],
        help="Supported: clean, loc<meters>, miss<fraction>, relfp<fraction>.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
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


def _checkpoint_root(args: argparse.Namespace, condition: str) -> Path:
    if condition == "wm_naive":
        return Path(args.shared_stage1_root)
    return Path(args.stage1_root)


def _load_model(args: argparse.Namespace, config: DoorRLConfig, seed: int, condition: str, device: torch.device):
    variant = _CONDITIONS[condition]
    model = DoorRLModelVariant(config.model, ModelVariant(variant))
    ckpt = _checkpoint_root(args, condition) / f"seed{seed}" / condition / "model.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _copy_batch(batch: SceneBatch, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> SceneBatch:
    return SceneBatch(
        tokens=tokens,
        token_mask=batch.token_mask if token_mask is None else token_mask,
        token_types=batch.token_types,
        actions=batch.actions,
        next_tokens=batch.next_tokens,
        rewards=batch.rewards,
        continues=batch.continues,
    )


def _dynamic_mask(batch: SceneBatch) -> torch.Tensor:
    mask = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    for token_type in _DYNAMIC_TYPES:
        mask |= batch.token_types == token_type
    return mask & batch.token_mask


def _relation_mask(batch: SceneBatch) -> torch.Tensor:
    return (batch.token_types == int(TokenType.RELATION)) & batch.token_mask


def _apply_corruption(batch: SceneBatch, corruption: str) -> SceneBatch:
    if corruption == "clean":
        return batch

    tokens = batch.tokens.clone()
    token_mask = batch.token_mask.clone()
    dyn_mask = _dynamic_mask(batch)
    rel_mask = _relation_mask(batch)

    if corruption.startswith("loc"):
        std = float(corruption[3:])
        dyn_xy_noise = torch.randn(tokens[..., 0:2].shape, device=tokens.device) * std
        rel_xy_noise = torch.randn(tokens[..., 0:2].shape, device=tokens.device) * std
        tokens[..., 0:2] = torch.where(dyn_mask.unsqueeze(-1), tokens[..., 0:2] + dyn_xy_noise, tokens[..., 0:2])
        tokens[..., 0:2] = torch.where(rel_mask.unsqueeze(-1), tokens[..., 0:2] + rel_xy_noise, tokens[..., 0:2])
        # Keep the relation distance feature consistent enough to represent noisy perception.
        distance_noise = torch.randn(tokens[..., 11].shape, device=tokens.device) * std
        tokens[..., 11] = torch.where(rel_mask, (tokens[..., 11] + distance_noise).clamp_min(0.0), tokens[..., 11])
        return _copy_batch(batch, tokens)

    if corruption.startswith("miss"):
        rate = float(corruption[4:])
        dyn_drop = (torch.rand(token_mask.shape, device=tokens.device) < rate) & dyn_mask
        rel_drop = (torch.rand(token_mask.shape, device=tokens.device) < rate) & rel_mask
        drop = dyn_drop | rel_drop
        tokens = torch.where(drop.unsqueeze(-1), torch.zeros_like(tokens), tokens)
        token_mask = token_mask & ~drop
        return _copy_batch(batch, tokens, token_mask)

    if corruption.startswith("relfp"):
        rate = float(corruption[5:])
        fp = (torch.rand(token_mask.shape, device=tokens.device) < rate) & rel_mask
        # Confidence/uncertainty fields used by the deployable DOOR+ variant.
        tokens[..., 15] = torch.where(rel_mask, torch.ones_like(tokens[..., 15]), tokens[..., 15])
        tokens[..., 15] = torch.where(fp, torch.full_like(tokens[..., 15], 0.05), tokens[..., 15])
        tokens[..., 16] = torch.where(fp, torch.ones_like(tokens[..., 16]), tokens[..., 16])
        tokens[..., 6] = torch.where(fp, torch.ones_like(tokens[..., 6]), tokens[..., 6])
        tokens[..., 8] = torch.where(fp, torch.zeros_like(tokens[..., 8]), tokens[..., 8])
        tokens[..., 9] = torch.where(fp, torch.ones_like(tokens[..., 9]), tokens[..., 9])
        tokens[..., 10] = torch.where(fp, torch.ones_like(tokens[..., 10]), tokens[..., 10])
        return _copy_batch(batch, tokens)

    raise ValueError(f"Unknown corruption: {corruption}")


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
def evaluate(model, loader, device: torch.device, args: argparse.Namespace, corruption: str, seed: int) -> Dict[str, float]:
    set_seed(seed)
    returns: List[torch.Tensor] = []
    coll_max: List[torch.Tensor] = []
    coll_step: List[torch.Tensor] = []
    action_mse: List[torch.Tensor] = []
    action_delta_l2: List[torch.Tensor] = []
    stability: List[torch.Tensor] = []
    n = 0
    for batch in loader:
        batch = _apply_corruption(batch.to(device), corruption)
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
    metric_names = [
        "teacher_action_mse",
        "teacher_action_delta_l2",
        "latent_return_mean",
        "imagined_collision_rate",
        "collision_mean",
        "collision_step_mean",
        "rollout_stability",
    ]
    summary: Dict[str, Dict] = {}
    for condition, by_corruption in raw.items():
        summary[condition] = {}
        for corruption, by_seed in by_corruption.items():
            rows = list(by_seed.values())
            summary[condition][corruption] = {"mean": {}, "std_across_seeds": {}, "seeds": rows}
            for name in metric_names:
                vals = torch.tensor([row[name] for row in rows], dtype=torch.float32)
                summary[condition][corruption]["mean"][name] = float(vals.mean().item())
                summary[condition][corruption]["std_across_seeds"][name] = (
                    float(vals.std().item()) if len(vals) > 1 else 0.0
                )
    return summary


def _write_markdown(out_dir: Path, summary: Dict, args: argparse.Namespace) -> None:
    lines = [
        "# TPAMI perception-noise robustness probe",
        "",
        "Evaluation-time input corruption on trained nuPlan Stage1 checkpoints; no retraining.",
        "",
        f"Setup: seeds {args.seeds}, horizon={args.horizon}, max_val_samples={args.max_val_samples or 'full val'}.",
        "",
        "| condition | corruption | Teacher MSE | Action ΔL2 | Return | CollRate | CollMean | Stability |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, by_corruption in summary.items():
        for corruption, block in by_corruption.items():
            mean = block["mean"]
            std = block["std_across_seeds"]
            lines.append(
                f"| {condition} | {corruption} | "
                f"{mean['teacher_action_mse']:.3f} +/- {std['teacher_action_mse']:.3f} | "
                f"{mean['teacher_action_delta_l2']:.3f} +/- {std['teacher_action_delta_l2']:.3f} | "
                f"{mean['latent_return_mean']:.3f} +/- {std['latent_return_mean']:.3f} | "
                f"{mean['imagined_collision_rate']:.3f} +/- {std['imagined_collision_rate']:.3f} | "
                f"{mean['collision_mean']:.3f} +/- {std['collision_mean']:.3f} | "
                f"{mean['rollout_stability']:.3f} |"
            )
    lines.extend([
        "",
        "Reading:",
        "",
        "- `locX` adds Gaussian localization noise with std `X` to dynamic and relation x/y features.",
        "- `missR` randomly masks dynamic and relation tokens with probability `R`.",
        "- `relfpR` turns a fraction `R` of existing relation tokens into high-risk false-positive relations.",
        "- This probe measures evaluation-time robustness only; it is not a noisy-perception training result.",
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
        condition: {corruption: {} for corruption in args.corruptions}
        for condition in args.conditions
    }
    for seed in args.seeds:
        print("\n" + "=" * 90, flush=True)
        print(f"[noise] seed={seed}", flush=True)
        config.seed = seed
        loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            print(f"[noise] seed={seed} condition={condition}", flush=True)
            model = _load_model(args, config, seed, condition, device)
            for corruption in args.corruptions:
                print(f"[noise] seed={seed} condition={condition} corruption={corruption}", flush=True)
                metrics = evaluate(model, loader, device, args, corruption, seed + abs(hash((condition, corruption))) % 100000)
                raw[condition][corruption][str(seed)] = {"seed": seed, **metrics}
                exp_dir = out_dir / f"seed{seed}" / condition / corruption
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "metrics.json").write_text(json.dumps(raw[condition][corruption][str(seed)], indent=2))
                print(metrics, flush=True)

    summary = {
        "note": "evaluation-time perception-noise robustness; no retraining",
        "setup": vars(args),
        "summary": _summarize(raw),
        "raw": raw,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out_dir, summary["summary"], args)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
