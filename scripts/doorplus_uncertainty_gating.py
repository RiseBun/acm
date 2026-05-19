"""DOOR+ uncertainty-gating and confidence-ablation probe.

The script supports both the diagnostic post-hoc oracle gate and the trainable
confidence-token DOOR+ selector. Confidence-ablation modes test whether
false-positive filtering depends on the confidence field.
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
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import TaskRewardCfg
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_CONDITIONS = {
    "door": "object_relation_decoupled",
    "doorplus_posthoc": "object_relation_decoupled",
    "doorplus_uncertainty": "object_relation_decoupled_uncertainty",
}
_CKPT_CONDITION = {
    "door": "wm_decoupled_no_vis",
    "doorplus_posthoc": "wm_decoupled_no_vis",
    "doorplus_uncertainty": "wm_doorplus_uncertainty",
}
_DYNAMIC_TYPES = {
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
}
_RISK_DIM = 6
_TTC_DIM = 8
_LANE_CONFLICT_DIM = 9
_PRIORITY_DIM = 10
_REL_CONF_DIM = 15
_REL_UNC_DIM = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuplan-root", default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument("--nuplan-index-json", default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"))
    parser.add_argument("--stage1-root", default=str(ROOT / "experiments" / "nuplan_stage1_50k"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "tpami_doorplus_uncertainty_gating"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--conditions", nargs="+", default=list(_CONDITIONS), choices=list(_CONDITIONS))
    parser.add_argument("--corruptions", nargs="+", default=["clean", "loc1.5", "miss0.2", "relfp0.2"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--fp-confidence", type=float, default=0.05)
    parser.add_argument("--safe-ttc", type=float, default=10.0)
    parser.add_argument(
        "--true-risk-threshold",
        type=float,
        default=0.5,
        help="Risk threshold used to define non-FP true high-risk relations.",
    )
    parser.add_argument(
        "--true-ttc-threshold",
        type=float,
        default=3.0,
        help="TTC threshold used to define non-FP true high-risk relations.",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=["normal", "proxy", "shuffled", "constant", "inverted", "proxy_inverted"],
        default="normal",
        help=(
            "Ablation for trainable DOOR+: normal keeps low confidence on "
            "false positives; proxy derives confidence from geometric relation "
            "consistency (distance/risk/TTC/relative velocity) without reading "
            "the FP mask; shuffled permutes relation confidence/uncertainty "
            "within each sample; constant disables confidence; inverted and "
            "proxy_inverted are negative controls."
        ),
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


def _load_model(args: argparse.Namespace, config: DoorRLConfig, seed: int, condition: str, device: torch.device):
    model = DoorRLModelVariant(config.model, ModelVariant(_CONDITIONS[condition]))
    ckpt_condition = _CKPT_CONDITION[condition]
    ckpt = Path(args.stage1_root) / f"seed{seed}" / ckpt_condition / "model.pt"
    if condition == "doorplus_uncertainty" and not ckpt.exists():
        # The deployable uncertainty selector adds no new learned parameters;
        # it can reuse the matching DOOR checkpoint and read confidence fields
        # at evaluation time.
        ckpt_condition = "wm_decoupled_no_vis"
        ckpt = Path(args.stage1_root) / f"seed{seed}" / ckpt_condition / "model.pt"
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


def _apply_corruption(batch: SceneBatch, corruption: str, fp_confidence: float = 0.05) -> Tuple[SceneBatch, torch.Tensor]:
    fp_mask = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    if corruption == "clean":
        return batch, fp_mask

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
        distance_noise = torch.randn(tokens[..., 11].shape, device=tokens.device) * std
        tokens[..., 11] = torch.where(rel_mask, (tokens[..., 11] + distance_noise).clamp_min(0.0), tokens[..., 11])
        return _copy_batch(batch, tokens), fp_mask

    if corruption.startswith("miss"):
        rate = float(corruption[4:])
        drop = ((torch.rand(token_mask.shape, device=tokens.device) < rate) & (dyn_mask | rel_mask))
        tokens = torch.where(drop.unsqueeze(-1), torch.zeros_like(tokens), tokens)
        token_mask = token_mask & ~drop
        return _copy_batch(batch, tokens, token_mask), fp_mask

    if corruption.startswith("relfp"):
        rate = float(corruption[5:])
        fp_mask = (torch.rand(token_mask.shape, device=tokens.device) < rate) & rel_mask
        tokens[..., _REL_CONF_DIM] = torch.where(rel_mask, torch.ones_like(tokens[..., _REL_CONF_DIM]), tokens[..., _REL_CONF_DIM])
        tokens[..., _REL_CONF_DIM] = torch.where(fp_mask, torch.full_like(tokens[..., _REL_CONF_DIM], float(fp_confidence)), tokens[..., _REL_CONF_DIM])
        tokens[..., _REL_UNC_DIM] = torch.where(fp_mask, torch.ones_like(tokens[..., _REL_UNC_DIM]), tokens[..., _REL_UNC_DIM])
        tokens[..., _RISK_DIM] = torch.where(fp_mask, torch.ones_like(tokens[..., _RISK_DIM]), tokens[..., _RISK_DIM])
        tokens[..., _TTC_DIM] = torch.where(fp_mask, torch.zeros_like(tokens[..., _TTC_DIM]), tokens[..., _TTC_DIM])
        tokens[..., _LANE_CONFLICT_DIM] = torch.where(fp_mask, torch.ones_like(tokens[..., _LANE_CONFLICT_DIM]), tokens[..., _LANE_CONFLICT_DIM])
        tokens[..., _PRIORITY_DIM] = torch.where(fp_mask, torch.ones_like(tokens[..., _PRIORITY_DIM]), tokens[..., _PRIORITY_DIM])
        return _copy_batch(batch, tokens), fp_mask

    raise ValueError(f"Unknown corruption: {corruption}")


def _apply_doorplus_gate(batch: SceneBatch, fp_mask: torch.Tensor, args: argparse.Namespace) -> SceneBatch:
    if int(fp_mask.sum().item()) == 0:
        return batch
    tokens = batch.tokens.clone()
    conf = float(args.fp_confidence)
    tokens[..., _RISK_DIM] = torch.where(fp_mask, tokens[..., _RISK_DIM] * conf, tokens[..., _RISK_DIM])
    tokens[..., _TTC_DIM] = torch.where(
        fp_mask,
        torch.full_like(tokens[..., _TTC_DIM], float(args.safe_ttc)),
        tokens[..., _TTC_DIM],
    )
    tokens[..., _LANE_CONFLICT_DIM] = torch.where(fp_mask, tokens[..., _LANE_CONFLICT_DIM] * conf, tokens[..., _LANE_CONFLICT_DIM])
    tokens[..., _PRIORITY_DIM] = torch.where(fp_mask, tokens[..., _PRIORITY_DIM] * conf, tokens[..., _PRIORITY_DIM])
    return _copy_batch(batch, tokens)


def _apply_confidence_mode(
    batch: SceneBatch,
    fp_mask: torch.Tensor,
    mode: str,
    fp_confidence: float,
) -> SceneBatch:
    if mode == "normal":
        return batch

    tokens = batch.tokens.clone()
    rel_mask = _relation_mask(batch)

    if mode in {"proxy", "proxy_inverted"}:
        proxy = _geometric_relation_confidence(tokens, rel_mask)
        if mode == "proxy_inverted":
            proxy = 1.0 - proxy
        tokens[..., _REL_CONF_DIM] = torch.where(
            rel_mask, proxy, tokens[..., _REL_CONF_DIM]
        )
        tokens[..., _REL_UNC_DIM] = torch.where(
            rel_mask, 1.0 - proxy, tokens[..., _REL_UNC_DIM]
        )
        return _copy_batch(batch, tokens)

    if mode == "constant":
        tokens[..., _REL_CONF_DIM] = torch.where(
            rel_mask, torch.ones_like(tokens[..., _REL_CONF_DIM]), tokens[..., _REL_CONF_DIM]
        )
        tokens[..., _REL_UNC_DIM] = torch.where(
            rel_mask, torch.zeros_like(tokens[..., _REL_UNC_DIM]), tokens[..., _REL_UNC_DIM]
        )
        return _copy_batch(batch, tokens)

    if mode == "inverted":
        non_fp_rel = rel_mask & ~fp_mask
        tokens[..., _REL_CONF_DIM] = torch.where(
            fp_mask, torch.ones_like(tokens[..., _REL_CONF_DIM]), tokens[..., _REL_CONF_DIM]
        )
        tokens[..., _REL_UNC_DIM] = torch.where(
            fp_mask, torch.zeros_like(tokens[..., _REL_UNC_DIM]), tokens[..., _REL_UNC_DIM]
        )
        tokens[..., _REL_CONF_DIM] = torch.where(
            non_fp_rel,
            torch.full_like(tokens[..., _REL_CONF_DIM], float(fp_confidence)),
            tokens[..., _REL_CONF_DIM],
        )
        tokens[..., _REL_UNC_DIM] = torch.where(
            non_fp_rel, torch.ones_like(tokens[..., _REL_UNC_DIM]), tokens[..., _REL_UNC_DIM]
        )
        return _copy_batch(batch, tokens)

    if mode == "shuffled":
        for b in range(tokens.size(0)):
            rel_idx = torch.nonzero(rel_mask[b], as_tuple=False).squeeze(-1)
            if rel_idx.numel() <= 1:
                continue
            perm = rel_idx[torch.randperm(rel_idx.numel(), device=tokens.device)]
            tokens[b, rel_idx, _REL_CONF_DIM] = tokens[b, perm, _REL_CONF_DIM]
            tokens[b, rel_idx, _REL_UNC_DIM] = tokens[b, perm, _REL_UNC_DIM]
        return _copy_batch(batch, tokens)

    raise ValueError(f"Unknown confidence mode: {mode}")


def _geometric_relation_confidence(tokens: torch.Tensor, rel_mask: torch.Tensor) -> torch.Tensor:
    """Detector-free confidence proxy from relation-geometry consistency.

    False-positive relation corruption edits semantic relation claims
    (risk/TTC/lane conflict/priority) but leaves the underlying relative
    geometry mostly intact. This proxy therefore trusts a high-risk relation
    only when the claimed TTC agrees with TTC recomputed from (dx, dy, dvx, dvy).
    It is not an oracle: it never reads ``fp_mask``.
    """
    dx = tokens[..., 0]
    dy = tokens[..., 1]
    rel_vx = tokens[..., 2]
    rel_vy = tokens[..., 3]
    risk = tokens[..., _RISK_DIM].clamp(0.0, 1.0)
    ttc_claim = tokens[..., _TTC_DIM].clamp(0.0, 20.0)
    lane_conflict = tokens[..., _LANE_CONFLICT_DIM].clamp(0.0, 1.0)
    priority = tokens[..., _PRIORITY_DIM].clamp(0.0, 1.0)
    distance = tokens[..., 11].clamp_min(0.0)
    geom_distance = torch.sqrt(dx * dx + dy * dy).clamp_min(0.1)
    distance = torch.where(distance > 0.0, distance, geom_distance)

    closing_speed = (dx * rel_vx + dy * rel_vy) / geom_distance
    ttc_geom = torch.where(
        closing_speed > 0.05,
        (geom_distance / closing_speed.clamp_min(0.05)).clamp(0.0, 20.0),
        torch.full_like(geom_distance, 20.0),
    )
    ttc_consistency = torch.exp(-torch.abs(ttc_claim - ttc_geom) / 4.0)
    distance_conf = torch.exp(-distance / 60.0).clamp(0.05, 1.0)
    semantic_strength = (0.4 * risk + 0.3 * lane_conflict + 0.3 * priority).clamp(0.0, 1.0)
    # High semantic confidence still needs geometric consistency; low-risk,
    # far-away relations stay moderately trusted rather than being forced out.
    confidence = (0.35 + 0.65 * ttc_consistency) * distance_conf
    confidence = confidence * (0.5 + 0.5 * semantic_strength)
    confidence = confidence.clamp(0.05, 1.0)
    return torch.where(rel_mask, confidence, torch.ones_like(confidence))


def _cat_mean(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    return float(torch.cat(xs).mean().item()) if xs else float("nan")


def _cat_std(values: Iterable[torch.Tensor]) -> float:
    xs = [v.detach().float().cpu() for v in values]
    return float(torch.cat(xs).std().item()) if xs else float("nan")


def _stability_score(latents: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(latents[:, :-1], latents[:, 1:], dim=-1, eps=1e-3)
    return (1.0 - cos).mean(dim=1)


def _selected_relation_fractions(
    model,
    batch: SceneBatch,
    fp_mask: torch.Tensor,
    true_risk_threshold: float,
    true_ttc_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(batch)
    selected = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    idx = output.abstraction.selected_indices
    mask = output.abstraction.selected_mask.bool()
    selected.scatter_(1, torch.where(mask, idx, torch.zeros_like(idx)), mask)
    denom = float(getattr(model, "top_k_rel", 4))
    rel_mask = _relation_mask(batch)
    selected_fp = selected & fp_mask & rel_mask

    risk = batch.tokens[..., _RISK_DIM]
    ttc = batch.tokens[..., _TTC_DIM]
    true_risk = (
        rel_mask
        & ~fp_mask
        & ((risk >= float(true_risk_threshold)) | ((ttc > 0.0) & (ttc <= float(true_ttc_threshold))))
    )
    selected_true_risk = selected & true_risk
    denom = max(denom, 1.0)
    return (
        selected_fp.float().sum(dim=1) / denom,
        (fp_mask & rel_mask).float().sum(dim=1) / denom,
        selected_true_risk.float().sum(dim=1) / denom,
        true_risk.float().sum(dim=1) / denom,
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device, args: argparse.Namespace, condition: str, corruption: str, seed: int) -> Dict[str, float]:
    set_seed(seed)
    returns: List[torch.Tensor] = []
    coll_max: List[torch.Tensor] = []
    coll_step: List[torch.Tensor] = []
    action_mse: List[torch.Tensor] = []
    action_delta_l2: List[torch.Tensor] = []
    stability: List[torch.Tensor] = []
    fp_rel_selected: List[torch.Tensor] = []
    fp_rel_density: List[torch.Tensor] = []
    true_risk_rel_selected: List[torch.Tensor] = []
    true_risk_rel_density: List[torch.Tensor] = []
    n = 0
    for batch in loader:
        batch, fp_mask = _apply_corruption(batch.to(device), corruption, fp_confidence=float(args.fp_confidence))
        if condition == "doorplus_posthoc":
            batch = _apply_doorplus_gate(batch, fp_mask, args)
        elif condition == "doorplus_uncertainty":
            batch = _apply_confidence_mode(
                batch,
                fp_mask,
                mode=str(getattr(args, "confidence_mode", "normal")),
                fp_confidence=float(args.fp_confidence),
            )

        fp_sel, fp_density, true_sel, true_density = _selected_relation_fractions(
            model,
            batch,
            fp_mask,
            true_risk_threshold=float(args.true_risk_threshold),
            true_ttc_threshold=float(args.true_ttc_threshold),
        )
        fp_rel_selected.append(fp_sel)
        fp_rel_density.append(fp_density)
        true_risk_rel_selected.append(true_sel)
        true_risk_rel_density.append(true_density)

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
        "selected_fp_rel_at_k": _cat_mean(fp_rel_selected),
        "fp_rel_density_per_k": _cat_mean(fp_rel_density),
        "selected_true_risk_rel_at_k": _cat_mean(true_risk_rel_selected),
        "true_risk_rel_density_per_k": _cat_mean(true_risk_rel_density),
        "confidence_mode": str(getattr(args, "confidence_mode", "normal")),
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
        "selected_fp_rel_at_k",
        "fp_rel_density_per_k",
        "selected_true_risk_rel_at_k",
        "true_risk_rel_density_per_k",
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
        "# DOOR+ uncertainty gating",
        "",
        "`doorplus_posthoc` reuses the DOOR checkpoint and suppresses artificial false-positive relation risk before selection/imagination. "
        "`doorplus_uncertainty` loads the trainable/deployable selector variant, where confidence is written into the token and used by the relation selector.",
        "",
        f"Setup: seeds {args.seeds}, horizon={args.horizon}, max_val_samples={args.max_val_samples or 'full val'}, "
        f"fp_confidence={args.fp_confidence}, confidence_mode={args.confidence_mode}.",
        "",
        "| condition | corruption | Return | CollRate | CollMean | Teacher MSE | Selected-FP-Rel@K | FP density/K | Selected-TrueRisk-Rel@K | TrueRisk density/K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, by_corruption in summary.items():
        for corruption, block in by_corruption.items():
            mean = block["mean"]
            std = block["std_across_seeds"]
            lines.append(
                f"| {condition} | {corruption} | "
                f"{mean['latent_return_mean']:.3f} +/- {std['latent_return_mean']:.3f} | "
                f"{mean['imagined_collision_rate']:.3f} +/- {std['imagined_collision_rate']:.3f} | "
                f"{mean['collision_mean']:.3f} +/- {std['collision_mean']:.3f} | "
                f"{mean['teacher_action_mse']:.3f} +/- {std['teacher_action_mse']:.3f} | "
                f"{mean['selected_fp_rel_at_k']:.3f} +/- {std['selected_fp_rel_at_k']:.3f} | "
                f"{mean['fp_rel_density_per_k']:.3f} | "
                f"{mean['selected_true_risk_rel_at_k']:.3f} +/- {std['selected_true_risk_rel_at_k']:.3f} | "
                f"{mean['true_risk_rel_density_per_k']:.3f} |"
            )
    lines.extend([
        "",
        "Reading:",
        "",
        "- `Selected-FP-Rel@K` is the number of selected artificial false-positive relation tokens divided by K_rel=4.",
        "- `Selected-TrueRisk-Rel@K` is the selected non-FP high-risk relation count divided by K_rel=4.",
        "- `doorplus_posthoc` is an oracle-confidence probe; `doorplus_uncertainty` is the deployable confidence-token selector variant.",
        "- A useful DOOR+ signal is lower `relfp` CollRate with clean metrics unchanged.",
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
        print(f"[doorplus] seed={seed}", flush=True)
        config.seed = seed
        loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            print(f"[doorplus] seed={seed} condition={condition}", flush=True)
            model = _load_model(args, config, seed, condition, device)
            for corruption in args.corruptions:
                print(f"[doorplus] seed={seed} condition={condition} corruption={corruption}", flush=True)
                metrics = evaluate(model, loader, device, args, condition, corruption, seed + len(condition) * 1000 + len(corruption))
                raw[condition][corruption][str(seed)] = {"seed": seed, **metrics}
                exp_dir = out_dir / f"seed{seed}" / condition / corruption
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "metrics.json").write_text(json.dumps(raw[condition][corruption][str(seed)], indent=2))
                print(metrics, flush=True)

    summary = {
        "note": "post-hoc DOOR+ uncertainty gating; no retraining",
        "setup": vars(args),
        "summary": _summarize(raw),
        "raw": raw,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out_dir, summary["summary"], args)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
