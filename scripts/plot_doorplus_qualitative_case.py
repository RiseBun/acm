#!/usr/bin/env python3
"""Qualitative DOOR+ false-positive relation suppression case.

Finds a nuPlan validation sample where baseline DOOR selects injected
false-positive relation tokens, while DOOR+ proxy confidence suppresses them
and keeps true-risk relations. The output is a two-panel BEV-style mechanism
figure plus a JSON sidecar with the selected token ids.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_RISK_DIM = 6
_TTC_DIM = 8
_LANE_CONFLICT_DIM = 9
_PRIORITY_DIM = 10
_DIST_DIM = 11
_REL_CONF_DIM = 15
_REL_UNC_DIM = 16
_TRUE_RISK_THRESHOLD = 0.5
_TRUE_TTC_THRESHOLD = 3.0
_DYN_TYPES = {
    int(TokenType.EGO),
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
}
_TYPE_NAME = {
    int(TokenType.EGO): "EGO",
    int(TokenType.VEHICLE): "VEH",
    int(TokenType.PEDESTRIAN): "PED",
    int(TokenType.CYCLIST): "CYC",
}
_TYPE_COLOR = {
    int(TokenType.EGO): "#d62728",
    int(TokenType.VEHICLE): "#1f77b4",
    int(TokenType.PEDESTRIAN): "#2ca02c",
    int(TokenType.CYCLIST): "#ff7f0e",
}


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], scene_names: Sequence[str]):
        self.dataset = dataset
        self.indices = list(indices)
        self.scene_names = list(scene_names)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        dataset_idx = self.indices[idx]
        item = dict(self.dataset[dataset_idx])
        item["_dataset_index"] = dataset_idx
        item["_scene_id"] = self.scene_names[dataset_idx]
        return item


def _collate(items: Sequence[Dict]) -> Tuple[SceneBatch, List[int], List[str]]:
    batch_items = []
    indices: List[int] = []
    scene_ids: List[str] = []
    for item in items:
        item = dict(item)
        indices.append(int(item.pop("_dataset_index")))
        scene_ids.append(str(item.pop("_scene_id")))
        batch_items.append(item)
    return SceneBatch.collate(batch_items), indices, scene_ids


def _build_loader(args: argparse.Namespace, config: DoorRLConfig) -> DataLoader:
    dataset = NuPlanPreprocessedDataset(
        config=config,
        data_root=args.nuplan_root,
        num_samples=args.nuplan_num_samples,
        index_json=args.nuplan_index_json,
        seed=args.seed,
        materialize_cache=False,
    )
    names = sorted(set(dataset.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled)))) if len(shuffled) > 1 else 0
    val_scenes = set(shuffled[len(shuffled) - n_val:])
    val_idx = dataset.indices_for_scenes(val_scenes)
    if args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]
    subset = IndexedSubset(dataset, val_idx, dataset.cache_scene_names)
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )


def _load_model(
    config: DoorRLConfig,
    ckpt: Path,
    variant: str,
    device: torch.device,
) -> DoorRLModelVariant:
    model = DoorRLModelVariant(config.model, ModelVariant(variant))
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _copy_batch(batch: SceneBatch, tokens: torch.Tensor) -> SceneBatch:
    return SceneBatch(
        tokens=tokens,
        token_mask=batch.token_mask,
        token_types=batch.token_types,
        actions=batch.actions,
        next_tokens=batch.next_tokens,
        rewards=batch.rewards,
        continues=batch.continues,
    )


def _relation_mask(batch: SceneBatch) -> torch.Tensor:
    return (batch.token_types == int(TokenType.RELATION)) & batch.token_mask


def _apply_relfp(
    batch: SceneBatch,
    rate: float,
    fp_confidence: float,
) -> Tuple[SceneBatch, torch.Tensor]:
    tokens = batch.tokens.clone()
    rel_mask = _relation_mask(batch)
    fp_mask = (torch.rand(batch.token_mask.shape, device=tokens.device) < rate) & rel_mask
    tokens[..., _REL_CONF_DIM] = torch.where(
        rel_mask, torch.ones_like(tokens[..., _REL_CONF_DIM]), tokens[..., _REL_CONF_DIM]
    )
    tokens[..., _REL_CONF_DIM] = torch.where(
        fp_mask, torch.full_like(tokens[..., _REL_CONF_DIM], fp_confidence), tokens[..., _REL_CONF_DIM]
    )
    tokens[..., _REL_UNC_DIM] = torch.where(
        fp_mask, torch.ones_like(tokens[..., _REL_UNC_DIM]), tokens[..., _REL_UNC_DIM]
    )
    tokens[..., _RISK_DIM] = torch.where(fp_mask, torch.ones_like(tokens[..., _RISK_DIM]), tokens[..., _RISK_DIM])
    tokens[..., _TTC_DIM] = torch.where(fp_mask, torch.zeros_like(tokens[..., _TTC_DIM]), tokens[..., _TTC_DIM])
    tokens[..., _LANE_CONFLICT_DIM] = torch.where(
        fp_mask, torch.ones_like(tokens[..., _LANE_CONFLICT_DIM]), tokens[..., _LANE_CONFLICT_DIM]
    )
    tokens[..., _PRIORITY_DIM] = torch.where(
        fp_mask, torch.ones_like(tokens[..., _PRIORITY_DIM]), tokens[..., _PRIORITY_DIM]
    )
    return _copy_batch(batch, tokens), fp_mask


def _geometric_relation_confidence(tokens: torch.Tensor, rel_mask: torch.Tensor) -> torch.Tensor:
    dx = tokens[..., 0]
    dy = tokens[..., 1]
    rel_vx = tokens[..., 2]
    rel_vy = tokens[..., 3]
    risk = tokens[..., _RISK_DIM].clamp(0.0, 1.0)
    ttc_claim = tokens[..., _TTC_DIM].clamp(0.0, 20.0)
    lane_conflict = tokens[..., _LANE_CONFLICT_DIM].clamp(0.0, 1.0)
    priority = tokens[..., _PRIORITY_DIM].clamp(0.0, 1.0)
    distance = tokens[..., _DIST_DIM].clamp_min(0.0)
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
    confidence = (0.35 + 0.65 * ttc_consistency) * distance_conf
    confidence = confidence * (0.5 + 0.5 * semantic_strength)
    confidence = confidence.clamp(0.05, 1.0)
    return torch.where(rel_mask, confidence, torch.ones_like(confidence))


def _apply_proxy_confidence(batch: SceneBatch) -> SceneBatch:
    tokens = batch.tokens.clone()
    rel_mask = _relation_mask(batch)
    proxy = _geometric_relation_confidence(tokens, rel_mask)
    tokens[..., _REL_CONF_DIM] = torch.where(rel_mask, proxy, tokens[..., _REL_CONF_DIM])
    tokens[..., _REL_UNC_DIM] = torch.where(rel_mask, 1.0 - proxy, tokens[..., _REL_UNC_DIM])
    return _copy_batch(batch, tokens)


def _selected_token_mask(output, batch: SceneBatch) -> torch.Tensor:
    selected = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    idx = output.abstraction.selected_indices
    mask = output.abstraction.selected_mask.bool()
    selected.scatter_(1, torch.where(mask, idx, torch.zeros_like(idx)), mask)
    return selected & batch.token_mask


def _true_risk_mask(batch: SceneBatch, fp_mask: torch.Tensor) -> torch.Tensor:
    rel_mask = _relation_mask(batch)
    risk = batch.tokens[..., _RISK_DIM]
    ttc = batch.tokens[..., _TTC_DIM]
    return (
        rel_mask
        & ~fp_mask
        & ((risk >= _TRUE_RISK_THRESHOLD) | ((ttc > 0.0) & (ttc <= _TRUE_TTC_THRESHOLD)))
    )


def _infer_relation_endpoint(tokens: torch.Tensor, types: torch.Tensor, mask: torch.Tensor, ridx: int) -> int:
    dyn_mask = torch.zeros_like(mask, dtype=torch.bool)
    for token_type in _DYN_TYPES:
        dyn_mask |= types == token_type
    dyn_mask &= mask
    dyn_idx = torch.nonzero(dyn_mask, as_tuple=False).flatten()
    dyn_idx = dyn_idx[dyn_idx != 0]
    if dyn_idx.numel() == 0:
        return -1
    dist = torch.sqrt(((tokens[dyn_idx, :2] - tokens[ridx, :2]) ** 2).sum(dim=1) + 1e-12)
    return int(dyn_idx[int(dist.argmin().item())].item())


def _case_score(
    door_sel: torch.Tensor,
    plus_sel: torch.Tensor,
    fp_mask: torch.Tensor,
    true_mask: torch.Tensor,
    sample_i: int,
) -> Tuple[int, Dict[str, List[int]]]:
    door_fp = torch.nonzero(door_sel[sample_i] & fp_mask[sample_i], as_tuple=False).flatten().tolist()
    plus_fp = torch.nonzero(plus_sel[sample_i] & fp_mask[sample_i], as_tuple=False).flatten().tolist()
    door_true = torch.nonzero(door_sel[sample_i] & true_mask[sample_i], as_tuple=False).flatten().tolist()
    plus_true = torch.nonzero(plus_sel[sample_i] & true_mask[sample_i], as_tuple=False).flatten().tolist()
    suppressed = sorted(set(door_fp) - set(plus_fp))
    score = 10 * len(suppressed) + 3 * len(plus_true) - 2 * len(plus_fp) - max(0, len(door_true) - len(plus_true))
    info = {
        "door_fp_selected": [int(x) for x in door_fp],
        "doorplus_fp_selected": [int(x) for x in plus_fp],
        "door_true_risk_selected": [int(x) for x in door_true],
        "doorplus_true_risk_selected": [int(x) for x in plus_true],
        "suppressed_fp_relations": [int(x) for x in suppressed],
    }
    return score, info


def _find_case(args, door_model, plus_model, loader, device) -> Dict:
    best: Dict | None = None
    set_seed(args.seed)
    with torch.no_grad():
        for batch, indices, scene_ids in loader:
            batch = batch.to(device)
            corrupt, fp_mask = _apply_relfp(batch, args.fp_rate, args.fp_confidence)
            true_mask = _true_risk_mask(corrupt, fp_mask)
            plus_batch = _apply_proxy_confidence(corrupt)
            door_out = door_model(corrupt)
            plus_out = plus_model(plus_batch)
            door_sel = _selected_token_mask(door_out, corrupt)
            plus_sel = _selected_token_mask(plus_out, plus_batch)
            for i in range(corrupt.tokens.size(0)):
                score, info = _case_score(door_sel, plus_sel, fp_mask, true_mask, i)
                if not info["suppressed_fp_relations"]:
                    continue
                if not info["doorplus_true_risk_selected"]:
                    continue
                if len(info["doorplus_fp_selected"]) >= len(info["door_fp_selected"]):
                    continue
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "batch_index": i,
                        "dataset_index": int(indices[i]),
                        "scene_id": scene_ids[i],
                        "tokens": corrupt.tokens[i].detach().cpu(),
                        "token_types": corrupt.token_types[i].detach().cpu(),
                        "token_mask": corrupt.token_mask[i].detach().cpu(),
                        "fp_mask": fp_mask[i].detach().cpu(),
                        "true_mask": true_mask[i].detach().cpu(),
                        "door_selected": door_sel[i].detach().cpu(),
                        "doorplus_selected": plus_sel[i].detach().cpu(),
                        "proxy_confidence": plus_batch.tokens[i, :, _REL_CONF_DIM].detach().cpu(),
                        **info,
                    }
    if best is None:
        raise RuntimeError("No DOOR+ qualitative case found. Try increasing --max-val-samples or --fp-rate.")
    return best


def _plot_panel(ax, case: Dict, selected_key: str, title: str) -> None:
    tokens: torch.Tensor = case["tokens"]
    types: torch.Tensor = case["token_types"]
    mask: torch.Tensor = case["token_mask"]
    selected: torch.Tensor = case[selected_key]
    fp_mask: torch.Tensor = case["fp_mask"]
    true_mask: torch.Tensor = case["true_mask"]
    proxy_conf: torch.Tensor = case["proxy_confidence"]

    ax.add_patch(plt.Circle((0, 0), 35.0, color="black", fill=False, linestyle=":", linewidth=0.7, alpha=0.35))
    for i in range(tokens.size(0)):
        if not bool(mask[i]):
            continue
        token_type = int(types[i].item())
        x, y = float(tokens[i, 0].item()), float(tokens[i, 1].item())
        if token_type in _DYN_TYPES:
            is_sel = bool(selected[i])
            marker = "*" if token_type == int(TokenType.EGO) else "o"
            size = 230 if token_type == int(TokenType.EGO) else (85 if is_sel else 28)
            ax.scatter(
                x,
                y,
                s=size,
                c=_TYPE_COLOR.get(token_type, "#777777"),
                marker=marker,
                edgecolors="#111111" if is_sel else _TYPE_COLOR.get(token_type, "#777777"),
                linewidths=1.5 if is_sel else 0.5,
                alpha=0.95 if is_sel else 0.55,
                zorder=3,
            )
        elif token_type == int(TokenType.RELATION):
            if not bool(selected[i]) and not bool(fp_mask[i]) and not bool(true_mask[i]):
                continue
            color = "#d62728" if bool(fp_mask[i]) else ("#2ca02c" if bool(true_mask[i]) else "#7f7f7f")
            endpoint = _infer_relation_endpoint(tokens, types, mask, i)
            if endpoint >= 0:
                ex, ey = float(tokens[endpoint, 0].item()), float(tokens[endpoint, 1].item())
                ax.plot([0, ex], [0, ey], color=color, lw=1.2 if bool(selected[i]) else 0.7, alpha=0.75 if bool(selected[i]) else 0.25, zorder=1)
            rel_marker = "X" if bool(fp_mask[i]) else "D"
            ax.scatter(
                x,
                y,
                s=130 if bool(selected[i]) else 55,
                marker=rel_marker,
                c=color,
                edgecolors="#111111" if bool(selected[i]) else color,
                linewidths=1.4 if bool(selected[i]) else 0.6,
                alpha=0.95 if bool(selected[i]) else 0.35,
                zorder=4,
            )
            if bool(selected[i]):
                ax.text(x + 0.6, y + 0.6, f"r{i}\nc={float(proxy_conf[i]):.2f}", fontsize=6.5, color=color)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-38, 38)
    ax.set_ylim(-38, 38)
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_xlabel("x [m] ego frame", fontsize=8.5)
    ax.set_ylabel("y [m] ego frame", fontsize=8.5)
    ax.tick_params(labelsize=7)


def _plot_case(case: Dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    globals()["plt"] = plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), dpi=170)
    door_fp = len(case["door_fp_selected"])
    plus_fp = len(case["doorplus_fp_selected"])
    door_true = len(case["door_true_risk_selected"])
    plus_true = len(case["doorplus_true_risk_selected"])
    _plot_panel(
        axes[0],
        case,
        "door_selected",
        f"Baseline DOOR under rel-fp noise\nselected FP rel: {door_fp}, true-risk rel: {door_true}",
    )
    _plot_panel(
        axes[1],
        case,
        "doorplus_selected",
        f"DOOR+ proxy confidence\nselected FP rel: {plus_fp}, true-risk rel: {plus_true}",
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=7, label="dynamic agent"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=9, label="false-positive relation"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markeredgecolor="#2ca02c", markersize=8, label="true-risk relation"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markeredgecolor="#111111", markersize=9, label="selected token"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(
        "DOOR+ suppresses unreliable relation tokens while retaining high-risk relations",
        fontsize=12.5,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out_dir / "doorplus_fp_relation_case.png", bbox_inches="tight")
    fig.savefig(out_dir / "doorplus_fp_relation_case.pdf", bbox_inches="tight")
    plt.close(fig)


def _json_case(case: Dict) -> Dict:
    keys = [
        "score",
        "dataset_index",
        "scene_id",
        "door_fp_selected",
        "doorplus_fp_selected",
        "door_true_risk_selected",
        "doorplus_true_risk_selected",
        "suppressed_fp_relations",
    ]
    out = {key: case[key] for key in keys}
    out["suppressed_fp_proxy_confidence"] = {
        str(idx): float(case["proxy_confidence"][idx].item())
        for idx in case["suppressed_fp_relations"]
    }
    out["doorplus_true_risk_proxy_confidence"] = {
        str(idx): float(case["proxy_confidence"][idx].item())
        for idx in case["doorplus_true_risk_selected"]
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuplan-root", default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument("--nuplan-index-json", default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"))
    parser.add_argument("--door-ckpt", type=Path, default=ROOT / "experiments" / "nuplan_stage1_50k" / "seed7" / "wm_decoupled_no_vis" / "model.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "tpami_qualitative_cases" / "doorplus_fp_relation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--fp-rate", type=float, default=0.2)
    parser.add_argument("--fp-confidence", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _build_loader(args, config)
    door_model = _load_model(config, args.door_ckpt, "object_relation_decoupled", device)
    plus_model = _load_model(config, args.door_ckpt, "object_relation_decoupled_uncertainty", device)
    case = _find_case(args, door_model, plus_model, loader, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "case.json").write_text(json.dumps(_json_case(case), indent=2))
    _plot_case(case, args.out_dir)
    print(json.dumps(_json_case(case), indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
