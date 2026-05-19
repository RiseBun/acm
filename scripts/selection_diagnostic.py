"""Selection diagnostics for DOOR-RL selectors.

This script is read-only with respect to training: it loads existing
checkpoints, forwards the validation split, logs per-token selector behavior,
and derives mechanism metrics such as CDR, miss rate, wasted relation slots,
and relation over-allocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models.abstraction import DecisionSufficientAbstraction
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_CONDITIONS: Dict[str, str] = {
    "bc": "object_only",
    "wm_object": "object_only",
    "wm_naive": "object_relation",
    "wm_decoupled": "object_relation_decoupled_visibility",
    "wm_decoupled_no_vis": "object_relation_decoupled",
}

_DYNAMIC_TYPES = {
    int(TokenType.EGO),
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
}
_AGENT_TYPES = {
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
}
_RARE_TYPES = {int(TokenType.PEDESTRIAN), int(TokenType.CYCLIST)}
_TYPE_NAMES = {
    int(TokenType.EGO): "ego",
    int(TokenType.VEHICLE): "vehicle",
    int(TokenType.PEDESTRIAN): "pedestrian",
    int(TokenType.CYCLIST): "cyclist",
    int(TokenType.MAP): "map",
    int(TokenType.SIGNAL): "signal",
    int(TokenType.RELATION): "relation",
    int(TokenType.PAD): "pad",
}

_TTC_DIM = 8
_LANE_CONFLICT_DIM = 9
_VISIBILITY_DIM = 7
_DISTANCE_DIM = 11
_POS_VEL_DIMS = 4
_CRITICAL_RADIUS_M = 20.0
_TTC_CRITICAL_SEC = 3.0
_ENDPOINT_MATCH_EPS_M = 2.0
_INTERACTION_RECALL_DIST_M = 1.0


@dataclass
class IndexedItem:
    item: Dict[str, torch.Tensor]
    sample_index: int
    scene_id: str


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], scene_names: Sequence[str]):
        self.dataset = dataset
        self.indices = list(indices)
        self.scene_names = list(scene_names)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> IndexedItem:
        sample_index = self.indices[idx]
        return IndexedItem(
            item=self.dataset[sample_index],
            sample_index=sample_index,
            scene_id=self.scene_names[sample_index],
        )


@dataclass
class MetaBatch:
    batch: SceneBatch
    sample_indices: List[int]
    scene_ids: List[str]

    def to(self, device: torch.device | str) -> "MetaBatch":
        return MetaBatch(
            batch=self.batch.to(device),
            sample_indices=self.sample_indices,
            scene_ids=self.scene_ids,
        )


def _collate_indexed(items: Sequence[IndexedItem]) -> MetaBatch:
    return MetaBatch(
        batch=SceneBatch.collate([x.item for x in items]),
        sample_indices=[x.sample_index for x in items],
        scene_ids=[x.scene_id for x in items],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log selector decisions and mechanism diagnostics."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--dataset", choices=["nuplan", "nuscenes"], default="nuplan")
    parser.add_argument(
        "--nuplan-root",
        default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split",
    )
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument(
        "--nuplan-index-json",
        default=str(ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"),
    )
    parser.add_argument("--nuscenes-root", default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument(
        "--token-cache-dir",
        default=str(ROOT / "experiments" / "_token_cache"),
    )
    parser.add_argument("--num-scenes", type=int, default=700)
    parser.add_argument(
        "--checkpoint-root",
        default=str(ROOT / "experiments" / "nuplan_stage1_50k"),
        help="Root with seed*/condition/model.pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "selection_diagnostic"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["wm_object", "wm_decoupled_no_vis"],
        choices=list(_CONDITIONS),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Optional cap for quick diagnostics. 0 uses the full val split.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip condition/seed pairs whose checkpoint is absent.",
    )
    return parser.parse_args()


def _build_val_loader(args: argparse.Namespace, config: DoorRLConfig, seed: int) -> DataLoader:
    if args.dataset == "nuplan":
        dataset = NuPlanPreprocessedDataset(
            config=config,
            data_root=args.nuplan_root,
            num_samples=args.nuplan_num_samples,
            index_json=args.nuplan_index_json,
            seed=seed,
            materialize_cache=False,
        )
    else:
        dataset = NuScenesSceneDataset(
            config=config,
            nuscenes_root=args.nuscenes_root,
            num_scenes=args.num_scenes,
            version="v1.0-trainval",
            cache_dir=args.token_cache_dir or None,
        )

    names = sorted(set(dataset.cache_scene_names))
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * len(shuffled)))) if len(shuffled) > 1 else 0
    val_scenes = set(shuffled[len(shuffled) - n_val:])
    val_idx = dataset.indices_for_scenes(val_scenes)
    if args.max_val_samples and args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]

    subset = IndexedSubset(dataset, val_idx, dataset.cache_scene_names)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=_collate_indexed,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    print(
        f"[loader] seed={seed} val_scenes={len(val_scenes)} "
        f"val_samples={len(val_idx)}",
        flush=True,
    )
    return DataLoader(subset, shuffle=False, **loader_kwargs)


def _load_model(
    args: argparse.Namespace,
    config: DoorRLConfig,
    seed: int,
    condition: str,
    device: torch.device,
) -> DoorRLModelVariant:
    variant = _CONDITIONS[condition]
    model = DoorRLModelVariant(config.model, ModelVariant(variant))
    ckpt = Path(args.checkpoint_root) / f"seed{seed}" / condition / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _score_tokens(
    abstraction: DecisionSufficientAbstraction,
    latent: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    ego = latent[:, :1, :]
    query = abstraction.query_proj(ego)
    keys = abstraction.key_proj(latent)
    similarity = (query * keys).sum(dim=-1) / math.sqrt(latent.size(-1))
    saliency = abstraction.score_proj(latent).squeeze(-1)
    scores = similarity + saliency
    return scores.masked_fill(~token_mask, float("nan"))


def _selector_scores(model: DoorRLModelVariant, batch: SceneBatch) -> Tuple[torch.Tensor, List[str]]:
    """Return per-token score and selector-head name for each token.

    For decoupled variants, dynamic and relation scores come from different
    heads and should be compared within type/head, not as one global scale.
    """
    latent = model.encoder(batch.tokens, batch.token_types)
    B, S, _ = latent.shape
    scores = torch.full((B, S), float("nan"), device=latent.device)
    heads = ["none"] * S
    variant = model.variant

    if variant == ModelVariant.OBJECT_ONLY:
        mask = batch.token_mask.clone()
        mask &= batch.token_types != int(TokenType.RELATION)
        scores = _score_tokens(model.abstraction, latent, mask)
        heads = ["shared_object"] * S
    elif variant == ModelVariant.OBJECT_RELATION:
        scores = _score_tokens(model.abstraction, latent, batch.token_mask)
        heads = ["shared_object_relation"] * S
    elif variant == ModelVariant.OBJECT_RELATION_VISIBILITY:
        visibility = batch.tokens[:, :, _VISIBILITY_DIM : _VISIBILITY_DIM + 1].clamp(0.0, 1.0)
        scores = _score_tokens(model.abstraction, latent * visibility, batch.token_mask)
        heads = ["shared_visibility"] * S
    elif variant in (
        ModelVariant.OBJECT_RELATION_DECOUPLED,
        ModelVariant.OBJECT_RELATION_DECOUPLED_VISIBILITY,
    ):
        dyn_mask = torch.zeros_like(batch.token_mask, dtype=torch.bool)
        for t in _DYNAMIC_TYPES:
            dyn_mask |= batch.token_types == t
        dyn_mask &= batch.token_mask
        rel_mask = (batch.token_types == int(TokenType.RELATION)) & batch.token_mask
        if getattr(model, "use_decoupled_visibility", False):
            visibility = batch.tokens[:, :, _VISIBILITY_DIM : _VISIBILITY_DIM + 1].clamp(0.0, 1.0)
            dyn_latent = latent * visibility
        else:
            dyn_latent = latent
        dyn_scores = _score_tokens(model.abstraction_dyn, dyn_latent, dyn_mask)
        rel_scores = _score_tokens(model.abstraction_rel, latent, rel_mask)
        scores = torch.where(dyn_mask, dyn_scores, scores)
        scores = torch.where(rel_mask, rel_scores, scores)
        heads = [
            "decoupled_dyn_or_rel" for _ in range(S)
        ]
    else:
        # Holistic variants do not expose a comparable top-k token selector.
        heads = ["holistic"] * S
    return scores, heads


def _selected_token_mask(output, batch: SceneBatch) -> torch.Tensor:
    selected = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    idx = output.abstraction.selected_indices
    mask = output.abstraction.selected_mask.bool()
    selected.scatter_(1, torch.where(mask, idx, torch.zeros_like(idx)), mask)
    return selected & batch.token_mask


def _dynamic_token_indices(token_types: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(token_mask, dtype=torch.bool)
    for t in _DYNAMIC_TYPES:
        mask |= token_types == t
    return torch.nonzero(mask & token_mask, as_tuple=False).flatten()


def _agent_token_indices(token_types: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(token_mask, dtype=torch.bool)
    for t in _AGENT_TYPES:
        mask |= token_types == t
    return torch.nonzero(mask & token_mask, as_tuple=False).flatten()


def _relation_token_indices(token_types: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(
        (token_types == int(TokenType.RELATION)) & token_mask,
        as_tuple=False,
    ).flatten()


def _infer_relation_endpoints(tokens: torch.Tensor, token_types: torch.Tensor, token_mask: torch.Tensor) -> Dict[int, Tuple[int, int]]:
    """Infer relation endpoints from the current token layout.

    Current adapters create ego-object relation tokens whose (x, y) equals the
    object's ego-relative position. Since endpoint ids are not serialized in
    the 40-dim token, we recover endpoint_j by nearest dynamic-agent position.
    """
    endpoints: Dict[int, Tuple[int, int]] = {}
    agent_idx = _agent_token_indices(token_types, token_mask)
    rel_idx = _relation_token_indices(token_types, token_mask)
    if agent_idx.numel() == 0:
        for ridx in rel_idx.tolist():
            endpoints[int(ridx)] = (0, -1)
        return endpoints

    agent_xy = tokens[agent_idx, :2]
    for ridx in rel_idx.tolist():
        rel_xy = tokens[ridx, :2]
        dist = torch.sqrt(((agent_xy - rel_xy) ** 2).sum(dim=1) + 1e-12)
        nearest_local = int(dist.argmin().item())
        endpoint_j = int(agent_idx[nearest_local].item())
        if float(dist[nearest_local].item()) > _ENDPOINT_MATCH_EPS_M:
            endpoint_j = -1
        endpoints[int(ridx)] = (0, endpoint_j)
    return endpoints


def _relation_features_by_endpoint(
    tokens: torch.Tensor,
    token_types: torch.Tensor,
    token_mask: torch.Tensor,
    endpoints: Dict[int, Tuple[int, int]],
) -> Dict[int, Dict[str, float]]:
    features: Dict[int, Dict[str, float]] = {}
    for ridx, (_, endpoint_j) in endpoints.items():
        if endpoint_j < 0:
            continue
        features[endpoint_j] = {
            "ttc": float(tokens[ridx, _TTC_DIM].item()),
            "lane_conflict": float(tokens[ridx, _LANE_CONFLICT_DIM].item()),
        }
    return features


def _nearest_match(
    pred_xyv: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_mask.numel() == 0 or int(pred_mask.sum().item()) == 0:
        inf = torch.full((gt_xy.size(0),), float("inf"), device=gt_xy.device)
        zero = torch.zeros(gt_xy.size(0), device=gt_xy.device, dtype=torch.long)
        return inf, zero
    valid_idx = torch.nonzero(pred_mask, as_tuple=False).flatten()
    valid_xy = pred_xyv[valid_idx, :2]
    dist = torch.sqrt(((gt_xy.unsqueeze(1) - valid_xy.unsqueeze(0)) ** 2).sum(dim=-1) + 1e-12)
    min_dist, local = dist.min(dim=1)
    return min_dist, valid_idx[local]


def _per_sample_downstream_metrics(batch: SceneBatch, output, sample_i: int) -> Dict[str, float]:
    token_types = batch.token_types[sample_i]
    token_mask = batch.token_mask[sample_i]
    tokens = batch.tokens[sample_i]
    next_tokens = batch.next_tokens[sample_i]

    selected_idx = output.abstraction.selected_indices[sample_i]
    selected_mask = output.abstraction.selected_mask[sample_i].bool()
    pred_next = output.world_model.predicted_next_tokens[sample_i]
    if bool(getattr(output.abstraction, "is_set_prediction", False)):
        pred_dyn_mask = selected_mask
    else:
        slot_types = token_types.gather(0, selected_idx)
        pred_dyn_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
        for t in _DYNAMIC_TYPES:
            pred_dyn_mask |= slot_types == t
        pred_dyn_mask &= selected_mask

    dyn_mask = torch.zeros_like(token_mask, dtype=torch.bool)
    for t in _DYNAMIC_TYPES:
        dyn_mask |= token_types == t
    dyn_mask &= token_mask
    if int(dyn_mask.sum().item()) == 0:
        rare_ade = float("nan")
        int_recall = float("nan")
    else:
        gt_next = next_tokens[dyn_mask, :_POS_VEL_DIMS]
        cur = tokens[dyn_mask, :2]
        types = token_types[dyn_mask]
        min_dist, _ = _nearest_match(pred_next[:, :_POS_VEL_DIMS], pred_dyn_mask, gt_next[:, :2])
        rare = torch.zeros_like(types, dtype=torch.bool)
        for t in _RARE_TYPES:
            rare |= types == t
        if int(rare.sum().item()) > 0:
            rare_ade = float(min_dist[rare].mean().item())
            rare_cur = cur[rare]
            rare_dist = torch.sqrt((rare_cur * rare_cur).sum(dim=1) + 1e-12)
            interactive = rare_dist < _CRITICAL_RADIUS_M
            if int(interactive.sum().item()) > 0:
                int_recall = float(
                    (min_dist[rare][interactive] < _INTERACTION_RECALL_DIST_M)
                    .float()
                    .mean()
                    .item()
                )
            else:
                int_recall = float("nan")
        else:
            rare_ade = float("nan")
            int_recall = float("nan")

    action_diff = output.policy.action_mean[sample_i] - batch.actions[sample_i]
    relation_mask = (token_types == int(TokenType.RELATION)) & token_mask
    relation_ttc = torch.where(
        relation_mask,
        tokens[:, _TTC_DIM],
        torch.full_like(tokens[:, _TTC_DIM], float("inf")),
    )
    collision_label = float((relation_ttc < _TTC_CRITICAL_SEC).any().item())
    collision_prob = float(torch.sigmoid(output.world_model.predicted_collision[sample_i]).item())
    return {
        "rare_ade": rare_ade,
        "interaction_recall_at_1m": int_recall,
        "teacher_action_mse": float(action_diff.pow(2).mean().item()),
        "teacher_action_delta_l2": float(action_diff.norm().item()),
        "collision_label": collision_label,
        "collision_prob": collision_prob,
    }


def _sample_mechanism_metrics(
    batch: SceneBatch,
    output,
    selected: torch.Tensor,
    sample_i: int,
) -> Dict[str, float]:
    tokens = batch.tokens[sample_i]
    token_types = batch.token_types[sample_i]
    token_mask = batch.token_mask[sample_i]
    selected_i = selected[sample_i]

    endpoints = _infer_relation_endpoints(tokens, token_types, token_mask)
    rel_by_endpoint = _relation_features_by_endpoint(tokens, token_types, token_mask, endpoints)
    dyn_idx = _dynamic_token_indices(token_types, token_mask)
    agent_idx = _agent_token_indices(token_types, token_mask)
    rel_idx = _relation_token_indices(token_types, token_mask)

    critical: List[int] = []
    for idx in agent_idx.tolist():
        x = float(tokens[idx, 0].item())
        y = float(tokens[idx, 1].item())
        distance = math.hypot(x, y)
        rel_features = rel_by_endpoint.get(idx, {})
        ttc = rel_features.get("ttc", float(tokens[idx, _TTC_DIM].item()))
        lane_conflict = rel_features.get("lane_conflict", float(tokens[idx, _LANE_CONFLICT_DIM].item()))
        is_rare = int(token_types[idx].item()) in _RARE_TYPES
        if (
            distance < _CRITICAL_RADIUS_M
            or ttc < _TTC_CRITICAL_SEC
            or lane_conflict > 0.5
            or (is_rare and distance < _CRITICAL_RADIUS_M)
        ):
            critical.append(int(idx))

    if critical:
        retained = sum(1 for idx in critical if bool(selected_i[idx].item()))
        cdr = retained / float(len(critical))
    else:
        retained = 0
        cdr = float("nan")
    miss_rate = 1.0 - cdr if math.isfinite(cdr) else float("nan")

    selected_dyn = {
        int(i) for i in dyn_idx.tolist() if bool(selected_i[int(i)].item())
    }
    selected_rel = [
        int(i) for i in rel_idx.tolist() if bool(selected_i[int(i)].item())
    ]
    wasted = 0
    for ridx in selected_rel:
        endpoint_i, endpoint_j = endpoints.get(ridx, (-1, -1))
        if endpoint_i == 0 and endpoint_j >= 0:
            is_wasted = endpoint_j not in selected_dyn
        else:
            is_wasted = endpoint_i not in selected_dyn and endpoint_j not in selected_dyn
        wasted += int(is_wasted)
    wasted_rate = wasted / float(len(selected_rel)) if selected_rel else float("nan")

    k_total = int(output.abstraction.selected_mask[sample_i].sum().item())
    roi = len(selected_rel) / float(k_total) if k_total > 0 else float("nan")
    n_agents = int(agent_idx.numel())
    n_rel = int(rel_idx.numel())
    rel_density = n_rel / float(n_agents + 1)
    selected_relation_ratio = roi

    downstream = _per_sample_downstream_metrics(batch, output, sample_i)
    return {
        "n_dynamic": int(dyn_idx.numel()),
        "n_agents": n_agents,
        "n_relations": n_rel,
        "n_critical_dynamic": len(critical),
        "n_critical_retained": retained,
        "cdr": cdr,
        "critical_agent_miss_rate": miss_rate,
        "selected_dynamic_count": len(selected_dyn),
        "selected_relation_count": len(selected_rel),
        "relation_wasted_slot_count": wasted,
        "relation_wasted_slot_rate": wasted_rate,
        "relation_overallocation_index": roi,
        "selected_relation_ratio": selected_relation_ratio,
        "relation_density": rel_density,
        **downstream,
    }


def _fmt_float(value: float) -> str:
    if value is None:
        return ""
    try:
        if not math.isfinite(float(value)):
            return ""
    except Exception:
        return ""
    return f"{float(value):.8g}"


def _write_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    clean[key] = _fmt_float(value)
                else:
                    clean[key] = value
            writer.writerow(clean)


@torch.no_grad()
def run_condition(
    args: argparse.Namespace,
    config: DoorRLConfig,
    seed: int,
    condition: str,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> Optional[Dict[str, float]]:
    try:
        model = _load_model(args, config, seed, condition, device)
    except FileNotFoundError as exc:
        if args.skip_missing:
            print(f"[skip] {exc}", flush=True)
            return None
        raise

    token_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    variant = _CONDITIONS[condition]
    sample_counter = 0

    for meta in val_loader:
        meta = meta.to(device)
        batch = meta.batch
        output = model(batch)
        scores, heads = _selector_scores(model, batch)
        selected = _selected_token_mask(output, batch)

        B, S = batch.tokens.shape[:2]
        for b in range(B):
            sample_id = f"{args.dataset}:{meta.scene_ids[b]}:{meta.sample_indices[b]}"
            sample_metrics = _sample_mechanism_metrics(batch, output, selected, b)
            sample_rows.append({
                "sample_id": sample_id,
                "dataset": args.dataset,
                "scene_id": meta.scene_ids[b],
                "sample_index": meta.sample_indices[b],
                "seed": seed,
                "condition": condition,
                "variant": variant,
                **sample_metrics,
            })

            endpoints = _infer_relation_endpoints(
                batch.tokens[b],
                batch.token_types[b],
                batch.token_mask[b],
            )
            rel_features = _relation_features_by_endpoint(
                batch.tokens[b],
                batch.token_types[b],
                batch.token_mask[b],
                endpoints,
            )
            for j in range(S):
                if not bool(batch.token_mask[b, j].item()):
                    continue
                token_type = int(batch.token_types[b, j].item())
                x = float(batch.tokens[b, j, 0].item())
                y = float(batch.tokens[b, j, 1].item())
                endpoint_i, endpoint_j = endpoints.get(j, (-1, -1))
                rel_info = rel_features.get(j, {})
                ttc_proxy = (
                    float(batch.tokens[b, j, _TTC_DIM].item())
                    if token_type == int(TokenType.RELATION)
                    else rel_info.get("ttc", float(batch.tokens[b, j, _TTC_DIM].item()))
                )
                lane_conflict = (
                    float(batch.tokens[b, j, _LANE_CONFLICT_DIM].item())
                    if token_type == int(TokenType.RELATION)
                    else rel_info.get("lane_conflict", float(batch.tokens[b, j, _LANE_CONFLICT_DIM].item()))
                )
                token_rows.append({
                    "sample_id": sample_id,
                    "dataset": args.dataset,
                    "scene_id": meta.scene_ids[b],
                    "sample_index": meta.sample_indices[b],
                    "seed": seed,
                    "condition": condition,
                    "variant": variant,
                    "token_id": j,
                    "token_type": _TYPE_NAMES.get(token_type, str(token_type)),
                    "selector_head": heads[j] if j < len(heads) else "",
                    "score": float(scores[b, j].item()) if torch.isfinite(scores[b, j]) else float("nan"),
                    "selected": int(bool(selected[b, j].item())),
                    "distance_to_ego": math.hypot(x, y),
                    "ttc_proxy": ttc_proxy,
                    "lane_conflict": lane_conflict,
                    "visibility": float(batch.tokens[b, j, _VISIBILITY_DIM].item()),
                    "is_rare_agent": int(token_type in _RARE_TYPES),
                    "relation_endpoint_i": endpoint_i if token_type == int(TokenType.RELATION) else "",
                    "relation_endpoint_j": endpoint_j if token_type == int(TokenType.RELATION) else "",
                })
        sample_counter += B
        if sample_counter % max(args.batch_size * 10, 1) == 0:
            print(
                f"[diag] seed={seed} condition={condition} samples={sample_counter}",
                flush=True,
            )

    pair_dir = out_dir / f"seed{seed}" / condition
    token_fields = [
        "sample_id", "dataset", "scene_id", "sample_index", "seed",
        "condition", "variant", "token_id", "token_type", "selector_head",
        "score", "selected", "distance_to_ego", "ttc_proxy",
        "lane_conflict", "visibility", "is_rare_agent",
        "relation_endpoint_i", "relation_endpoint_j",
    ]
    sample_fields = [
        "sample_id", "dataset", "scene_id", "sample_index", "seed",
        "condition", "variant", "n_dynamic", "n_agents", "n_relations",
        "n_critical_dynamic", "n_critical_retained", "cdr",
        "critical_agent_miss_rate", "selected_dynamic_count",
        "selected_relation_count", "relation_wasted_slot_count",
        "relation_wasted_slot_rate", "relation_overallocation_index",
        "selected_relation_ratio", "relation_density", "rare_ade",
        "interaction_recall_at_1m", "teacher_action_mse",
        "teacher_action_delta_l2", "collision_label", "collision_prob",
    ]
    _write_rows(pair_dir / "token_selection_log.csv", token_rows, token_fields)
    _write_rows(pair_dir / "sample_mechanism_metrics.csv", sample_rows, sample_fields)
    summary = _summarize_sample_rows(sample_rows)
    (pair_dir / "summary.json").write_text(json.dumps(_json_sanitize(summary), indent=2))
    _plot_sample_rows(sample_rows, pair_dir / "figures", title=f"{condition} seed{seed}")
    print(f"[diag] wrote {pair_dir}", flush=True)
    return summary


def _finite_pairs(rows: Sequence[Dict[str, object]], x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _json_sanitize(value):
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summarize_sample_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    scalar_keys = [
        "cdr",
        "critical_agent_miss_rate",
        "relation_wasted_slot_rate",
        "relation_overallocation_index",
        "selected_relation_ratio",
        "relation_density",
        "rare_ade",
        "interaction_recall_at_1m",
        "teacher_action_mse",
        "collision_label",
        "collision_prob",
    ]
    summary: Dict[str, object] = {"n_samples": len(rows), "mean": {}, "correlation": {}}
    for key in scalar_keys:
        vals = []
        for row in rows:
            try:
                val = float(row[key])
            except Exception:
                continue
            if math.isfinite(val):
                vals.append(val)
        summary["mean"][key] = float(np.mean(vals)) if vals else float("nan")

    for y_key in [
        "rare_ade",
        "interaction_recall_at_1m",
        "collision_label",
        "collision_prob",
        "teacher_action_mse",
    ]:
        x, y = _finite_pairs(rows, "critical_agent_miss_rate", y_key)
        summary["correlation"][f"miss_rate__vs__{y_key}"] = {
            "n": int(x.size),
            "pearson": _pearson(x, y),
            "spearman": _spearman(x, y),
        }
    for y_key in ["selected_relation_ratio", "cdr", "rare_ade"]:
        x, y = _finite_pairs(rows, "relation_density", y_key)
        summary["correlation"][f"relation_density__vs__{y_key}"] = {
            "n": int(x.size),
            "pearson": _pearson(x, y),
            "spearman": _spearman(x, y),
        }
    return summary


def _plot_sample_rows(rows: Sequence[Dict[str, object]], out_dir: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def scatter(x_key: str, y_key: str, filename: str, xlabel: str, ylabel: str) -> None:
        x, y = _finite_pairs(rows, x_key, y_key)
        if x.size == 0:
            return
        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=160)
        ax.scatter(x, y, s=10, alpha=0.35, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nr={_pearson(x, y):.3f}, rho={_spearman(x, y):.3f}, n={x.size}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    scatter("critical_agent_miss_rate", "rare_ade", "missrate_vs_rare_ade.png", "Critical Agent Miss Rate", "Rare ADE")
    scatter("critical_agent_miss_rate", "interaction_recall_at_1m", "missrate_vs_interaction_recall.png", "Critical Agent Miss Rate", "Interaction Recall@1m")
    scatter("critical_agent_miss_rate", "collision_prob", "missrate_vs_collision_prob.png", "Critical Agent Miss Rate", "Collision probability")
    scatter("critical_agent_miss_rate", "teacher_action_mse", "missrate_vs_teacher_action_mse.png", "Critical Agent Miss Rate", "Teacher Action MSE")
    scatter("relation_density", "selected_relation_ratio", "reldensity_vs_selected_relation_ratio.png", "Relation Density", "Selected relation ratio / ROI")
    scatter("relation_density", "cdr", "reldensity_vs_cdr.png", "Relation Density", "CDR")
    scatter("relation_density", "rare_ade", "reldensity_vs_rare_ade.png", "Relation Density", "Rare ADE")


def _write_top_summary(out_dir: Path, results: Dict[str, Dict[str, Dict[str, object]]]) -> None:
    payload = {"results": results}
    (out_dir / "summary.json").write_text(json.dumps(_json_sanitize(payload), indent=2))
    lines = [
        "# Selection Diagnostic Summary",
        "",
        "| condition | seed | samples | CDR ↑ | MissRate ↓ | WastedRel ↓ | ROI | RelDensity | Miss~RareADE rho | Miss~ActionMSE rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, by_seed in results.items():
        for seed, summary in by_seed.items():
            mean = summary["mean"]
            corr = summary["correlation"]
            lines.append(
                f"| {condition} | {seed} | {summary['n_samples']} | "
                f"{_fmt_float(mean['cdr'])} | "
                f"{_fmt_float(mean['critical_agent_miss_rate'])} | "
                f"{_fmt_float(mean['relation_wasted_slot_rate'])} | "
                f"{_fmt_float(mean['relation_overallocation_index'])} | "
                f"{_fmt_float(mean['relation_density'])} | "
                f"{_fmt_float(corr['miss_rate__vs__rare_ade']['spearman'])} | "
                f"{_fmt_float(corr['miss_rate__vs__teacher_action_mse']['spearman'])} |"
            )
    lines.extend([
        "",
        "Notes:",
        "",
        "- `CDR = |S_dyn ∩ C_dyn| / |C_dyn|`; `MissRate = 1 - CDR`.",
        "- Current adapters serialize ego-object relation features, but not endpoint ids. Endpoint `j` is inferred by nearest dynamic-token `(x,y)`; unmatched endpoints are `-1`.",
        "- For ego-object relations, a selected relation is counted as wasted when its non-ego endpoint is not selected.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results: Dict[str, Dict[str, Dict[str, object]]] = {
        condition: {} for condition in args.conditions
    }
    for seed in args.seeds:
        set_seed(seed)
        config.seed = seed
        val_loader = _build_val_loader(args, config, seed)
        for condition in args.conditions:
            print(f"[diag] seed={seed} condition={condition}", flush=True)
            summary = run_condition(
                args=args,
                config=config,
                seed=seed,
                condition=condition,
                val_loader=val_loader,
                device=device,
                out_dir=out_dir,
            )
            if summary is not None:
                all_results[condition][str(seed)] = summary
    _write_top_summary(out_dir, all_results)
    print(f"[diag] wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
