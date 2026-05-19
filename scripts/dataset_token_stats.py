"""Dataset token/statistics analysis for nuScenes vs nuPlan.

The goal is explanatory rather than evaluative: quantify how token density,
rare-agent presence, visibility, relation features, and short-horizon motion
differ across datasets so the Stage-1 ranking reversal has concrete context.
"""
from __future__ import annotations

import argparse
import json
import math
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
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "dataset_token_stats"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=16)
    parser.add_argument("--nuscenes-root", default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument("--nuscenes-num-scenes", type=int, default=700)
    parser.add_argument(
        "--token-cache-dir",
        default=str(ROOT / "experiments" / "_token_cache"),
    )
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
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap per dataset for quick debugging. 0 uses all selected samples.",
    )
    return parser.parse_args()


def _loader_kwargs(args: argparse.Namespace, batch_size: int) -> Dict:
    kwargs = dict(
        batch_size=batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return kwargs


def _build_nuscenes_loader(args: argparse.Namespace, config: DoorRLConfig):
    dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        num_scenes=args.nuscenes_num_scenes,
        version="v1.0-trainval",
        cache_dir=args.token_cache_dir or None,
    )
    indices = list(range(len(dataset)))
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]
    return DataLoader(
        Subset(dataset, indices),
        shuffle=False,
        **_loader_kwargs(args, config.training.batch_size),
    ), len(indices)


def _build_nuplan_loader(args: argparse.Namespace, config: DoorRLConfig):
    dataset = NuPlanPreprocessedDataset(
        config=config,
        data_root=args.nuplan_root,
        num_samples=args.nuplan_num_samples,
        index_json=args.nuplan_index_json,
        seed=args.seed,
        materialize_cache=False,
    )
    indices = list(range(len(dataset)))
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]
    return DataLoader(
        Subset(dataset, indices),
        shuffle=False,
        **_loader_kwargs(args, config.training.batch_size),
    ), len(indices)


class Running:
    def __init__(self) -> None:
        self.values: List[torch.Tensor] = []

    def add(self, tensor: torch.Tensor) -> None:
        self.values.append(tensor.detach().float().cpu().reshape(-1))

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan")}
        x = torch.cat(self.values)
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std().item()) if x.numel() > 1 else 0.0,
            "p50": float(torch.quantile(x, 0.50).item()),
            "p90": float(torch.quantile(x, 0.90).item()),
        }


def _count_type(types: torch.Tensor, mask: torch.Tensor, token_type: TokenType) -> torch.Tensor:
    return ((types == int(token_type)) & mask).sum(dim=1).float()


@torch.no_grad()
def collect_stats(name: str, loader: DataLoader, expected_samples: int) -> Dict:
    print(f"[stats] {name}: collecting {expected_samples} samples", flush=True)
    dyn_count = Running()
    vehicle_count = Running()
    ped_count = Running()
    cyc_count = Running()
    rare_count = Running()
    map_count = Running()
    relation_count = Running()
    visibility_dyn = Running()
    visibility_all = Running()
    relation_ttc = Running()
    relation_conflict = Running()
    relation_priority = Running()
    action_l2 = Running()
    action_abs0 = Running()
    action_abs1 = Running()
    ego_next_disp = Running()
    dyn_next_disp = Running()
    interactive_dyn_count = Running()
    n_samples = 0

    for batch in loader:
        tokens = batch.tokens
        next_tokens = batch.next_tokens
        mask = batch.token_mask
        types = batch.token_types
        actions = batch.actions

        vehicle = _count_type(types, mask, TokenType.VEHICLE)
        ped = _count_type(types, mask, TokenType.PEDESTRIAN)
        cyc = _count_type(types, mask, TokenType.CYCLIST)
        rare = ped + cyc
        dyn = vehicle + rare + _count_type(types, mask, TokenType.EGO)
        rel = _count_type(types, mask, TokenType.RELATION)

        dyn_mask = (
            (types == int(TokenType.EGO))
            | (types == int(TokenType.VEHICLE))
            | (types == int(TokenType.PEDESTRIAN))
            | (types == int(TokenType.CYCLIST))
        ) & mask
        rel_mask = (types == int(TokenType.RELATION)) & mask

        dyn_count.add(dyn)
        vehicle_count.add(vehicle)
        ped_count.add(ped)
        cyc_count.add(cyc)
        rare_count.add(rare)
        map_count.add(_count_type(types, mask, TokenType.MAP))
        relation_count.add(rel)
        visibility_dyn.add(tokens[:, :, 7][dyn_mask])
        visibility_all.add(tokens[:, :, 7][mask])
        relation_ttc.add(tokens[:, :, 8][rel_mask])
        relation_conflict.add(tokens[:, :, 9][rel_mask])
        relation_priority.add(tokens[:, :, 10][rel_mask])
        action_l2.add(actions.norm(dim=1))
        action_abs0.add(actions[:, 0].abs())
        action_abs1.add(actions[:, 1].abs())

        ego_disp = (next_tokens[:, 0, :2] - tokens[:, 0, :2]).norm(dim=1)
        ego_next_disp.add(ego_disp)
        dyn_disp = (next_tokens[:, :, :2] - tokens[:, :, :2]).norm(dim=-1)
        dyn_next_disp.add(dyn_disp[dyn_mask])
        interactive_dyn_count.add(((tokens[:, :, 13] > 0.5) & dyn_mask).sum(dim=1).float())
        n_samples += tokens.size(0)

    result = {
        "dataset": name,
        "n_samples": n_samples,
        "expected_samples": expected_samples,
        "metrics": {
            "dynamic_tokens_per_sample": dyn_count.summary(),
            "vehicle_tokens_per_sample": vehicle_count.summary(),
            "pedestrian_tokens_per_sample": ped_count.summary(),
            "cyclist_tokens_per_sample": cyc_count.summary(),
            "rare_tokens_per_sample": rare_count.summary(),
            "map_tokens_per_sample": map_count.summary(),
            "relation_tokens_per_sample": relation_count.summary(),
            "visibility_dynamic": visibility_dyn.summary(),
            "visibility_all_valid": visibility_all.summary(),
            "relation_ttc": relation_ttc.summary(),
            "relation_lane_conflict": relation_conflict.summary(),
            "relation_priority": relation_priority.summary(),
            "teacher_action_l2": action_l2.summary(),
            "teacher_action_abs_dim0": action_abs0.summary(),
            "teacher_action_abs_dim1": action_abs1.summary(),
            "ego_next_displacement": ego_next_disp.summary(),
            "dynamic_next_displacement": dyn_next_disp.summary(),
            "interactive_dynamic_tokens_per_sample": interactive_dyn_count.summary(),
        },
    }
    return result


def _fmt(block: Dict[str, float]) -> str:
    return f"{block['mean']:.3f} ± {block['std']:.3f} / p90 {block['p90']:.3f}"


def write_markdown(out_dir: Path, results: Dict[str, Dict]) -> None:
    nusc = results["nuscenes"]["metrics"]
    nup = results["nuplan_50k"]["metrics"]
    lines = [
        "# nuScenes vs nuPlan Token/Agent Statistics",
        "",
        "Purpose: explain why representation quality, Stage-1 policy learning, and downstream planner-like behavior need not rank variants identically.",
        "",
        "| Statistic | nuScenes 700 scenes | nuPlan 50k NPZ | Reading |",
        "|---|---:|---:|---|",
        (
            f"| Dynamic tokens / sample | {_fmt(nusc['dynamic_tokens_per_sample'])} | "
            f"{_fmt(nup['dynamic_tokens_per_sample'])} | planning density / interaction budget |"
        ),
        (
            f"| Rare tokens / sample | {_fmt(nusc['rare_tokens_per_sample'])} | "
            f"{_fmt(nup['rare_tokens_per_sample'])} | pedestrian/cyclist pressure |"
        ),
        (
            f"| Relation tokens / sample | {_fmt(nusc['relation_tokens_per_sample'])} | "
            f"{_fmt(nup['relation_tokens_per_sample'])} | relation-context availability |"
        ),
        (
            f"| Dynamic visibility | {_fmt(nusc['visibility_dynamic'])} | "
            f"{_fmt(nup['visibility_dynamic'])} | whether visibility weighting has signal |"
        ),
        (
            f"| Relation TTC | {_fmt(nusc['relation_ttc'])} | "
            f"{_fmt(nup['relation_ttc'])} | risk/interaction feature scale |"
        ),
        (
            f"| Teacher action L2 | {_fmt(nusc['teacher_action_l2'])} | "
            f"{_fmt(nup['teacher_action_l2'])} | action-label scale and policy target |"
        ),
        (
            f"| Ego next displacement | {_fmt(nusc['ego_next_displacement'])} | "
            f"{_fmt(nup['ego_next_displacement'])} | short-horizon motion scale |"
        ),
        (
            f"| Dynamic next displacement | {_fmt(nusc['dynamic_next_displacement'])} | "
            f"{_fmt(nup['dynamic_next_displacement'])} | world-model target scale |"
        ),
        "",
        "## Takeaways",
        "",
        "- Use these statistics as explanatory evidence, not as performance metrics.",
        "- The main question is whether nuPlan has denser or cleaner relation/action structure, which can make decoupled relation-aware abstraction more useful for policy learning.",
        "- Visibility should be interpreted as a dataset-specific inductive bias: if its distribution carries little contrast or interacts poorly with action labels, no-vis can be better.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size

    nusc_loader, nusc_n = _build_nuscenes_loader(args, config)
    nuplan_loader, nuplan_n = _build_nuplan_loader(args, config)

    results = {
        "setup": vars(args),
        "nuscenes": collect_stats("nuscenes", nusc_loader, nusc_n),
        "nuplan_50k": collect_stats("nuplan_50k", nuplan_loader, nuplan_n),
    }
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    write_markdown(out_dir, results)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
