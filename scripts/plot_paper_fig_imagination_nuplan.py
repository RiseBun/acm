#!/usr/bin/env python
"""Paper-grade Figure 4b: nuPlan Stage 1 imagination on the
``lane_conflict`` subset.

Loads ``experiments/nuplan_stage1_50k/seed7/<cond>/model.pt`` for
``wm_object`` and ``wm_decoupled_no_vis``, runs deterministic K-step
imagination on the seed-7 nuPlan val split, masks to the
``lane_conflict`` subset (``tokens[:, :, 9] > 0.5`` on any RELATION
slot, matching ``interaction_subset_analysis.py``), and produces a
2-row figure:

  Row 1 (aggregate over lane_conflict samples)
    A. P(collision) per rollout step (mean ± std)
    B. ||a_mean||₂ per rollout step
    C. ego-slot cosine step distance (1 − cos(eₜ, eₜ₊₁))

  Row 2 (per-sample representative cases on lane_conflict)
    D. cumulative latent return for picked samples
    E. per-step P(collision) for picked samples
    F. per-step action norm

Sample picking on the lane_conflict mask:
  * "decoupled-wins"  = sample with largest gap
                          return_decoupled - return_object
  * "object-collides" = sample where object's max-step coll > 0.5
                          and decoupled's max-step coll < 0.5
                          (largest gap in coll_max)
  * "typical"         = sample with smallest |coll_max_obj - coll_max_dec|
                          inside the lane_conflict subset.

Usage:
    PYTHONPATH=src python scripts/plot_paper_fig_imagination_nuplan.py \
        --max-val-samples 5000

The script also persists the per-step numerics in an .npz so reviewers
can audit the statistics independently of the figure styling.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import DEFAULT_REWARD_CFG
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


_CONDITIONS = {
    "wm_object":           "object_only",
    "wm_decoupled_no_vis": "object_relation_decoupled",
}
_DISPLAY = {
    "wm_object":           "Object-only",
    "wm_decoupled_no_vis": "Decoupled (no visibility)",
}
_COLOR = {
    "wm_object":           "#1f77b4",
    "wm_decoupled_no_vis": "#d62728",
}


def _build_val_loader(args, config, seed):
    full = NuPlanPreprocessedDataset(
        config=config,
        data_root=args.nuplan_root,
        num_samples=args.nuplan_num_samples,
        index_json=args.nuplan_index_json,
        seed=seed,
        materialize_cache=False,
    )
    names = sorted(set(full.cache_scene_names))
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled)))) if len(shuffled) > 1 else 0
    val_scenes = set(shuffled[len(shuffled) - n_val:])
    val_idx = full.indices_for_scenes(val_scenes)
    if args.max_val_samples and args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]
    print(f"[loader] seed={seed} val_scenes={len(val_scenes)} "
          f"val_samples={len(val_idx)}", flush=True)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    if args.loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(full, val_idx), shuffle=False, **loader_kwargs)


def _lane_conflict_mask(batch: SceneBatch) -> torch.Tensor:
    relation = (batch.token_types == int(TokenType.RELATION)) & batch.token_mask
    return ((batch.tokens[:, :, 9] > 0.5) & relation).any(dim=1)


def _load_model(cond, ckpt_path, config, device):
    variant_name = _CONDITIONS[cond]
    model = DoorRLModelVariant(config.model, ModelVariant(variant_name))
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _rollout_metrics(model, val_loader, device, horizon):
    """Per-sample per-step metrics on the *full* val loader.

    We compute on every sample so we can subset by lane_conflict at
    plot time without re-running the model.
    """
    all_coll, all_an, all_eg, all_ret, all_lc = [], [], [], [], []
    for bi, batch in enumerate(val_loader):
        batch = batch.to(device)
        traj = imagine_trajectory(
            model, batch,
            horizon=horizon, deterministic=True,
            reward_cfg=DEFAULT_REWARD_CFG,
            detach_world_model=True,
            action_sample_clip=5.0,
        )
        all_coll.append(traj.collisions.cpu())
        all_an.append(traj.action_means.norm(dim=-1).cpu())
        a = traj.ego_latents[:, :-1, :]
        b = traj.ego_latents[:, 1:, :]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-3)
        all_eg.append((1.0 - cos).cpu())
        all_ret.append(traj.rewards.cumsum(dim=1).cpu())
        all_lc.append(_lane_conflict_mask(batch).cpu())
        if bi % 10 == 0:
            print(f"  batch {bi}", flush=True)
    coll = torch.cat(all_coll, dim=0).numpy()
    an = torch.cat(all_an, dim=0).numpy()
    eg = torch.cat(all_eg, dim=0).numpy()
    ret = torch.cat(all_ret, dim=0).numpy()
    lc = torch.cat(all_lc, dim=0).numpy()
    return {
        "coll": coll, "action_norm": an, "ego_cos_step": eg,
        "ret_cumsum": ret,
        "coll_max": coll.max(axis=1),
        "return_sum": ret[:, -1],
        "lane_conflict_mask": lc,
    }


def _band(ax, x, vals, color, label, alpha=0.18):
    mu = vals.mean(axis=0)
    sd = vals.std(axis=0)
    ax.plot(x, mu, color=color, label=label, lw=2.0, marker="o", markersize=4)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=alpha, lw=0)


def _pick_samples(metrics, lc_mask) -> Dict[str, int]:
    """All picks restricted to lane_conflict samples."""
    obj = metrics["wm_object"]
    dec = metrics["wm_decoupled_no_vis"]
    n = obj["coll"].shape[0]
    valid = lc_mask
    if not valid.any():
        return {}

    return_gap = dec["return_sum"] - obj["return_sum"]
    return_gap_masked = np.where(valid, return_gap, -np.inf)
    decoupled_wins = int(np.argmax(return_gap_masked))

    coll_gap = obj["coll_max"] - dec["coll_max"]
    obj_collides = (obj["coll_max"] > 0.5) & (dec["coll_max"] < 0.5)
    coll_gap_masked = np.where(valid & obj_collides, coll_gap, -np.inf)
    if (valid & obj_collides).any():
        object_collides = int(np.argmax(coll_gap_masked))
    else:
        coll_gap_relaxed = np.where(valid, coll_gap, -np.inf)
        object_collides = int(np.argmax(coll_gap_relaxed))

    coll_diff_abs = np.abs(obj["coll_max"] - dec["coll_max"])
    typical_score = np.where(valid, coll_diff_abs, np.inf)
    typical = int(np.argmin(typical_score))

    return {
        "decoupled-wins":  decoupled_wins,
        "object-collides": object_collides,
        "typical":         typical,
    }


def _plot(metrics, picks, horizon, conditions, lc_mask, out_pdf, out_png):
    x_step = np.arange(1, horizon + 1)
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    titles_top = [
        "(A) Imagined collision probability",
        "(B) Policy mean action norm  ||a_mean||₂",
        "(C) Ego-slot cosine step distance  1 − cos(eₜ, eₜ₊₁)",
    ]
    keys_top = ["coll", "action_norm", "ego_cos_step"]
    ylabels_top = ["P(collision)", "||a_mean||₂", "1 − cos"]
    n_lc = int(lc_mask.sum())
    for ci, key in enumerate(keys_top):
        ax = fig.add_subplot(gs[0, ci])
        for cond in conditions:
            _band(ax, x_step, metrics[cond][key][lc_mask],
                  color=_COLOR[cond],
                  label=f"{_DISPLAY[cond]}  (lane_conflict, n={n_lc})")
        ax.set_xlabel("rollout step k")
        ax.set_ylabel(ylabels_top[ci])
        ax.set_title(titles_top[ci], fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(x_step)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left")
            ax.axhline(0.5, color="gray", lw=0.7, linestyle="--", alpha=0.6)

    titles_bot = [
        "(D) Cumulative latent return  (lane_conflict picks)",
        "(E) Per-step P(collision)  (lane_conflict picks)",
        "(F) Action norm  (lane_conflict picks)",
    ]
    pick_styles = {
        "object-collides": ("-", "object collides"),
        "decoupled-wins":  ("--", "decoupled wins"),
        "typical":         (":", "typical"),
    }
    keys_bot = ["ret_cumsum", "coll", "action_norm"]
    ylabels_bot = ["cum. reward", "P(collision)", "||a_mean||₂"]
    for ci, (key, ylabel, title) in enumerate(zip(keys_bot, ylabels_bot, titles_bot)):
        ax = fig.add_subplot(gs[1, ci])
        for label, idx in picks.items():
            ls, ldesc = pick_styles[label]
            for cond in conditions:
                vals = metrics[cond][key][idx]
                ax.plot(x_step, vals, color=_COLOR[cond],
                        linestyle=ls, lw=1.7,
                        label=f"{_DISPLAY[cond]} ({ldesc})"
                              if ci == 0 else None)
        ax.set_xlabel("rollout step k")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(x_step)
        if ci == 0:
            ax.legend(fontsize=7, loc="best", ncol=1, frameon=False)
        if key == "coll":
            ax.axhline(0.5, color="gray", lw=0.7, linestyle="--", alpha=0.6)

    fig.suptitle(
        "nuPlan Stage 1 imagination on lane_conflict — decoupled (no vis) "
        "stays bounded; object-only saturates",
        fontsize=13, y=1.0,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=str(_ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuplan-root",
                        default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=50000)
    parser.add_argument(
        "--nuplan-index-json",
        default=str(_ROOT / "experiments" / "nuplan_50k_balanced_paths_seed7.json"),
    )
    parser.add_argument(
        "--stage1-root",
        default=str(_ROOT / "experiments" / "nuplan_stage1_50k" / "seed7"),
        help="Per-condition stage-1 ckpt root (must contain "
             "<cond>/model.pt under it).",
    )
    parser.add_argument("--out-dir",
                        default=str(_ROOT / "experiments" / "figures" / "case_studies"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--conditions", nargs="+",
                        default=["wm_object", "wm_decoupled_no_vis"])
    args = parser.parse_args()

    out_pdf_dir = Path(args.out_dir) / "pdf"
    out_png_dir = Path(args.out_dir) / "png"
    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    config.seed = args.seed

    val_loader = _build_val_loader(args, config, args.seed)

    metrics = {}
    lc_mask = None
    for cond in args.conditions:
        ckpt = Path(args.stage1_root) / cond / "model.pt"
        if not ckpt.exists():
            print(f"[skip] {cond}: ckpt not found at {ckpt}")
            continue
        print(f"[load] {cond} <- {ckpt}", flush=True)
        model = _load_model(cond, ckpt, config, device)
        m = _rollout_metrics(model, val_loader, device, args.horizon)
        metrics[cond] = m
        if lc_mask is None:
            lc_mask = m["lane_conflict_mask"].astype(bool)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        n_lc = int(lc_mask.sum())
        sub_lc = m["lane_conflict_mask"].astype(bool)
        print(f"  N_total={m['coll'].shape[0]}  N_lc={int(sub_lc.sum())}  "
              f"return_lc={m['return_sum'][sub_lc].mean():.3f}  "
              f"coll_rate_lc={(m['coll_max'][sub_lc] > 0.5).mean():.3f}")

    if "wm_object" not in metrics or "wm_decoupled_no_vis" not in metrics:
        print("[fatal] need both wm_object and wm_decoupled_no_vis to plot Fig 4b")
        return

    picks = _pick_samples(metrics, lc_mask)
    print(f"[picked] {picks}")

    out_pdf = out_pdf_dir / "fig4b_imagination_nuplan_lane_conflict.pdf"
    out_png = out_png_dir / "fig4b_imagination_nuplan_lane_conflict.png"
    _plot(metrics, picks, args.horizon, args.conditions, lc_mask, out_pdf, out_png)
    print(f"  -> {out_pdf}")
    print(f"  -> {out_png}")

    # Persist numerics for audit + reproducibility.
    npz_path = Path(args.out_dir) / "fig4b_imagination_nuplan_metrics.npz"
    np.savez(
        npz_path,
        lane_conflict_mask=lc_mask.astype(np.uint8),
        **{
            f"{cond}__{k}": v
            for cond, m in metrics.items() for k, v in m.items()
        },
        picks_idx=np.array(list(picks.values())),
        picks_label=np.array(list(picks.keys())),
    )
    print(f"  -> {npz_path}")
    summary = {
        cond: {
            "n_samples": int(m["coll"].shape[0]),
            "n_lane_conflict": int(lc_mask.sum()),
            "return_sum_lc_mean": float(m["return_sum"][lc_mask].mean()),
            "imagined_collision_rate_lc": float((m["coll_max"][lc_mask] > 0.5).mean()),
            "ego_cos_step_lc_mean": float(m["ego_cos_step"][lc_mask].mean()),
            "action_norm_lc_max_mean": float(m["action_norm"][lc_mask].max(axis=1).mean()),
        }
        for cond, m in metrics.items()
    }
    summary_path = Path(args.out_dir) / "fig4b_imagination_nuplan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> {summary_path}")


if __name__ == "__main__":
    main()
