#!/usr/bin/env python
"""Paper-grade Figure 3: nuScenes Stage 1 imagination trajectory.

Loads the Stage 1 checkpoints under
``experiments/stage1_pilot_x/seed7/<cond>/`` for ``wm_object`` and
``wm_decoupled``, runs deterministic K-step imagination on the shared
nuScenes val split, and produces a 2-row figure:

  Row 1 (aggregate, mean ± std band over val samples)
    A. imagined collision probability per rollout step
    B. policy mean action norm  ||a_mean_t||_2
    C. ego-slot cosine step distance  1 - cos(e_t, e_{t+1})

  Row 2 (per-sample representative cases, picked from the val set)
    D. cumulative latent return over the rollout (3 picked samples × 2 models)
    E. step-by-step collision probability for the picked samples
    F. action norm for the picked samples

Sample picking strategy (deterministic, seed-driven):
  * "decoupled-collapse"   = sample with largest gap
                              max_t coll_decoupled - max_t coll_object
                              (i.e. decoupled goes high, object stays low)
  * "object-advantage"     = sample with largest gap
                              return_object - return_decoupled
  * "typical-stable"       = sample with smallest |return_object - return_decoupled|
                              that is non-degenerate (action norm > 0.5 for both)

Usage:
    PYTHONPATH=src python scripts/plot_paper_fig_imagination_nuscenes.py \
        --stage1-root experiments/stage1_pilot_x/seed7 \
        --num-scenes 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import DEFAULT_REWARD_CFG
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch
from doorrl.utils import set_seed


_CONDITIONS = {
    "wm_object":           "object_only",
    "wm_decoupled":        "object_relation_decoupled_visibility",
    "wm_decoupled_no_vis": "object_relation_decoupled",
}

_DISPLAY = {
    "wm_object":           "Object-only (Stage 1)",
    "wm_decoupled":        "Decoupled +visibility (Stage 1)",
    "wm_decoupled_no_vis": "Decoupled, no visibility",
}

_COLOR = {
    "wm_object":           "#1f77b4",
    "wm_decoupled":        "#d62728",
    "wm_decoupled_no_vis": "#9467bd",
}


def _build_val_loader(args, config: DoorRLConfig) -> DataLoader:
    full = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        num_scenes=args.num_scenes,
        version="v1.0-trainval",
        cache_dir=args.token_cache_dir or None,
    )
    names = sorted(set(full.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * n_total))) if n_total > 1 else 0
    n_train = n_total - n_val
    val_scenes = set(shuffled[n_train:])
    val_idx = full.indices_for_scenes(val_scenes)
    print(f"val: {n_val} scenes / {len(val_idx)} samples")
    return DataLoader(
        Subset(full, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=0,
        pin_memory=True,
    )


def _load_model(cond: str, ckpt_path: Path, config: DoorRLConfig,
                device: torch.device) -> torch.nn.Module:
    variant_name = _CONDITIONS[cond]
    model = DoorRLModelVariant(config.model, ModelVariant(variant_name))
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _rollout_metrics(model, val_loader, device, horizon: int):
    """Per-sample per-step metrics for one model.

    Returns
    -------
    dict with arrays of shape:
        coll        [N, K]    sigmoid(collision_t)
        action_norm [N, K]    ||a_mean_t||_2
        ego_cos_step[N, K]    1 - cos(e_t, e_{t+1})
        ret_cumsum  [N, K]    cumulative reward through step k
        rewards     [N, K]
        return_sum  [N]
        coll_max    [N]
    """
    coll_all, an_all, eg_all, ret_all, rew_all = [], [], [], [], []
    for batch in val_loader:
        batch = batch.to(device)
        traj = imagine_trajectory(
            model, batch,
            horizon=horizon, deterministic=True,
            reward_cfg=DEFAULT_REWARD_CFG,
            detach_world_model=True,
        )
        coll_all.append(traj.collisions.cpu())                 # [B, K]
        action_norm = traj.action_means.norm(dim=-1).cpu()     # [B, K]
        an_all.append(action_norm)
        # cosine step distance on ego_latents [B, K+1, D]
        a = traj.ego_latents[:, :-1, :]
        b = traj.ego_latents[:, 1:, :]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-3)
        eg_all.append((1.0 - cos).cpu())                       # [B, K]
        rew_all.append(traj.rewards.cpu())                     # [B, K]
        ret_all.append(traj.rewards.cumsum(dim=1).cpu())       # [B, K]

    coll = torch.cat(coll_all, dim=0).numpy()
    an = torch.cat(an_all, dim=0).numpy()
    eg = torch.cat(eg_all, dim=0).numpy()
    ret = torch.cat(ret_all, dim=0).numpy()
    rew = torch.cat(rew_all, dim=0).numpy()
    return {
        "coll": coll, "action_norm": an, "ego_cos_step": eg,
        "ret_cumsum": ret, "rewards": rew,
        "return_sum": rew.sum(axis=1),
        "coll_max": coll.max(axis=1),
    }


def _band(ax, x, vals, color, label, alpha=0.18):
    mu = vals.mean(axis=0)
    sd = vals.std(axis=0)
    ax.plot(x, mu, color=color, label=label, lw=2.0, marker="o", markersize=4)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=alpha, lw=0)


def _pick_samples(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, int]:
    obj = metrics["wm_object"]
    dec = metrics["wm_decoupled"]
    n = obj["coll_max"].shape[0]
    decoupled_collapse = int(np.argmax(dec["coll_max"] - obj["coll_max"]))
    object_advantage = int(np.argmax(obj["return_sum"] - dec["return_sum"]))
    # Typical: small gap, both with reasonable action norm.
    diff = np.abs(obj["return_sum"] - dec["return_sum"])
    legal = (
        (obj["action_norm"].mean(axis=1) > 0.5)
        & (dec["action_norm"].mean(axis=1) > 0.5)
    )
    if legal.any():
        diff_masked = np.where(legal, diff, np.inf)
        typical = int(np.argmin(diff_masked))
    else:
        typical = int(np.argmin(diff))
    return {
        "decoupled-collapse": decoupled_collapse,
        "object-advantage": object_advantage,
        "typical": typical,
    }


def _plot(metrics: Dict[str, Dict[str, np.ndarray]],
          picks: Dict[str, int],
          horizon: int, conditions: List[str],
          out_pdf: Path, out_png: Path):
    x_step = np.arange(1, horizon + 1)
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # Row 1: aggregate mean ± std bands.
    titles_top = [
        "(A) Imagined collision probability",
        "(B) Policy mean action norm  ||a_mean||₂",
        "(C) Ego-slot cosine step distance  1 − cos(eₜ, eₜ₊₁)",
    ]
    keys_top = ["coll", "action_norm", "ego_cos_step"]
    ylabel_top = ["P(collision)", "||a_mean||₂", "1 − cos"]
    for ci, key in enumerate(keys_top):
        ax = fig.add_subplot(gs[0, ci])
        for cond in conditions:
            _band(ax, x_step, metrics[cond][key],
                  color=_COLOR[cond], label=_DISPLAY[cond])
        ax.set_xlabel("rollout step k")
        ax.set_ylabel(ylabel_top[ci])
        ax.set_title(titles_top[ci], fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(x_step)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left")
            ax.axhline(0.5, color="gray", lw=0.7, linestyle="--", alpha=0.6)

    # Row 2: per-sample picks.
    titles_bot = [
        "(D) Cumulative latent return  (representative samples)",
        "(E) Per-step collision probability",
        "(F) Action norm",
    ]
    pick_styles = {
        "decoupled-collapse": ("-", "decoupled collapses"),
        "object-advantage":   ("--", "object-only wins"),
        "typical":            (":", "typical"),
    }
    for ci, (key, ylabel, title) in enumerate(zip(
        ["ret_cumsum", "coll", "action_norm"],
        ["cum. reward", "P(collision)", "||a_mean||₂"],
        titles_bot,
    )):
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
        "nuScenes Stage 1 imagination — object-only stays bounded; "
        "decoupled+visibility drifts and saturates",
        fontsize=13, y=1.0,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=str(_ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuscenes-root",
                        default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument("--token-cache-dir",
                        default=str(_ROOT / "experiments" / "_token_cache"))
    parser.add_argument("--stage1-root",
                        default=str(_ROOT / "experiments" / "stage1_pilot_x" / "seed7"))
    parser.add_argument("--out-dir",
                        default=str(_ROOT / "experiments" / "figures" / "case_studies"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-scenes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--conditions", nargs="+",
                        default=["wm_object", "wm_decoupled"],
                        help="Subset of {wm_object, wm_decoupled, wm_decoupled_no_vis}.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap on val batches for speed during iteration.")
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

    val_loader = _build_val_loader(args, config)
    if args.max_batches is not None:
        # Quick-iteration: clip the loader to max_batches.
        from itertools import islice

        class _Clipped:
            def __init__(self, base, k): self.base, self.k = base, k
            def __iter__(self): return iter(islice(self.base, self.k))

        val_loader = _Clipped(val_loader, args.max_batches)

    metrics = {}
    for cond in args.conditions:
        ckpt = Path(args.stage1_root) / cond / "model.pt"
        if not ckpt.exists():
            print(f"[skip] {cond}: ckpt not found at {ckpt}")
            continue
        print(f"[load] {cond} <- {ckpt}")
        model = _load_model(cond, ckpt, config, device)
        metrics[cond] = _rollout_metrics(
            model, val_loader, device, horizon=args.horizon,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        m = metrics[cond]
        print(f"  N={m['coll'].shape[0]}  return_mean={m['return_sum'].mean():.3f}  "
              f"coll_rate={(m['coll_max'] > 0.5).mean():.3f}  "
              f"|a|max_mean={m['action_norm'].max(axis=1).mean():.2f}")

    if "wm_object" not in metrics or "wm_decoupled" not in metrics:
        print("[fatal] need both wm_object and wm_decoupled to plot Fig 3")
        return

    picks = _pick_samples(metrics)
    print(f"[picked] {picks}")

    out_pdf = out_pdf_dir / "fig3_imagination_nuscenes.pdf"
    out_png = out_png_dir / "fig3_imagination_nuscenes.png"
    _plot(metrics, picks, args.horizon, args.conditions, out_pdf, out_png)
    print(f"  -> {out_pdf}")
    print(f"  -> {out_png}")

    # Persist the underlying numerics so reviewers can audit.
    npz_path = Path(args.out_dir) / "fig3_imagination_nuscenes_metrics.npz"
    np.savez(
        npz_path,
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
            "return_sum_mean": float(m["return_sum"].mean()),
            "return_sum_std": float(m["return_sum"].std()),
            "imagined_collision_rate": float((m["coll_max"] > 0.5).mean()),
            "ego_cos_step_mean": float(m["ego_cos_step"].mean()),
            "action_norm_mean_max": float(m["action_norm"].max(axis=1).mean()),
        }
        for cond, m in metrics.items()
    }
    summary_path = Path(args.out_dir) / "fig3_imagination_nuscenes_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> {summary_path}")


if __name__ == "__main__":
    main()
