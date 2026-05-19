#!/usr/bin/env python
"""Paper-grade Figure 1 / Figure 2: naive vs decoupled slot selection.

Builds on the `slot_selections_seed7.pkl` produced by
`extract_slot_selections.py`. Compared with `plot_slot_scenes.py` (which
produces an exploration-grade 4-panel grid for many cases), this script
renders a focused 2-panel comparison with:

  * Same scene shown side-by-side under naive shared top-k
    (`object_relation`) and our decoupled typed-budget abstraction
    (`object_relation_decoupled`).
  * **Red dashed circles** around dynamic agents that decoupled
    selected but naive did not — the visual claim of the figure.
  * **Green dashed circles** around dynamic agents naive selected but
    decoupled did not (usually rare; shown for honesty).
  * GT next-frame motion arrows on every dynamic agent.
  * A clear caption with the per-panel selection statistics.

Outputs both PDF (paper) and PNG (preview) at the requested DPI.

Selection criteria (ranked):

    score = (#near dyn agents decoupled selected) − (#near dyn agents naive selected)

We require a strict information advantage (`score >= 2`) and an
honesty constraint (decoupled's near-field next-token prediction error
is at least 1.2x lower than naive's), then pick the top-K.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


_TYPE_NAMES = {0: "EGO", 1: "VEH", 2: "PED", 3: "CYC", 4: "MAP",
               5: "SIG", 6: "REL", 7: "PAD"}
_TYPE_COLOR = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e",
               4: "#9467bd", 5: "#8c564b", 6: "#e377c2"}
_DYN_TYPES = (0, 1, 2, 3)
_NEAR_R_DEFAULT = 18.0


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _dyn_indices(sample: dict, near_r: float) -> set[int]:
    types = sample["token_types"]
    mask = sample["token_mask"]
    xy = sample["tokens"][:, :2]
    dists = xy.norm(dim=-1)
    dyn_mask = torch.zeros_like(mask)
    for t in _DYN_TYPES:
        dyn_mask |= (types == t)
    near = dyn_mask & mask & (dists <= near_r) & (dists > 0.01)
    s = set(torch.nonzero(near, as_tuple=False).flatten().tolist())
    s.discard(0)
    return s


def _sel_set(sample: dict) -> set[int]:
    if sample.get("is_set_prediction", False):
        return set()
    return {
        int(i) for i, m in zip(
            sample["selected_indices"].tolist(),
            sample["selected_mask"].bool().tolist(),
        ) if m
    }


def _nearest_pred_err(sample: dict, near_r: float) -> float:
    tokens = sample["tokens"]
    types = sample["token_types"]
    mask = sample["token_mask"]
    next_tokens = sample["next_tokens"]
    pred_next = sample["predicted_next_tokens"]
    sel_mask = sample["selected_mask"]
    xy_now = tokens[:, :2]
    dyn_mask = torch.zeros_like(mask)
    for t in _DYN_TYPES:
        dyn_mask |= (types == t)
    gt_sel = dyn_mask & mask & (xy_now.norm(dim=-1) <= near_r)
    gt_sel[0] = False
    if gt_sel.sum() == 0:
        return float("nan")
    gt_next_xy = next_tokens[gt_sel][:, :2]
    pred_xy = pred_next[:, :2]
    valid = sel_mask.bool()
    if valid.sum() == 0:
        return float("nan")
    pred_xy = pred_xy[valid]
    diff = gt_next_xy.unsqueeze(1) - pred_xy.unsqueeze(0)
    dist = diff.norm(dim=-1)
    return float(dist.min(dim=-1).values.mean().item())


def _pick_cases(
    variants_data: Dict[str, List[dict]],
    near_r: float,
    top_k: int,
    min_score: int,
    err_advantage_ratio: float,
    min_agents: int,
) -> List[Tuple[int, int, dict]]:
    naive = variants_data["object_relation"]
    dec = variants_data["object_relation_decoupled"]
    assert len(naive) == len(dec)

    ranked = []
    for i, (n_s, d_s) in enumerate(zip(naive, dec)):
        near_set = _dyn_indices(n_s, near_r)
        if len(near_set) < min_agents:
            continue
        n_sel_set = _sel_set(n_s)
        d_sel_set = _sel_set(d_s)
        n_near_sel = near_set & n_sel_set
        d_near_sel = near_set & d_sel_set
        score = len(d_near_sel) - len(n_near_sel)
        if score < min_score:
            continue
        n_err = _nearest_pred_err(n_s, near_r)
        d_err = _nearest_pred_err(d_s, near_r)
        # Require honest err advantage too (we don't want a case where
        # naive happened to pick a stable agent and decoupled overfit).
        if (
            np.isnan(n_err) or np.isnan(d_err)
            or n_err < d_err * err_advantage_ratio
        ):
            continue
        info = {
            "n_near": len(near_set),
            "naive_sel": len(n_near_sel),
            "dec_sel": len(d_near_sel),
            "naive_err": n_err,
            "dec_err": d_err,
            "missed_by_naive": d_near_sel - n_near_sel,
            "missed_by_dec": n_near_sel - d_near_sel,
        }
        ranked.append((i, score, info))

    ranked.sort(key=lambda x: (-x[1], x[2]["naive_err"] - x[2]["dec_err"] * -1))
    return ranked[:top_k]


def _draw_panel(ax, sample: dict, near_r: float, title: str,
                missed_by_naive: set[int], missed_by_dec: set[int],
                show_naive_misses: bool):
    tokens = _to_np(sample["tokens"])
    types = _to_np(sample["token_types"])
    mask = _to_np(sample["token_mask"].bool())
    next_tokens = _to_np(sample["next_tokens"])
    sel_mask = _to_np(sample["selected_mask"].bool())
    sel_idx_list = _to_np(sample["selected_indices"]).tolist()
    valid_sel_idx = {
        int(i) for i, m in zip(sel_idx_list, sel_mask.tolist()) if m
    }

    # Near-field circle.
    ax.add_patch(plt.Circle((0, 0), near_r, color="black", fill=False,
                            linestyle=":", linewidth=0.6, alpha=0.4))
    ax.text(near_r * 0.72, near_r * 0.72, f"{int(near_r)} m",
            fontsize=7, color="gray", alpha=0.6)

    for i in range(len(mask)):
        if not mask[i]:
            continue
        t = int(types[i])
        if t == 7 or t == 4 or t == 5 or t == 6:
            continue  # only dyn agents on the BEV (cleaner figure)
        x, y = tokens[i, 0], tokens[i, 1]
        is_sel = (i in valid_sel_idx)
        marker = "*" if t == 0 else "o"
        size = 260 if t == 0 else (95 if is_sel else 30)
        edgecolor = "#111111" if is_sel else _TYPE_COLOR.get(t, "#444")
        if t == 0:
            edgecolor = "#8b0000"
        lw = 1.8 if is_sel else 0.6
        face = _TYPE_COLOR.get(t, "#888")
        alpha = 1.0 if is_sel or t == 0 else 0.7
        ax.scatter(x, y, s=size, c=face, edgecolors=edgecolor,
                   linewidths=lw, marker=marker, alpha=alpha, zorder=3)
        if t in _DYN_TYPES and t != 0:
            nx, ny = next_tokens[i, 0], next_tokens[i, 1]
            ax.annotate(
                "", xy=(nx, ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                color=_TYPE_COLOR.get(t, "#444"),
                                alpha=0.55),
                zorder=2,
            )

    # Highlight the missed-by-naive agents in this panel:
    #   * On the naive panel, draw red dashed callouts at agents the
    #     naive variant DROPPED but decoupled kept.
    #   * On the decoupled panel, draw green dashed callouts at the same
    #     agents to emphasise that decoupled DID keep them.
    miss_color = "#d62728" if show_naive_misses else "#2ca02c"
    miss_label = "missed by naive" if show_naive_misses else "kept by decoupled"
    for idx in missed_by_naive:
        if not mask[idx]:
            continue
        x, y = tokens[idx, 0], tokens[idx, 1]
        ax.add_patch(plt.Circle(
            (x, y), 1.6, fill=False, edgecolor=miss_color,
            linestyle="--", linewidth=2.0, zorder=5,
        ))

    # And the (rare) reverse: agents naive kept but decoupled dropped.
    for idx in missed_by_dec:
        if not mask[idx]:
            continue
        x, y = tokens[idx, 0], tokens[idx, 1]
        ax.add_patch(plt.Circle(
            (x, y), 1.6, fill=False, edgecolor="#7f7f7f",
            linestyle=":", linewidth=1.4, zorder=5,
        ))

    ax.set_xlim(-near_r * 1.6, near_r * 1.6)
    ax.set_ylim(-near_r * 1.6, near_r * 1.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="#cccccc", lw=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.5, zorder=0)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [m] (ego frame)", fontsize=8.5)
    ax.set_ylabel("y [m] (ego frame)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, pad=6)


def _build_legend(fig, show_misses: bool):
    handles = []
    for t in _DYN_TYPES:
        marker = "*" if t == 0 else "o"
        handles.append(plt.Line2D(
            [0], [0], marker=marker, color="w",
            markerfacecolor=_TYPE_COLOR[t],
            markeredgecolor="#8b0000" if t == 0 else _TYPE_COLOR[t],
            markersize=10 if t == 0 else 7,
            label=_TYPE_NAMES[t],
            linestyle="",
        ))
    handles.append(plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="lightgray", markeredgecolor="black",
        markersize=10, markeredgewidth=1.6,
        label="selected by abstraction",
        linestyle="",
    ))
    if show_misses:
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="white", markeredgecolor="#d62728",
            markersize=12, markeredgewidth=2.0,
            label="dropped by naive (kept by decoupled)",
            linestyle="--",
        ))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04),
               ncol=len(handles), fontsize=9, frameon=False)


def _plot_case(naive_s: dict, dec_s: dict, info: dict, near_r: float,
               idx: int, out_pdf: Path, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2))
    title_naive = (
        "Object+Relation-16  (naive, shared top-k)\n"
        f"$\\bf{{{info['naive_sel']}}}$ / {info['n_near']} near agents selected"
        f"   ·   pred err = {info['naive_err']:.2f} m"
    )
    title_dec = (
        "Decoupled typed-budget  (Ours, 12 dyn + 4 rel)\n"
        f"$\\bf{{{info['dec_sel']}}}$ / {info['n_near']} near agents selected"
        f"   ·   pred err = {info['dec_err']:.2f} m"
    )
    _draw_panel(axes[0], naive_s, near_r, title_naive,
                missed_by_naive=info["missed_by_naive"],
                missed_by_dec=info["missed_by_dec"],
                show_naive_misses=True)
    _draw_panel(axes[1], dec_s, near_r, title_dec,
                missed_by_naive=info["missed_by_naive"],
                missed_by_dec=info["missed_by_dec"],
                show_naive_misses=False)

    err_ratio = (info["naive_err"] / max(info["dec_err"], 1e-6)) if info["dec_err"] > 0 else float("inf")
    fig.suptitle(
        "Naive shared top-k drops dynamic agents that typed-budget keeps  "
        f"(prediction-err ratio  {err_ratio:.0f}×)",
        fontsize=12.5, y=0.995,
    )
    _build_legend(fig, show_misses=True)
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="experiments/figures/slot_selections_seed7.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures/case_studies",
    )
    parser.add_argument("--near-r", type=float, default=_NEAR_R_DEFAULT)
    parser.add_argument("--top-k", type=int, default=4,
                        help="Generate the top-K differentiating cases.")
    parser.add_argument("--min-score", type=int, default=2,
                        help="Minimum (dec_sel - naive_sel) to include a case.")
    parser.add_argument("--err-advantage-ratio", type=float, default=1.2,
                        help="Require naive_err >= ratio * dec_err.")
    parser.add_argument("--min-agents", type=int, default=4)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    out_pdf_dir = Path(args.out_dir) / "pdf"
    out_png_dir = Path(args.out_dir) / "png"
    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    variants_data = payload["variants"]
    naive = variants_data["object_relation"]
    dec = variants_data["object_relation_decoupled"]

    if args.sample_indices:
        cases = []
        for i in args.sample_indices:
            near_set = _dyn_indices(naive[i], args.near_r)
            n_sel_set = _sel_set(naive[i])
            d_sel_set = _sel_set(dec[i])
            n_near_sel = near_set & n_sel_set
            d_near_sel = near_set & d_sel_set
            cases.append((i, len(d_near_sel) - len(n_near_sel), {
                "n_near": len(near_set),
                "naive_sel": len(n_near_sel),
                "dec_sel": len(d_near_sel),
                "naive_err": _nearest_pred_err(naive[i], args.near_r),
                "dec_err": _nearest_pred_err(dec[i], args.near_r),
                "missed_by_naive": d_near_sel - n_near_sel,
                "missed_by_dec": n_near_sel - d_near_sel,
            }))
    else:
        cases = _pick_cases(
            variants_data, near_r=args.near_r,
            top_k=args.top_k,
            min_score=args.min_score,
            err_advantage_ratio=args.err_advantage_ratio,
            min_agents=args.min_agents,
        )

    if not cases:
        print("[plot_paper_fig_slot_compare] No cases passed the selection "
              "criteria. Try lowering --min-score or --err-advantage-ratio.")
        return

    print(f"[plot_paper_fig_slot_compare] selected {len(cases)} cases:")
    print(f"{'idx':<6}{'n_near':<8}{'naive_sel':<11}{'dec_sel':<9}"
          f"{'naive_err':<11}{'dec_err':<10}{'missed_by_naive_idxs'}")
    print("-" * 80)
    for idx, score, info in cases:
        print(
            f"{idx:<6}{info['n_near']:<8}{info['naive_sel']:<11}"
            f"{info['dec_sel']:<9}{info['naive_err']:<11.2f}"
            f"{info['dec_err']:<10.2f}{sorted(info['missed_by_naive'])}"
        )

    for rank, (idx, score, info) in enumerate(cases):
        out_pdf = out_pdf_dir / f"fig1_slot_compare_case{rank:02d}_idx{idx}.pdf"
        out_png = out_png_dir / f"fig1_slot_compare_case{rank:02d}_idx{idx}.png"
        _plot_case(naive[idx], dec[idx], info, args.near_r, idx, out_pdf, out_png)
        print(f"  -> {out_pdf}")
        print(f"  -> {out_png}")


if __name__ == "__main__":
    main()
