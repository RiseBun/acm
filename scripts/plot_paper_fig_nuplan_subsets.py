#!/usr/bin/env python
"""Paper-grade Figure 4a: nuPlan Stage 1 ranking reversal on
interaction-heavy subsets.

Loads ``experiments/nuplan_interaction_subset_50k/summary.json`` (3-seed
mean/std) and renders a 3-panel bar chart comparing ``wm_object`` and
``wm_decoupled_no_vis`` across the canonical val subsets.

Story:
  * On ``all_val`` decoupled_no_vis is already 3× better in collision
    rate, 14× higher in latent return.
  * The gap widens on ``lane_conflict`` (~67% of val) and especially on
    ``low_ttc_proxy`` (~47% of val), where object-only collapses.
  * Subsets are ordered by "expected interaction difficulty" so that the
    figure reads left-to-right as "easier -> harder".

This is the inverse of the nuScenes Stage 1 finding (Fig 3): on nuPlan,
the typed-budget decoupled abstraction without visibility weighting wins
across every subset, with the largest gap on the precise interaction
subsets the figure picks out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# Canonical ordering: "easier on the left, harder on the right".
# Skip ``high_interaction_union`` (just the union of others) and
# ``rare_agent_dense`` (numerically close to ``dense_agents``).
_SUBSETS = [
    ("all_val",          "all val"),
    ("dense_agents",     "dense agents"),
    ("lane_conflict",    "lane conflict"),
    ("low_ttc_proxy",    "low TTC proxy"),
]

_METRICS = [
    ("teacher_action_mse",      "(A) Teacher-action MSE  (↓)",  "MSE"),
    ("latent_return_mean",      "(B) Latent return  (↑)",        "return"),
    ("imagined_collision_rate", "(C) Imagined collision rate  (↓)", "P(coll)"),
]

_CONDITIONS = ["wm_object", "wm_decoupled_no_vis"]
_DISPLAY = {
    "wm_object":           "Object-only",
    "wm_decoupled_no_vis": "Decoupled (no visibility)",
}
_COLOR = {
    "wm_object":           "#1f77b4",
    "wm_decoupled_no_vis": "#d62728",
}

# Subsets to highlight as "hard interaction" with a soft shaded band.
_HIGHLIGHT = {"lane_conflict", "low_ttc_proxy"}


def _build(summary: Dict, conditions: List[str], subsets):
    """Pull (mean, std) for each (cond, subset, metric)."""
    out = {}
    for met, _, _ in _METRICS:
        out[met] = {}
        for cond in conditions:
            mu, sd = [], []
            for sname, _ in subsets:
                m = summary[cond][sname]
                mu.append(float(m[f"{met}_mean"]))
                sd.append(float(m[f"{met}_std"]))
            out[met][cond] = (np.array(mu), np.array(sd))
    return out


def _plot(data, conditions, subsets, sample_counts,
          out_pdf: Path, out_png: Path):
    n_sub = len(subsets)
    x = np.arange(n_sub)
    bw = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    for ax_i, (met, title, ylabel) in enumerate(_METRICS):
        ax = axes[ax_i]
        # Highlight band on hard subsets.
        for i, (sname, _) in enumerate(subsets):
            if sname in _HIGHLIGHT:
                ax.axvspan(i - 0.5, i + 0.5, color="#fff3e0",
                           alpha=0.5, zorder=0)

        for ci, cond in enumerate(conditions):
            mu, sd = data[met][cond]
            offset = (ci - (len(conditions) - 1) / 2.0) * bw
            ax.bar(
                x + offset, mu, width=bw, yerr=sd, capsize=3.5,
                color=_COLOR[cond], edgecolor="black", linewidth=0.6,
                label=_DISPLAY[cond] if ax_i == 0 else None,
                error_kw=dict(ecolor="#222222", lw=0.9),
            )
            # Numeric labels above each bar.
            for xi, (m_v, s_v) in enumerate(zip(mu, sd)):
                ax.text(
                    xi + offset, m_v + s_v + (mu.max() * 0.02),
                    f"{m_v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#222222",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [s[1] + f"\n(n≈{int(sample_counts[s[0]])})" for s in subsets],
            fontsize=8.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", which="major", pad=2)
        if met == "imagined_collision_rate":
            ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6)
            ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
        if ax_i == 0:
            ax.legend(fontsize=9.5, loc="upper left", frameon=False)

    # Add a footnote about the highlight.
    fig.text(
        0.5, -0.02,
        "shaded subsets = interaction-heavy "
        "(lane-conflict relations OR closest relation TTC ≤ 5 s)",
        ha="center", va="bottom", fontsize=9, color="#555555",
    )
    fig.suptitle(
        "nuPlan Stage 1 (50 k samples, 3 seeds) — "
        "decoupled-no-vis dominates on interaction-heavy subsets",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="experiments/nuplan_interaction_subset_50k/summary.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures/case_studies",
    )
    args = parser.parse_args()

    out_pdf_dir = Path(args.out_dir) / "pdf"
    out_png_dir = Path(args.out_dir) / "png"
    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary) as f:
        payload = json.load(f)

    data = _build(payload["summary"], _CONDITIONS, _SUBSETS)
    sample_counts = {
        sname: payload["summary"][_CONDITIONS[0]][sname]["n_samples_mean"]
        for sname, _ in _SUBSETS
    }

    out_pdf = out_pdf_dir / "fig4a_nuplan_subset_bars.pdf"
    out_png = out_png_dir / "fig4a_nuplan_subset_bars.png"
    _plot(data, _CONDITIONS, _SUBSETS, sample_counts, out_pdf, out_png)
    print(f"[plot_paper_fig_nuplan_subsets] -> {out_pdf}")
    print(f"[plot_paper_fig_nuplan_subsets] -> {out_png}")

    # Also print the table the figure encodes, for the paper text.
    print("\nTable (mean ± std across seeds 7, 42, 123):")
    print(f"{'subset':<22}{'metric':<24}{'wm_object':>16}{'wm_decoupled_no_vis':>22}{'Δ (dec - obj)':>16}")
    print("-" * 100)
    for sname, sdisp in _SUBSETS:
        for met, _, _ in _METRICS:
            o_mu, o_sd = payload["summary"]["wm_object"][sname][f"{met}_mean"], \
                         payload["summary"]["wm_object"][sname][f"{met}_std"]
            d_mu, d_sd = payload["summary"]["wm_decoupled_no_vis"][sname][f"{met}_mean"], \
                         payload["summary"]["wm_decoupled_no_vis"][sname][f"{met}_std"]
            delta = d_mu - o_mu
            print(
                f"{sdisp:<22}{met:<24}"
                f"{f'{o_mu:.2f} ± {o_sd:.2f}':>16}"
                f"{f'{d_mu:.2f} ± {d_sd:.2f}':>22}"
                f"{f'{delta:+.2f}':>16}"
            )


if __name__ == "__main__":
    main()
