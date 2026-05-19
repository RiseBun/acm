#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper_assets" / "data"
FIG_DIR = ROOT / "figures" / "paper"
PKL_PATH = ROOT / "figures" / "slot_selections_seed7.pkl"


PALETTE = {
    "object_only": "#6B7280",
    "object_relation": "#D55E5E",
    "object_relation_visibility": "#F2A65A",
    "object_relation_decoupled": "#1F77B4",
    "object_relation_decoupled_visibility": "#2CA58D",
    "holistic_16slot": "#A8B0BB",
    "holistic": "#374151",
    "wm_object": "#6B7280",
    "wm_decoupled": "#2CA58D",
    "wm_decoupled_no_vis": "#1F77B4",
    "nuscenes": "#7A7A7A",
    "nuplan": "#1F77B4",
}


LABELS = {
    "holistic_16slot": "Holistic-16",
    "object_only": "Object-Only",
    "object_relation": "Naive Obj+Rel",
    "object_relation_visibility": "Naive+Vis",
    "object_relation_decoupled": "Decoupled",
    "object_relation_decoupled_visibility": "Decoupled+Vis",
    "holistic": "Holistic-full",
    "wm_object": "WM-Object",
    "wm_decoupled": "WM-Decoupled",
    "wm_decoupled_no_vis": "WM-Decoupled-NoVis",
}


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "grid.linestyle": "--",
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(name: str) -> dict:
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def _beautify_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_color("#9AA4B2")
    ax.spines["bottom"].set_color("#9AA4B2")


def _add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.12,
        1.05,
        tag,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _plot_stage0_slot_budget() -> None:
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    variants = payload["variants"]
    type_names = {
        0: "EGO",
        1: "VEH",
        2: "PED",
        3: "CYC",
        4: "MAP",
        5: "SIG",
        6: "REL",
    }
    type_order = [0, 1, 2, 3, 4, 6]
    type_colors = {
        0: "#D84A4A",
        1: "#4A90C2",
        2: "#4CAF50",
        3: "#F39C34",
        4: "#9B79C6",
        6: "#D472B6",
    }
    variant_order = [
        "holistic_16slot",
        "object_only",
        "object_relation",
        "object_relation_visibility",
        "object_relation_decoupled",
        "object_relation_decoupled_visibility",
        "holistic",
    ]

    def count_slot_types(samples: list[dict]) -> tuple[dict, int, bool]:
        if not samples:
            return {}, 0, False
        is_set_pred = bool(samples[0].get("is_set_prediction", False))
        if is_set_pred:
            return {"learned": float(samples[0]["selected_indices"].numel())}, int(
                samples[0]["selected_indices"].numel()
            ), True

        totals: dict[int, float] = {}
        n = 0
        k = 0
        for sample in samples:
            sel = sample["selected_indices"]
            sel_mask = sample["selected_mask"].bool()
            token_types = sample["token_types"]
            selected_types = token_types.gather(0, sel.long())
            for t, m in zip(selected_types.tolist(), sel_mask.tolist()):
                if not m:
                    continue
                totals[int(t)] = totals.get(int(t), 0.0) + 1.0
            n += 1
            k = max(k, int(sel.numel()))
        return {t: v / max(n, 1) for t, v in totals.items()}, k, False

    stats = {vid: count_slot_types(variants[vid]) for vid in variant_order}
    labels = [
        "Holistic-16\n(learned)",
        "Object-Only",
        "Naive Obj+Rel",
        "Naive+Vis",
        "Decoupled",
        "Decoupled+Vis",
        "Holistic-full",
    ]
    x = np.arange(len(variant_order))

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    learned_heights = []
    for vid in variant_order:
        counts, k, is_set = stats[vid]
        learned_heights.append(k if is_set else 0.0)
    ax.bar(
        x,
        learned_heights,
        color="#B8B8B8",
        edgecolor="#444444",
        linewidth=0.8,
        hatch="xx",
        label="learned / mixed",
        zorder=2,
    )

    bottoms = np.zeros(len(variant_order))
    for token_type in type_order:
        heights = []
        for vid in variant_order:
            counts, _, is_set = stats[vid]
            heights.append(0.0 if is_set else counts.get(token_type, 0.0))
        heights = np.array(heights)
        bars = ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=type_colors[token_type],
            edgecolor="white",
            linewidth=0.7,
            label=type_names[token_type],
            zorder=3,
        )
        for bar, h, b in zip(bars, heights, bottoms):
            if h >= 0.55:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    b + h / 2.0,
                    f"{h:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if token_type in {0, 1, 4} else "black",
                )
        bottoms += heights

    ax.axhline(16, color="#444444", linewidth=1.0, linestyle=":")
    ax.text(
        len(variant_order) - 0.2,
        16.2,
        "fair 16-slot budget",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.annotate(
        "naive mixing over-allocates REL\nand starves dynamic agents",
        xy=(2, 15.2),
        xytext=(1.55, 22.0),
        arrowprops=dict(arrowstyle="->", color=PALETTE["object_relation"], lw=1.2),
        fontsize=9,
        color=PALETTE["object_relation"],
        ha="center",
    )
    ax.annotate(
        "typed budgets preserve dynamic context\nwhile keeping relation slots",
        xy=(4, 13.2),
        xytext=(4.9, 24.2),
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["object_relation_decoupled"],
            lw=1.2,
        ),
        fontsize=9,
        color=PALETTE["object_relation_decoupled"],
        ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average selected slots per sample")
    ax.set_ylim(0, 30)
    _beautify_axis(ax)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    _save(fig, "paper_stage0_slot_budget")


def _plot_stage0_metrics() -> None:
    agg = _load_json("experiments__table3_fair_fix2_aggregate.json")["metrics"]
    metrics = [
        ("dyn_rollout_mse", "Dyn Rollout MSE", False),
        ("rare_ade", "Rare ADE (m)", False),
        ("interaction_recall_at_1m", "Interaction Recall @ 1 m", True),
    ]
    variants = [
        "object_only",
        "object_relation",
        "object_relation_decoupled",
        "object_relation_decoupled_visibility",
    ]
    labels = ["Object-Only", "Naive Obj+Rel", "Decoupled", "Decoupled+Vis"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    for idx, (metric_key, title, higher_better) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(variants))
        means = [agg[v][metric_key]["mean"] for v in variants]
        stds = [agg[v][metric_key]["std"] for v in variants]
        seed_values = [agg[v][metric_key]["values"] for v in variants]
        colors = [PALETTE[v] for v in variants]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="white",
            linewidth=0.9,
            capsize=4,
            zorder=3,
        )
        rng = np.random.default_rng(17 + idx)
        for xpos, vals in zip(x, seed_values):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=18,
                color="#1F2937",
                alpha=0.85,
                zorder=4,
            )
        for bar, m in zip(bars, means):
            offset = (max(means) - min(0.0, min(means))) * 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel("Higher is better" if higher_better else "Lower is better")
        _beautify_axis(ax)
        if metric_key == "interaction_recall_at_1m":
            ax.set_ylim(0.35, 1.05)
        if metric_key == "rare_ade":
            ax.annotate(
                "-55.2% vs Object-Only",
                xy=(2, means[2]),
                xytext=(2.8, max(means) * 0.93),
                arrowprops=dict(
                    arrowstyle="->",
                    color=PALETTE["object_relation_decoupled"],
                    lw=1.1,
                ),
                fontsize=8.5,
                color=PALETTE["object_relation_decoupled"],
                ha="left",
            )
        if metric_key == "interaction_recall_at_1m":
            ax.annotate(
                "+8.3 pts",
                xy=(2, means[2]),
                xytext=(2.45, 0.92),
                arrowprops=dict(
                    arrowstyle="->",
                    color=PALETTE["object_relation_decoupled"],
                    lw=1.1,
                ),
                fontsize=8.5,
                color=PALETTE["object_relation_decoupled"],
                ha="left",
            )
        _add_panel_tag(ax, f"({chr(ord('a') + idx)})")
    fig.suptitle(
        "Stage 0 fair-budget representation metrics (mean ± sample std across 3 seeds; dots show per-seed values)",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "paper_stage0_metrics")


def _extract_stage1_point(summary: dict, condition: str, metric: str) -> tuple[float, float, list[float]]:
    if "conditions" in summary:
        cond = summary["conditions"][condition]
        mean = float(cond["mean"][metric])
        std = float(cond["std_across_seeds"][metric])
        vals = [float(seed[metric]) for seed in cond["seeds"]]
        return mean, std, vals
    cond = summary[condition][metric]
    return float(cond["mean"]), float(cond["std"]), [float(v) for v in cond["values"]]


def _plot_stage1_cross_dataset() -> None:
    nusc = _load_json("stage1_pilot_x__X_summary.json")
    nup20 = _load_json("nuplan_stage1_20k__summary.json")
    nup50 = _load_json("nuplan_stage1_50k__summary.json")

    datasets = ["nuScenes", "nuPlan 20k", "nuPlan 50k"]
    xs = np.arange(len(datasets))
    methods = [
        ("wm_object", "WM-Object"),
        ("wm_decoupled", "WM-Decoupled"),
        ("wm_decoupled_no_vis", "WM-Decoupled-NoVis"),
    ]
    source = [nusc, nup20, nup50]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharex=True)
    metric_specs = [
        ("latent_return_mean", "Imagined return", True),
        ("imagined_collision_rate", "Imagined collision rate", False),
    ]
    for panel_idx, (metric_key, ylabel, higher_better) in enumerate(metric_specs):
        ax = axes[panel_idx]
        for method_key, method_label in methods:
            color = PALETTE[method_key]
            means, stds = [], []
            valid_x = []
            for i, summary in enumerate(source):
                if "conditions" in summary and method_key not in summary["conditions"]:
                    continue
                if "conditions" not in summary and method_key not in summary:
                    continue
                mean, std, _ = _extract_stage1_point(summary, method_key, metric_key)
                valid_x.append(i)
                means.append(mean)
                stds.append(std)
            if not valid_x:
                continue
            ax.errorbar(
                valid_x,
                means,
                yerr=stds,
                color=color,
                marker="o",
                markersize=5.5,
                linewidth=2.0,
                capsize=4,
                label=method_label,
                zorder=3,
            )
            for xpos, mean in zip(valid_x, means):
                ax.scatter(
                    [xpos],
                    [mean],
                    s=36,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        _beautify_axis(ax)
        if metric_key == "latent_return_mean":
            ax.axvspan(-0.45, 0.45, color="#F3F4F6", alpha=0.7, zorder=0)
            ax.axvspan(0.55, 2.45, color="#EFF6FF", alpha=0.7, zorder=0)
            ax.annotate(
                "object-only remains\nmost stable on nuScenes",
                xy=(0, 31.79),
                xytext=(0.25, 38.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["wm_object"], lw=1.0),
                fontsize=8.5,
                color=PALETTE["wm_object"],
            )
            ax.annotate(
                "ranking flips on nuPlan",
                xy=(1.8, 16.0),
                xytext=(1.15, 31.0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=PALETTE["wm_decoupled_no_vis"],
                    lw=1.0,
                ),
                fontsize=8.5,
                color=PALETTE["wm_decoupled_no_vis"],
            )
        _add_panel_tag(ax, f"({chr(ord('a') + panel_idx)})")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle(
        "Stage 1 cross-dataset imagination-policy ranking reversal",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "paper_stage1_cross_dataset")


def _plot_planner_subset() -> None:
    planner = _load_json("nuplan_planner_sanity_50k__summary.json")["summary"]
    subsets = _load_json("nuplan_interaction_subset_50k__summary.json")["summary"]

    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.28)

    # Left: planner-like sanity as paired horizontal dumbbells.
    ax0 = fig.add_subplot(gs[0, 0])
    metrics = [
        ("teacher_action_mse", "Teacher action MSE", "down"),
        ("latent_return_mean", "Imagined return", "up"),
        ("imagined_collision_rate", "Imagined collision", "down"),
    ]
    ypos = np.arange(len(metrics))[::-1]
    for y, (metric, label, direction) in zip(ypos, metrics):
        obj_mean = planner["wm_object"]["mean"][metric]
        dec_mean = planner["wm_decoupled_no_vis"]["mean"][metric]
        ax0.plot([obj_mean, dec_mean], [y, y], color="#C7CDD6", lw=2.0, zorder=1)
        ax0.scatter(
            obj_mean, y, s=64, color=PALETTE["wm_object"], edgecolor="white", linewidth=0.8, zorder=3
        )
        ax0.scatter(
            dec_mean, y, s=64, color=PALETTE["wm_decoupled_no_vis"], edgecolor="white", linewidth=0.8, zorder=3
        )
        better = dec_mean < obj_mean if direction == "down" else dec_mean > obj_mean
        if better:
            ax0.text(
                dec_mean + (0.18 if direction == "down" else 0.25),
                y + 0.12,
                "better",
                color=PALETTE["wm_decoupled_no_vis"],
                fontsize=8,
                ha="left",
            )
    ax0.set_yticks(ypos)
    ax0.set_yticklabels([m[1] for m in metrics])
    ax0.set_title("nuPlan 50k planner-like sanity")
    ax0.set_xlabel("Metric value")
    _beautify_axis(ax0)
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["wm_object"], markersize=8, label="WM-Object"),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["wm_decoupled_no_vis"], markersize=8, label="WM-Decoupled-NoVis"),
    ]
    ax0.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    _add_panel_tag(ax0, "(a)")

    # Right: subset collision comparison.
    ax1 = fig.add_subplot(gs[0, 1])
    subset_order = [
        "lane_conflict",
        "low_ttc_proxy",
        "rare_agent_dense",
        "dense_agents",
        "high_interaction_union",
    ]
    pretty = {
        "lane_conflict": "Lane conflict",
        "low_ttc_proxy": "Low TTC",
        "rare_agent_dense": "Rare + dense",
        "dense_agents": "Dense agents",
        "high_interaction_union": "Interaction union",
    }
    y = np.arange(len(subset_order))
    obj_vals = [subsets["wm_object"][k]["imagined_collision_rate_mean"] for k in subset_order]
    dec_vals = [subsets["wm_decoupled_no_vis"][k]["imagined_collision_rate_mean"] for k in subset_order]
    h = 0.36
    ax1.barh(y + h / 2, obj_vals, height=h, color=PALETTE["wm_object"], label="WM-Object", zorder=3)
    ax1.barh(
        y - h / 2,
        dec_vals,
        height=h,
        color=PALETTE["wm_decoupled_no_vis"],
        label="WM-Decoupled-NoVis",
        zorder=3,
    )
    for idx, (ov, dv) in enumerate(zip(obj_vals, dec_vals)):
        delta = ov - dv
        ax1.text(max(ov, dv) + 0.015, idx, f"Δ {delta:.2f}", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([pretty[k] for k in subset_order])
    ax1.invert_yaxis()
    ax1.set_xlabel("Imagined collision rate")
    ax1.set_title("Interaction-conditioned subsets")
    _beautify_axis(ax1)
    ax1.legend(loc="upper right", frameon=False)
    ax1.annotate(
        "largest gap",
        xy=(dec_vals[0], 0 - h / 2),
        xytext=(0.34, 0.45),
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["wm_decoupled_no_vis"],
            lw=1.0,
        ),
        fontsize=8.5,
        color=PALETTE["wm_decoupled_no_vis"],
    )
    _add_panel_tag(ax1, "(b)")

    fig.suptitle(
        "Downstream offline evidence on nuPlan 50k",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "paper_planner_subset_summary")


def _plot_dataset_stats() -> None:
    stats = _load_json("dataset_token_stats__summary.json")
    nusc = stats["nuscenes"]["metrics"]
    nup = stats["nuplan_50k"]["metrics"]
    metric_order = [
        ("dynamic_tokens_per_sample", "Dynamic tokens / sample"),
        ("rare_tokens_per_sample", "Rare tokens / sample"),
        ("visibility_dynamic", "Dynamic visibility"),
        ("teacher_action_l2", "Teacher action $L_2$"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7))
    for idx, (metric, label) in enumerate(metric_order):
        ax = axes[idx]
        vals = [nusc[metric]["mean"], nup[metric]["mean"]]
        errs = [nusc[metric]["std"], nup[metric]["std"]]
        colors = [PALETTE["nuscenes"], PALETTE["nuplan"]]
        bars = ax.bar(
            [0, 1],
            vals,
            yerr=errs,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            capsize=4,
            zorder=3,
        )
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(errs) * 0.05 + (max(vals) * 0.03 if max(vals) > 1 else 0.02),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["nuScenes", "nuPlan"])
        ax.set_title(label)
        _beautify_axis(ax)
        _add_panel_tag(ax, f"({chr(ord('a') + idx)})")
    fig.suptitle(
        "Dataset-statistics context for the ranking reversal",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "paper_dataset_stats")


def _draw_summary_stage0(ax: plt.Axes, agg: dict) -> None:
    variants = [
        "object_only",
        "object_relation",
        "object_relation_decoupled",
        "object_relation_decoupled_visibility",
    ]
    labels = ["Obj", "Naive", "Dec", "Dec+Vis"]
    metric = "interaction_recall_at_1m"
    means = [agg[v][metric]["mean"] for v in variants]
    stds = [agg[v][metric]["std"] for v in variants]
    ax.bar(
        np.arange(len(variants)),
        means,
        yerr=stds,
        color=[PALETTE[v] for v in variants],
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 1.05)
    ax.set_title("Stage 0 interaction recall")
    ax.set_ylabel("IntRec@1m")
    _beautify_axis(ax)
    ax.annotate(
        "shared relation mixing collapses",
        xy=(1, means[1]),
        xytext=(0.55, 0.48),
        arrowprops=dict(arrowstyle="->", color=PALETTE["object_relation"], lw=1.0),
        fontsize=8,
        color=PALETTE["object_relation"],
    )
    ax.annotate(
        "typed-budget variants recover interaction-critical agents",
        xy=(2, means[2]),
        xytext=(1.9, 0.78),
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["object_relation_decoupled"],
            lw=1.0,
        ),
        fontsize=8,
        color=PALETTE["object_relation_decoupled"],
        ha="left",
    )


def _draw_summary_stage1(ax: plt.Axes, nusc: dict, nup20: dict, nup50: dict) -> None:
    datasets = ["nuScenes", "20k", "50k"]
    xs = np.arange(3)
    for method in ["wm_object", "wm_decoupled_no_vis"]:
        color = PALETTE[method]
        vals = [
            _extract_stage1_point(nusc, method, "imagined_collision_rate")[0],
            _extract_stage1_point(nup20, method, "imagined_collision_rate")[0],
            _extract_stage1_point(nup50, method, "imagined_collision_rate")[0],
        ]
        errs = [
            _extract_stage1_point(nusc, method, "imagined_collision_rate")[1],
            _extract_stage1_point(nup20, method, "imagined_collision_rate")[1],
            _extract_stage1_point(nup50, method, "imagined_collision_rate")[1],
        ]
        ax.errorbar(xs, vals, yerr=errs, color=color, marker="o", lw=2.0, capsize=4, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets)
    ax.set_title("Stage 1 ranking reversal")
    ax.set_ylabel("Collision rate")
    _beautify_axis(ax)
    ax.legend(["WM-Object", "WM-Decoupled-NoVis"], frameon=False, loc="upper right")


def _draw_summary_subsets(ax: plt.Axes, subsets: dict) -> None:
    keys = ["lane_conflict", "low_ttc_proxy", "high_interaction_union"]
    pretty = {
        "lane_conflict": "Lane conflict",
        "low_ttc_proxy": "Low TTC",
        "high_interaction_union": "Interaction union",
    }
    y = np.arange(len(keys))
    obj = [subsets["wm_object"][k]["imagined_collision_rate_mean"] for k in keys]
    dec = [subsets["wm_decoupled_no_vis"][k]["imagined_collision_rate_mean"] for k in keys]
    h = 0.32
    ax.barh(y + h / 2, obj, height=h, color=PALETTE["wm_object"], zorder=3)
    ax.barh(y - h / 2, dec, height=h, color=PALETTE["wm_decoupled_no_vis"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Collision rate")
    ax.set_title("Interaction-heavy subsets")
    _beautify_axis(ax)


def _draw_summary_stats(ax: plt.Axes, stats: dict) -> None:
    nusc = stats["nuscenes"]["metrics"]
    nup = stats["nuplan_50k"]["metrics"]
    metrics = ["dynamic_tokens_per_sample", "rare_tokens_per_sample", "teacher_action_l2"]
    x = np.arange(len(metrics))
    nusc_vals = [nusc[m]["mean"] for m in metrics]
    nup_vals = [nup[m]["mean"] for m in metrics]
    w = 0.36
    ax.bar(x - w / 2, nusc_vals, width=w, color=PALETTE["nuscenes"], zorder=3)
    ax.bar(x + w / 2, nup_vals, width=w, color=PALETTE["nuplan"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Dyn", "Rare", "Act-$L_2$"])
    ax.set_title("Dataset context")
    _beautify_axis(ax)
    ax.legend(["nuScenes", "nuPlan"], frameon=False, loc="upper left")


def _plot_summary_composite() -> None:
    agg = _load_json("experiments__table3_fair_fix2_aggregate.json")["metrics"]
    nusc = _load_json("stage1_pilot_x__X_summary.json")
    nup20 = _load_json("nuplan_stage1_20k__summary.json")
    nup50 = _load_json("nuplan_stage1_50k__summary.json")
    subsets = _load_json("nuplan_interaction_subset_50k__summary.json")["summary"]
    stats = _load_json("dataset_token_stats__summary.json")

    fig = plt.figure(figsize=(13.0, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_summary_stage0(ax_a, agg)
    _add_panel_tag(ax_a, "(a)")
    _draw_summary_stage1(ax_b, nusc, nup20, nup50)
    _add_panel_tag(ax_b, "(b)")
    _draw_summary_subsets(ax_c, subsets)
    _add_panel_tag(ax_c, "(c)")
    _draw_summary_stats(ax_d, stats)
    _add_panel_tag(ax_d, "(d)")

    fig.suptitle(
        "Compact summary of the paper's data-driven evidence",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "paper_summary_charts")


def main() -> None:
    _setup_style()
    _plot_stage0_slot_budget()
    _plot_stage0_metrics()
    _plot_stage1_cross_dataset()
    _plot_planner_subset()
    _plot_dataset_stats()
    _plot_summary_composite()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
