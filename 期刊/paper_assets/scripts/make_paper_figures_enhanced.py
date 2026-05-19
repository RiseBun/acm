#!/usr/bin/env python3
"""
Enhanced paper figures for IEEE T-IV submission
Professional visualization with improved aesthetics and clarity
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper_assets" / "data"
FIG_DIR = ROOT / "figures" / "paper_enhanced"
PKL_PATH = ROOT / "figures" / "slot_selections_seed7.pkl"

# IEEE-compliant color palette (colorblind-safe, print-friendly)
PALETTE = {
    "object_only": "#4A6FA5",  # Professional blue-gray
    "object_relation": "#E06C75",  # Soft red
    "object_relation_visibility": "#F4A261",  # Warm orange
    "object_relation_decoupled": "#2A9D8F",  # Teal (our method)
    "object_relation_decoupled_visibility": "#264653",  # Dark teal
    "holistic_16slot": "#A8AAB5",  # Light gray
    "holistic": "#6C757D",  # Medium gray
    "wm_object": "#4A6FA5",
    "wm_decoupled": "#2A9D8F",
    "wm_decoupled_no_vis": "#E76F51",  # Burnt orange for emphasis
    "nuscenes": "#6C757D",
    "nuplan": "#4A6FA5",
}

LABELS = {
    "holistic_16slot": "Holistic-16",
    "object_only": "Object-Only",
    "object_relation": "Naive Obj+Rel",
    "object_relation_visibility": "Naive+Vis",
    "object_relation_decoupled": "Decoupled (Ours)",
    "object_relation_decoupled_visibility": "Decoupled+Vis (Ours)",
    "holistic": "Holistic-full",
    "wm_object": "WM-Object",
    "wm_decoupled": "WM-Decoupled",
    "wm_decoupled_no_vis": "WM-Decoupled-NoVis",
}


def _setup_style() -> None:
    """Configure matplotlib for IEEE-quality figures"""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _load_json(name: str) -> dict:
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def _beautify_axis(ax: plt.Axes) -> None:
    """Clean, professional axis styling"""
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5, color="#999999")
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(direction="out", length=4, width=0.8)


def _add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.08,
        1.02,
        tag,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        fontfamily="serif",
    )


def _plot_stage0_slot_budget() -> None:
    """Enhanced slot budget distribution plot"""
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    variants = payload["variants"]
    type_names = {0: "EGO", 1: "VEH", 2: "PED", 3: "CYC", 4: "MAP", 6: "REL"}
    type_order = [0, 1, 2, 3, 4, 6]
    
    # More professional color scheme for token types
    type_colors = {
        0: "#D62728",  # Red for ego
        1: "#1F77B4",  # Blue for vehicle
        2: "#2CA02C",  # Green for pedestrian
        3: "#FF7F0E",  # Orange for cyclist
        4: "#9467BD",  # Purple for map
        6: "#E377C2",  # Pink for relation
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
        "Holistic-16",
        "Object-Only",
        "Naive Obj+Rel",
        "Naive+Vis",
        "Decoupled\n(Ours)",
        "Decoupled+Vis\n(Ours)",
        "Holistic-full",
    ]
    x = np.arange(len(variant_order))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    
    # Plot learned/mixed baseline
    learned_heights = []
    for vid in variant_order:
        counts, k, is_set = stats[vid]
        learned_heights.append(k if is_set else 0.0)
    
    ax.bar(
        x,
        learned_heights,
        color="#CCCCCC",
        edgecolor="#666666",
        linewidth=0.6,
        hatch="///",
        label="Learned selection",
        zorder=2,
        alpha=0.7,
    )

    # Stacked bars for token types
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
            linewidth=0.5,
            label=type_names[token_type],
            zorder=3,
        )
        
        # Add labels to larger segments
        for bar, h, b in zip(bars, heights, bottoms):
            if h >= 0.8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    b + h / 2.0,
                    f"{h:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
        bottoms += heights

    # Add budget line
    ax.axhline(16, color="#333333", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(
        len(variant_order) - 0.3,
        16.3,
        "16-slot budget",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
        fontweight="bold",
    )
    
    # Annotations
    ax.annotate(
        "REL over-allocation\nstarves dynamic agents",
        xy=(2, 14.5),
        xytext=(1.3, 22.0),
        arrowprops=dict(arrowstyle="->", color=PALETTE["object_relation"], lw=1.5),
        fontsize=8,
        color=PALETTE["object_relation"],
        ha="center",
        fontweight="bold",
    )
    
    ax.annotate(
        "Typed budgets preserve\ndynamic context",
        xy=(4, 12.5),
        xytext=(4.8, 24.0),
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["object_relation_decoupled"],
            lw=1.5,
        ),
        fontsize=8,
        color=PALETTE["object_relation_decoupled"],
        ha="center",
        fontweight="bold",
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Average slots per sample", fontsize=9)
    ax.set_ylim(0, 28)
    _beautify_axis(ax)
    ax.legend(
        ncol=7, 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15), 
        frameon=False,
        fontsize=7,
    )
    
    _save(fig, "paper_stage0_slot_budget")


def _plot_stage0_metrics() -> None:
    """Enhanced Stage 0 metrics visualization"""
    agg = _load_json("experiments__table3_fair_fix2_aggregate.json")["metrics"]
    metrics = [
        ("dyn_rollout_mse", "Dyn Rollout MSE", False),
        ("rare_ade", "Rare ADE (m)", False),
        ("interaction_recall_at_1m", "Interaction Recall @ 1m", True),
    ]
    variants = [
        "object_only",
        "object_relation",
        "object_relation_decoupled",
        "object_relation_decoupled_visibility",
    ]
    labels = ["Object-\nOnly", "Naive\nObj+Rel", "Decoupled\n(Ours)", "Decoupled\n+Vis (Ours)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    for idx, (metric_key, title, higher_better) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(variants))
        means = [agg[v][metric_key]["mean"] for v in variants]
        stds = [agg[v][metric_key]["std"] for v in variants]
        seed_values = [agg[v][metric_key]["values"] for v in variants]
        colors = [PALETTE[v] for v in variants]
        
        # Bars with error bars
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="white",
            linewidth=1.0,
            capsize=5,
            zorder=3,
            width=0.6,
        )
        
        # Individual seed points with jitter
        rng = np.random.default_rng(17 + idx)
        for xpos, vals in zip(x, seed_values):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=25,
                color="#1F2937",
                alpha=0.7,
                zorder=4,
                edgecolors="white",
                linewidth=0.5,
            )
        
        # Value labels on bars
        for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
            offset = (max(means) * 0.02) if max(means) > 0 else 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset + s,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
        
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Value" if not higher_better else "Value", fontsize=8)
        _beautify_axis(ax)
        
        # Set appropriate y-limits
        if metric_key == "interaction_recall_at_1m":
            ax.set_ylim(0.3, 1.05)
        elif metric_key == "rare_ade":
            ax.set_ylim(0, max(means) * 1.15)
            # Add improvement annotation
            improvement = (means[0] - means[2]) / means[0] * 100
            ax.annotate(
                f"↓{improvement:.1f}%",
                xy=(2, means[2]),
                xytext=(2.6, max(means) * 0.85),
                arrowprops=dict(
                    arrowstyle="->",
                    color=PALETTE["object_relation_decoupled"],
                    lw=1.5,
                ),
                fontsize=8.5,
                color=PALETTE["object_relation_decoupled"],
                ha="left",
                fontweight="bold",
            )
        elif metric_key == "dyn_rollout_mse":
            ax.set_ylim(0, max([m for m in means if m < 20]) * 1.3)
        
        _add_panel_tag(ax, f"({chr(ord('a') + idx)})")
    
    fig.suptitle(
        "Stage 0: Representation Sufficiency under Fair 16-Slot Budget",
        y=0.98,
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.12, hspace=0.3, wspace=0.3)
    _save(fig, "paper_stage0_metrics")


def _plot_stage1_cross_dataset() -> None:
    """Enhanced cross-dataset policy learning visualization"""
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

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    metric_specs = [
        ("latent_return_mean", "Imagined Return", True),
        ("imagined_collision_rate", "Collision Rate", False),
    ]
    
    for panel_idx, (metric_key, ylabel, higher_better) in enumerate(metric_specs):
        ax = axes[panel_idx]
        
        for method_key, method_label in methods:
            color = PALETTE[method_key]
            means, stds, valid_x = [], [], []
            
            for i, summary in enumerate(source):
                if "conditions" in summary and method_key not in summary["conditions"]:
                    continue
                if "conditions" not in summary and method_key not in summary:
                    continue
                if "conditions" in summary:
                    cond = summary["conditions"][method_key]
                    mean = float(cond["mean"][metric_key])
                    std = float(cond["std_across_seeds"][metric_key])
                else:
                    cond = summary[method_key][metric_key]
                    mean = float(cond["mean"])
                    std = float(cond["std"])
                valid_x.append(i)
                means.append(mean)
                stds.append(std)
            
            if not valid_x:
                continue
                
            linewidth = 2.5 if method_key == "wm_decoupled_no_vis" else 1.8
            markersize = 8 if method_key == "wm_decoupled_no_vis" else 6
            
            ax.errorbar(
                valid_x,
                means,
                yerr=stds,
                color=color,
                marker="o",
                markersize=markersize,
                linewidth=linewidth,
                capsize=5,
                elinewidth=1.5,
                capthick=1.5,
                label=method_label,
                zorder=3,
            )
        
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        _beautify_axis(ax)
        
        # Background shading for datasets
        if metric_key == "latent_return_mean":
            ax.axvspan(-0.45, 0.45, color="#F8F9FA", alpha=0.5, zorder=0)
            ax.axvspan(0.55, 2.45, color="#E3F2FD", alpha=0.4, zorder=0)
            
            # Annotations for key insights
            ax.annotate(
                "Object-only\nmost stable",
                xy=(0, 31.79),
                xytext=(0.15, 40.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["wm_object"], lw=1.2),
                fontsize=7.5,
                color=PALETTE["wm_object"],
                ha="left",
            )
            ax.annotate(
                "Decoupled-NoVis\ndominates",
                xy=(1.8, 16.0),
                xytext=(1.2, 32.0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=PALETTE["wm_decoupled_no_vis"],
                    lw=1.2,
                ),
                fontsize=7.5,
                color=PALETTE["wm_decoupled_no_vis"],
                ha="left",
            )
        
        _add_panel_tag(ax, f"({chr(ord('a') + panel_idx)})")
    
    axes[1].legend(loc="upper right", frameon=False, fontsize=7.5)
    fig.suptitle(
        "Stage 1: Cross-Dataset Policy Learning Ranking Reversal",
        y=1.02,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "paper_stage1_cross_dataset")


def _plot_planner_subset() -> None:
    """Enhanced planner sanity and subset analysis"""
    planner = _load_json("nuplan_planner_sanity_50k__summary.json")["summary"]
    subsets = _load_json("nuplan_interaction_subset_50k__summary.json")["summary"]

    fig = plt.figure(figsize=(11, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.25)

    # Left: planner-like sanity as paired horizontal dumbbells
    ax0 = fig.add_subplot(gs[0, 0])
    metrics = [
        ("teacher_action_mse", "Teacher Action MSE", "down"),
        ("latent_return_mean", "Imagined Return", "up"),
        ("imagined_collision_rate", "Collision Rate", "down"),
    ]
    ypos = np.arange(len(metrics))[::-1]
    
    for y, (metric, label, direction) in zip(ypos, metrics):
        obj_mean = planner["wm_object"]["mean"][metric]
        dec_mean = planner["wm_decoupled_no_vis"]["mean"][metric]
        
        # Connecting line
        ax0.plot([obj_mean, dec_mean], [y, y], color="#CCCCCC", lw=2.5, zorder=1)
        
        # Data points with size based on importance
        ax0.scatter(
            obj_mean, y, s=80, color=PALETTE["wm_object"], 
            edgecolor="white", linewidth=1.0, zorder=3, alpha=0.9
        )
        ax0.scatter(
            dec_mean, y, s=100, color=PALETTE["wm_decoupled_no_vis"], 
            edgecolor="white", linewidth=1.2, zorder=3, alpha=0.95
        )
        
        # Add value labels
        ax0.text(
            obj_mean, y - 0.15, f"{obj_mean:.2f}",
            ha="center", va="top", fontsize=7, color=PALETTE["wm_object"]
        )
        ax0.text(
            dec_mean, y + 0.15, f"{dec_mean:.2f}",
            ha="center", va="bottom", fontsize=7, 
            color=PALETTE["wm_decoupled_no_vis"], fontweight="bold"
        )
        
        # Improvement indicator
        better = dec_mean < obj_mean if direction == "down" else dec_mean > obj_mean
        if better:
            improvement = abs(dec_mean - obj_mean) / obj_mean * 100
            ax0.text(
                (obj_mean + dec_mean) / 2,
                y + 0.22,
                f"↓{improvement:.1f}%" if direction == "down" else f"↑{improvement:.1f}%",
                color=PALETTE["wm_decoupled_no_vis"],
                fontsize=7.5,
                ha="center",
                fontweight="bold",
            )
    
    ax0.set_yticks(ypos)
    ax0.set_yticklabels([m[1] for m in metrics], fontsize=8)
    ax0.set_title("nuPlan 50k: Planner-Like Sanity Check", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Metric Value", fontsize=8)
    _beautify_axis(ax0)
    
    legend_handles = [
        mpl.lines.Line2D(
            [0], [0], marker="o", color="w", 
            markerfacecolor=PALETTE["wm_object"], 
            markersize=9, label="WM-Object"
        ),
        mpl.lines.Line2D(
            [0], [0], marker="o", color="w", 
            markerfacecolor=PALETTE["wm_decoupled_no_vis"], 
            markersize=10, label="WM-Decoupled-NoVis"
        ),
    ]
    ax0.legend(handles=legend_handles, loc="lower center", 
               bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=7.5)
    _add_panel_tag(ax0, "(a)")

    # Right: subset collision comparison
    ax1 = fig.add_subplot(gs[0, 1])
    subset_order = [
        "lane_conflict",
        "low_ttc_proxy",
        "rare_agent_dense",
        "dense_agents",
        "high_interaction_union",
    ]
    pretty = {
        "lane_conflict": "Lane Conflict",
        "low_ttc_proxy": "Low TTC",
        "rare_agent_dense": "Rare + Dense",
        "dense_agents": "Dense Agents",
        "high_interaction_union": "Interaction Union",
    }
    y = np.arange(len(subset_order))
    obj_vals = [subsets["wm_object"][k]["imagined_collision_rate_mean"] for k in subset_order]
    dec_vals = [subsets["wm_decoupled_no_vis"][k]["imagined_collision_rate_mean"] for k in subset_order]
    
    h = 0.35
    bars1 = ax1.barh(
        y + h / 2, obj_vals, height=h, 
        color=PALETTE["wm_object"], label="WM-Object", zorder=3, alpha=0.85
    )
    bars2 = ax1.barh(
        y - h / 2, dec_vals, height=h,
        color=PALETTE["wm_decoupled_no_vis"], 
        label="WM-Decoupled-NoVis", zorder=3, alpha=0.9
    )
    
    # Add delta labels
    for idx, (ov, dv) in enumerate(zip(obj_vals, dec_vals)):
        delta = ov - dv
        ax1.text(
            max(ov, dv) + 0.01, idx, 
            f"Δ={delta:.2f}", va="center", fontsize=7,
            color=PALETTE["wm_decoupled_no_vis"], fontweight="bold"
        )
    
    ax1.set_yticks(y)
    ax1.set_yticklabels([pretty[k] for k in subset_order], fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Collision Rate", fontsize=8)
    ax1.set_title("Interaction-Conditioned Subsets", fontsize=9, fontweight="bold")
    _beautify_axis(ax1)
    ax1.legend(loc="upper right", frameon=False, fontsize=7.5)
    
    # Highlight largest gap
    max_delta_idx = np.argmax([ov - dv for ov, dv in zip(obj_vals, dec_vals)])
    ax1.annotate(
        "Largest\ngap",
        xy=(dec_vals[max_delta_idx], max_delta_idx - h / 2),
        xytext=(0.32, 0.6),
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["wm_decoupled_no_vis"],
            lw=1.5,
        ),
        fontsize=7.5,
        color=PALETTE["wm_decoupled_no_vis"],
        ha="center",
        fontweight="bold",
    )
    _add_panel_tag(ax1, "(b)")

    fig.suptitle(
        "Downstream Evidence on nuPlan 50k",
        y=1.02,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "paper_planner_subset_summary")


def _plot_dataset_stats() -> None:
    """Enhanced dataset statistics comparison"""
    stats = _load_json("dataset_token_stats__summary.json")
    nusc = stats["nuscenes"]["metrics"]
    nup = stats["nuplan_50k"]["metrics"]
    metric_order = [
        ("dynamic_tokens_per_sample", "Dynamic Tokens\nper Sample"),
        ("rare_tokens_per_sample", "Rare Tokens\nper Sample"),
        ("visibility_dynamic", "Dynamic\nVisibility"),
        ("teacher_action_l2", "Teacher Action\n$L_2$ Norm"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    
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
            capsize=5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            zorder=3,
            width=0.5,
        )
        
        # Value labels
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + errs[j] + max(vals) * 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["nuScenes", "nuPlan"], fontsize=7.5)
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        _beautify_axis(ax)
        _add_panel_tag(ax, f"({chr(ord('a') + idx)})")
    
    fig.suptitle(
        "Dataset Statistics Context for Ranking Reversal",
        y=1.02,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "paper_dataset_stats")


def _plot_summary_composite() -> None:
    """Enhanced composite summary figure"""
    agg = _load_json("experiments__table3_fair_fix2_aggregate.json")["metrics"]
    nusc = _load_json("stage1_pilot_x__X_summary.json")
    nup20 = _load_json("nuplan_stage1_20k__summary.json")
    nup50 = _load_json("nuplan_stage1_50k__summary.json")
    subsets = _load_json("nuplan_interaction_subset_50k__summary.json")["summary"]
    stats = _load_json("dataset_token_stats__summary.json")

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel (a): Stage 0 interaction recall
    variants = ["object_only", "object_relation", "object_relation_decoupled", "object_relation_decoupled_visibility"]
    labels_short = ["Obj", "Naive", "Dec", "Dec+Vis"]
    metric = "interaction_recall_at_1m"
    means = [agg[v][metric]["mean"] for v in variants]
    stds = [agg[v][metric]["std"] for v in variants]
    
    ax_a.bar(
        np.arange(len(variants)),
        means,
        yerr=stds,
        color=[PALETTE[v] for v in variants],
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        zorder=3,
        width=0.55,
    )
    ax_a.set_xticks(np.arange(len(variants)))
    ax_a.set_xticklabels(labels_short, fontsize=7.5)
    ax_a.set_ylim(0.3, 1.08)
    ax_a.set_title("Stage 0: Interaction Recall", fontsize=9, fontweight="bold")
    ax_a.set_ylabel("IntRec@1m", fontsize=8)
    _beautify_axis(ax_a)
    _add_panel_tag(ax_a, "(a)")

    # Panel (b): Stage 1 ranking reversal
    datasets = ["nuScenes", "20k", "50k"]
    xs = np.arange(3)
    for method in ["wm_object", "wm_decoupled_no_vis"]:
        color = PALETTE[method]
        vals, errs = [], []
        for summary in [nusc, nup20, nup50]:
            if "conditions" in summary:
                cond = summary["conditions"][method]
                vals.append(float(cond["mean"]["imagined_collision_rate"]))
                errs.append(float(cond["std_across_seeds"]["imagined_collision_rate"]))
            else:
                vals.append(float(summary[method]["imagined_collision_rate"]["mean"]))
                errs.append(float(summary[method]["imagined_collision_rate"]["std"]))
        
        linewidth = 2.5 if method == "wm_decoupled_no_vis" else 1.8
        ax_b.errorbar(
            xs, vals, yerr=errs, color=color, marker="o", 
            lw=linewidth, capsize=4, zorder=3,
            elinewidth=1.2, capthick=1.2
        )
    
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(datasets, fontsize=7.5)
    ax_b.set_title("Stage 1: Ranking Reversal", fontsize=9, fontweight="bold")
    ax_b.set_ylabel("Collision Rate", fontsize=8)
    _beautify_axis(ax_b)
    ax_b.legend(["WM-Object", "WM-Decoupled-NoVis"], frameon=False, 
                loc="upper right", fontsize=7)
    _add_panel_tag(ax_b, "(b)")

    # Panel (c): Interaction-heavy subsets
    keys = ["lane_conflict", "low_ttc_proxy", "high_interaction_union"]
    pretty = {
        "lane_conflict": "Lane Conflict",
        "low_ttc_proxy": "Low TTC",
        "high_interaction_union": "Interaction Union",
    }
    y = np.arange(len(keys))
    obj = [subsets["wm_object"][k]["imagined_collision_rate_mean"] for k in keys]
    dec = [subsets["wm_decoupled_no_vis"][k]["imagined_collision_rate_mean"] for k in keys]
    h = 0.3
    ax_c.barh(y + h / 2, obj, height=h, color=PALETTE["wm_object"], zorder=3, alpha=0.85)
    ax_c.barh(y - h / 2, dec, height=h, color=PALETTE["wm_decoupled_no_vis"], zorder=3, alpha=0.9)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([pretty[k] for k in keys], fontsize=7.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Collision Rate", fontsize=8)
    ax_c.set_title("Interaction Subsets", fontsize=9, fontweight="bold")
    _beautify_axis(ax_c)
    _add_panel_tag(ax_c, "(c)")

    # Panel (d): Dataset context
    nusc_stats = stats["nuscenes"]["metrics"]
    nup_stats = stats["nuplan_50k"]["metrics"]
    metrics_list = ["dynamic_tokens_per_sample", "rare_tokens_per_sample", "teacher_action_l2"]
    x = np.arange(len(metrics_list))
    nusc_vals = [nusc_stats[m]["mean"] for m in metrics_list]
    nup_vals = [nup_stats[m]["mean"] for m in metrics_list]
    w = 0.3
    ax_d.bar(x - w / 2, nusc_vals, width=w, color=PALETTE["nuscenes"], zorder=3, alpha=0.85)
    ax_d.bar(x + w / 2, nup_vals, width=w, color=PALETTE["nuplan"], zorder=3, alpha=0.9)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Dynamic", "Rare", "Action"], fontsize=7.5)
    ax_d.set_title("Dataset Context", fontsize=9, fontweight="bold")
    _beautify_axis(ax_d)
    ax_d.legend(["nuScenes", "nuPlan"], frameon=False, loc="upper left", fontsize=7)
    _add_panel_tag(ax_d, "(d)")

    fig.suptitle(
        "Summary: Key Evidence Supporting Our Claims",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "paper_summary_charts")


def main() -> None:
    _setup_style()
    print("Generating enhanced Stage 0 slot budget figure...")
    _plot_stage0_slot_budget()
    print("Generating enhanced Stage 0 metrics figure...")
    _plot_stage0_metrics()
    print("Generating enhanced Stage 1 cross-dataset figure...")
    _plot_stage1_cross_dataset()
    print("Generating enhanced planner subset figure...")
    _plot_planner_subset()
    print("Generating enhanced dataset stats figure...")
    _plot_dataset_stats()
    print("Generating enhanced summary composite figure...")
    _plot_summary_composite()
    print(f"\n✓ All enhanced figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
