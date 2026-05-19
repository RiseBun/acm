#!/usr/bin/env python3
"""Build paper-ready closure figures/tables for the NeurIPS reframe.

This script consumes existing experiment outputs only. It does not train or
evaluate models. Outputs are written under:

  期刊/paper_assets/neurips_closure_2026-04-28/
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "期刊" / "paper_assets" / "neurips_closure_2026-04-28"
FIG = OUT / "figures"
TAB = OUT / "tables"


COLORS = {
    "wm_naive": "#D55E5E",
    "wm_object": "#6B7280",
    "wm_decoupled_no_vis": "#1F77B4",
    "decoupled": "#1F77B4",
    "object": "#6B7280",
    "naive": "#D55E5E",
    "green": "#2CA58D",
    "orange": "#F2A65A",
}


def setup_style() -> None:
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
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def selection_rows() -> List[dict]:
    rows: List[dict] = []

    shared = read_json(
        ROOT / "experiments" / "selection_diagnostic_shared_relation_20k" / "summary.json"
    )["results"]["wm_naive"]
    for seed, block in shared.items():
        m = block["mean"]
        c = block["correlation"]
        rows.append(
            {
                "condition": "wm_naive",
                "dataset": "nuPlan20k",
                "seed": int(seed),
                "cdr": m["cdr"],
                "miss": m["critical_agent_miss_rate"],
                "wasted": m["relation_wasted_slot_rate"],
                "roi": m["relation_overallocation_index"],
                "rare_ade": m["rare_ade"],
                "intrec": m["interaction_recall_at_1m"],
                "miss_intrec_spearman": c["miss_rate__vs__interaction_recall_at_1m"][
                    "spearman"
                ],
            }
        )

    main = read_json(
        ROOT / "experiments" / "selection_diagnostic_nuplan50k" / "summary.json"
    )["results"]
    for condition in ("wm_object", "wm_decoupled_no_vis"):
        for seed, block in main[condition].items():
            m = block["mean"]
            c = block["correlation"]
            rows.append(
                {
                    "condition": condition,
                    "dataset": "nuPlan50k",
                    "seed": int(seed),
                    "cdr": m["cdr"],
                    "miss": m["critical_agent_miss_rate"],
                    "wasted": m["relation_wasted_slot_rate"],
                    "roi": m["relation_overallocation_index"],
                    "rare_ade": m["rare_ade"],
                    "intrec": m["interaction_recall_at_1m"],
                    "miss_intrec_spearman": c[
                        "miss_rate__vs__interaction_recall_at_1m"
                    ]["spearman"],
                }
            )
    return rows


def write_selection_table(rows: List[dict]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    path = TAB / "selection_diagnostic_table.csv"
    fields = [
        "condition",
        "dataset",
        "seed",
        "cdr",
        "miss",
        "wasted",
        "roi",
        "rare_ade",
        "intrec",
        "miss_intrec_spearman",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_selection(rows: List[dict]) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    out: Dict[str, dict] = {}
    for condition, rs in groups.items():
        out[condition] = {
            "cdr_mean": mean(r["cdr"] for r in rs),
            "cdr_std": std(r["cdr"] for r in rs),
            "miss_mean": mean(r["miss"] for r in rs),
            "miss_std": std(r["miss"] for r in rs),
            "wasted_mean": mean(r["wasted"] for r in rs if r["wasted"] is not None),
            "wasted_std": std(r["wasted"] for r in rs if r["wasted"] is not None),
            "roi_mean": mean(r["roi"] for r in rs if r["roi"] is not None),
            "roi_std": std(r["roi"] for r in rs if r["roi"] is not None),
            "rare_ade_mean": mean(r["rare_ade"] for r in rs),
            "intrec_mean": mean(r["intrec"] for r in rs),
            "miss_intrec_rho_mean": mean(r["miss_intrec_spearman"] for r in rs),
        }
    return out


def plot_selection_bars(rows: List[dict]) -> None:
    grouped = grouped_selection(rows)
    order = ["wm_naive", "wm_object", "wm_decoupled_no_vis"]
    labels = ["Naive\nshared", "Object\nonly", "Typed\nbudget"]
    metrics = [
        ("cdr_mean", "cdr_std", "Critical Dynamic Retention", "higher"),
        ("miss_mean", "miss_std", "Critical Agent Miss Rate", "lower"),
        ("wasted_mean", "wasted_std", "Relation Wasted Slot Rate", "lower"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharex=False)
    for ax, (key, err_key, title, _) in zip(axes, metrics):
        vals = [grouped[o][key] for o in order]
        errs = [grouped[o][err_key] for o in order]
        colors = [COLORS["wm_naive"], COLORS["wm_object"], COLORS["wm_decoupled_no_vis"]]
        ax.bar(range(len(order)), vals, yerr=errs, capsize=3, color=colors, edgecolor="#333")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, max(v + e for v, e in zip(vals, errs)) * 1.15))
    save(fig, "fig_selection_diagnostics_bars")


def read_sample_metric_csv(path: Path, condition: str, max_rows: int = 5000) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                miss = float(row["critical_agent_miss_rate"])
                intrec = float(row["interaction_recall_at_1m"])
                rare = float(row["rare_ade"])
            except Exception:
                continue
            if not (math.isfinite(miss) and math.isfinite(intrec)):
                continue
            rows.append(
                {
                    "condition": condition,
                    "miss": miss,
                    "intrec": intrec,
                    "rare_ade": rare,
                }
            )
    return rows


def plot_missrate_scatter() -> None:
    data: List[dict] = []
    for seed in (7, 42, 123):
        data += read_sample_metric_csv(
            ROOT
            / "experiments"
            / "selection_diagnostic_shared_relation_20k"
            / f"seed{seed}"
            / "wm_naive"
            / "sample_mechanism_metrics.csv",
            "Naive shared",
            max_rows=4000,
        )
        data += read_sample_metric_csv(
            ROOT
            / "experiments"
            / "selection_diagnostic_nuplan50k"
            / f"seed{seed}"
            / "wm_decoupled_no_vis"
            / "sample_mechanism_metrics.csv",
            "Typed budget",
            max_rows=4000,
        )

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for condition, color in [
        ("Naive shared", COLORS["wm_naive"]),
        ("Typed budget", COLORS["wm_decoupled_no_vis"]),
    ]:
        xs = [r["miss"] for r in data if r["condition"] == condition]
        ys = [r["intrec"] for r in data if r["condition"] == condition]
        ax.scatter(xs, ys, s=8, alpha=0.18, color=color, label=condition, edgecolors="none")

    ax.set_xlabel("Critical Agent Miss Rate")
    ax.set_ylabel("Interaction Recall @ 1m")
    ax.set_title("Missing critical agents predicts interaction loss")
    ax.legend(frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    save(fig, "fig_missrate_vs_interaction_recall")


def plot_relation_ablation() -> None:
    payload = read_json(
        ROOT / "experiments" / "nuplan_relation_feature_ablation_50k" / "summary.json"
    )["summary"]["wm_decoupled_no_vis"]
    order = ["none", "no_ttc_risk", "no_lane_priority", "no_relation_semantics"]
    labels = ["None", "No\nTTC/risk", "No\nlane/priority", "No relation\nsemantics"]
    metrics = [
        ("imagined_collision_rate", "Collision rate"),
        ("teacher_action_mse", "Teacher action MSE"),
        ("latent_return_mean", "Latent return"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [payload[k]["mean"][metric] for k in order]
        errs = [payload[k]["std_across_seeds"][metric] for k in order]
        ax.bar(
            range(len(order)),
            vals,
            yerr=errs,
            capsize=3,
            color=[COLORS["wm_decoupled_no_vis"], COLORS["wm_naive"], COLORS["orange"], "#A855F7"],
            edgecolor="#333333",
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
    save(fig, "fig_relation_semantics_ablation")


def plot_budget_sensitivity() -> None:
    payload = read_json(
        ROOT / "experiments" / "stage0_budget_10_6_nuscenes" / "summary.json"
    )
    ten = payload["aggregate"]
    twelve = payload["baseline_12_4"]
    metrics = [
        ("dyn_rollout_mse", "Dyn rollout MSE"),
        ("rare_ade", "Rare ADE"),
        ("interaction_recall_at_1m", "IntRec@1m"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [twelve[metric]["mean"], ten[metric]["mean"]]
        errs = [twelve[metric]["seed_std"], ten[metric]["seed_std"]]
        ax.bar(
            [0, 1],
            vals,
            yerr=errs,
            capsize=3,
            color=[COLORS["wm_decoupled_no_vis"], COLORS["orange"]],
            edgecolor="#333333",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["12/4\nmain", "10/6\nrel-heavy"])
        ax.set_title(title)
    save(fig, "fig_budget_sensitivity_12_4_vs_10_6")


def write_summary(rows: List[dict]) -> None:
    grouped = grouped_selection(rows)
    relation = read_json(
        ROOT / "experiments" / "nuplan_relation_feature_ablation_50k" / "summary.json"
    )["summary"]["wm_decoupled_no_vis"]
    bc = read_json(ROOT / "experiments" / "nuplan_bc_baseline_50k" / "summary.json")[
        "aggregate"
    ]

    def fmt(value: float) -> str:
        return f"{value:.3f}" if math.isfinite(float(value)) else "N/A"

    lines = [
        "# NeurIPS Closure Artifacts",
        "",
        "This folder contains paper-ready artifacts generated from existing experiments. No model was trained or evaluated by this script.",
        "",
        "## Generated Figures",
        "",
        "- `figures/fig_selection_diagnostics_bars.{png,pdf}`",
        "- `figures/fig_missrate_vs_interaction_recall.{png,pdf}`",
        "- `figures/fig_relation_semantics_ablation.{png,pdf}`",
        "- `figures/fig_budget_sensitivity_12_4_vs_10_6.{png,pdf}`",
        "",
        "## Selection Diagnostic Takeaways",
        "",
        "| condition | CDR mean | MissRate mean | WastedRel mean | ROI mean | MissRate~IntRec rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("wm_naive", "wm_object", "wm_decoupled_no_vis"):
        g = grouped[condition]
        lines.append(
            f"| {condition} | {fmt(g['cdr_mean'])} | {fmt(g['miss_mean'])} | "
            f"{fmt(g['wasted_mean'])} | {fmt(g['roi_mean'])} | "
            f"{fmt(g['miss_intrec_rho_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Relation Semantic Ablation",
            "",
            f"- Baseline collision rate: {relation['none']['mean']['imagined_collision_rate']:.3f}",
            f"- Removing TTC/risk collision rate: {relation['no_ttc_risk']['mean']['imagined_collision_rate']:.3f}",
            f"- Removing lane/priority collision rate: {relation['no_lane_priority']['mean']['imagined_collision_rate']:.3f}",
            "",
            "## External-Style BC Anchor",
            "",
            f"- BC latent return: {bc['latent_return_mean']['mean']:.3f} +/- {bc['latent_return_mean']['seed_std']:.3f}",
            f"- BC teacher action MSE: {bc['teacher_action_mse']['mean']:.3f} +/- {bc['teacher_action_mse']['seed_std']:.3f}",
            "",
            "## Remaining Optional Experiment",
            "",
            "- `wm_naive` nuPlan 50k Stage1 training remains optional. Current Stage0 50k artifacts do not include an `object_relation/model.pt` warm-start, so this job would likely train from scratch or require first producing the Stage0 warm-start.",
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = selection_rows()
    write_selection_table(rows)
    plot_selection_bars(rows)
    plot_missrate_scatter()
    plot_relation_ablation()
    plot_budget_sensitivity()
    write_summary(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
