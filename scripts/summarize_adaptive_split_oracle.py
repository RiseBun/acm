#!/usr/bin/env python3
"""Summarize true Stage-0 oracle/adaptive split results.

Input layout is the output of ``run_stage0_table3.py --dump-per-sample-metrics``:

    <experiment_dir>/split_14_2/object_relation_decoupled/per_sample_metrics.jsonl
    <experiment_dir>/split_12_4/object_relation_decoupled/per_sample_metrics.jsonl
    <experiment_dir>/split_10_6/object_relation_decoupled/per_sample_metrics.jsonl

The script selects a split per scene, then aggregates the original per-sample
metric numerators/denominators so the reported numbers use the same definitions
as Table 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence


LOWER_IS_BETTER = ("dyn_rollout_mse", "rare_ade")
HIGHER_IS_BETTER = ("interaction_recall_at_1m", "collision_f1")


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def _aggregate(rows: Sequence[Mapping]) -> Dict[str, float]:
    dyn_se = sum(float(r.get("dyn_rollout_se_sum") or 0.0) for r in rows)
    dyn_n = sum(int(r.get("dyn_rollout_elem_count") or 0) for r in rows)
    action_se = sum(float(r.get("action_se_sum") or 0.0) for r in rows)
    action_n = sum(int(r.get("action_elem_count") or 0) for r in rows)
    rare_sum = sum(float(r.get("rare_ade_sum") or 0.0) for r in rows)
    rare_n = sum(int(r.get("rare_ade_count") or 0) for r in rows)
    inter_hit = sum(int(r.get("interaction_hit") or 0) for r in rows)
    inter_total = sum(int(r.get("interaction_total") or 0) for r in rows)
    tp = sum(int(r.get("collision_tp") or 0) for r in rows)
    fp = sum(int(r.get("collision_fp") or 0) for r in rows)
    fn = sum(int(r.get("collision_fn") or 0) for r in rows)
    tn = sum(int(r.get("collision_tn") or 0) for r in rows)
    precision, recall, f1 = _f1(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "num_samples": len(rows),
        "dyn_rollout_mse": dyn_se / dyn_n if dyn_n else 0.0,
        "rare_ade": rare_sum / rare_n if rare_n else 0.0,
        "interaction_recall_at_1m": inter_hit / inter_total if inter_total else 0.0,
        "collision_f1": f1,
        "collision_precision": precision,
        "collision_recall": recall,
        "collision_accuracy": (tp + tn) / total if total else 0.0,
        "action_mse": action_se / action_n if action_n else 0.0,
        "interaction_total": inter_total,
        "rare_ade_count": rare_n,
        "collision_tp": tp,
        "collision_fp": fp,
        "collision_fn": fn,
        "collision_tn": tn,
    }


def _group_by(rows: Iterable[Mapping], key: str) -> Dict[str, List[Mapping]]:
    grouped: Dict[str, List[Mapping]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _scene_metrics(rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]]):
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split, by_scene in rows_by_split_scene.items():
        for scene_id, rows in by_scene.items():
            out.setdefault(scene_id, {})[split] = _aggregate(rows)
    return out


def _normalize(values: Mapping[str, float], higher_is_better: bool) -> Dict[str, float]:
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {k: 0.0 for k in values}
    if higher_is_better:
        return {k: (hi - v) / (hi - lo) for k, v in values.items()}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _choose_oracle(
    scene_to_split_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    objective: str,
) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for scene_id, metrics_by_split in scene_to_split_metrics.items():
        if objective == "dyn_rare_intrec":
            score: Dict[str, float] = {split: 0.0 for split in metrics_by_split}
            for metric in ("dyn_rollout_mse", "rare_ade"):
                values = {
                    split: float(metrics[metric])
                    for split, metrics in metrics_by_split.items()
                }
                norm = _normalize(values, higher_is_better=False)
                for split, value in norm.items():
                    score[split] += value
            values = {
                split: float(metrics["interaction_recall_at_1m"])
                for split, metrics in metrics_by_split.items()
            }
            norm = _normalize(values, higher_is_better=True)
            for split, value in norm.items():
                score[split] += value
            choices[scene_id] = min(score, key=score.get)
        elif objective == "collision":
            choices[scene_id] = max(
                metrics_by_split,
                key=lambda split: float(metrics_by_split[split]["collision_f1"]),
            )
        elif objective == "pareto":
            score = {split: 0.0 for split in metrics_by_split}
            for metric in LOWER_IS_BETTER:
                norm = _normalize(
                    {
                        split: float(metrics[metric])
                        for split, metrics in metrics_by_split.items()
                    },
                    higher_is_better=False,
                )
                for split, value in norm.items():
                    score[split] += value
            for metric in HIGHER_IS_BETTER:
                norm = _normalize(
                    {
                        split: float(metrics[metric])
                        for split, metrics in metrics_by_split.items()
                    },
                    higher_is_better=True,
                )
                for split, value in norm.items():
                    score[split] += value
            choices[scene_id] = min(score, key=score.get)
        else:
            raise ValueError(f"Unknown objective: {objective}")
    return choices


def _choose_adaptive_heuristic(
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    splits: Sequence[str],
) -> Dict[str, str]:
    # Scene statistics are data properties, so read them from the first split.
    reference = rows_by_split_scene[splits[0]]
    scene_dyn = {
        scene_id: sum(int(r.get("num_dynamic_tokens") or 0) for r in rows) / len(rows)
        for scene_id, rows in reference.items()
    }
    scene_risk = {
        scene_id: sum(int(r.get("num_risky_relation_tokens") or 0) for r in rows) / len(rows)
        for scene_id, rows in reference.items()
    }
    dyn_thr = median(scene_dyn.values())
    risk_thr = median(scene_risk.values())
    choices: Dict[str, str] = {}
    for scene_id in reference:
        if scene_risk[scene_id] > risk_thr and "10/6" in splits:
            choices[scene_id] = "10/6"
        elif scene_dyn[scene_id] > dyn_thr and "14/2" in splits:
            choices[scene_id] = "14/2"
        elif "12/4" in splits:
            choices[scene_id] = "12/4"
        else:
            choices[scene_id] = splits[0]
    return choices


def _rows_for_policy(
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    choices: Mapping[str, str],
) -> List[Mapping]:
    rows: List[Mapping] = []
    for scene_id, split in choices.items():
        rows.extend(rows_by_split_scene[split][scene_id])
    return rows


def _choice_histogram(choices: Mapping[str, str]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for split in choices.values():
        hist[split] = hist.get(split, 0) + 1
    return dict(sorted(hist.items()))


def _format_table(results: Mapping[str, Mapping[str, float]]) -> str:
    lines = [
        "| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in results.items():
        hist = metrics.get("choice_histogram", {})
        hist_text = ", ".join(f"{k}:{v}" for k, v in hist.items()) if hist else "-"
        lines.append(
            f"| `{name}` | {metrics['dyn_rollout_mse']:.4f} | "
            f"{metrics['rare_ade']:.4f} | "
            f"{metrics['interaction_recall_at_1m']:.4f} | "
            f"{metrics['collision_f1']:.4f} | "
            f"{metrics['action_mse']:.4f} | {hist_text} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing split_* outputs.",
    )
    parser.add_argument(
        "--variant",
        default="object_relation_decoupled",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["14/2", "12/4", "10/6"],
    )
    args = parser.parse_args()

    rows_by_split_scene: Dict[str, Dict[str, List[Mapping]]] = {}
    all_scene_ids = None
    for split in args.splits:
        split_dir = f"split_{split.replace('/', '_')}"
        path = args.experiment_dir / split_dir / args.variant / "per_sample_metrics.jsonl"
        rows = _load_jsonl(path)
        by_scene = _group_by(rows, "scene_id")
        rows_by_split_scene[split] = by_scene
        scene_ids = set(by_scene)
        all_scene_ids = scene_ids if all_scene_ids is None else all_scene_ids & scene_ids

    if not all_scene_ids:
        raise RuntimeError("No common scenes across split outputs.")

    # Restrict all policies to scenes available in every split.
    common = set(all_scene_ids)
    for split in list(rows_by_split_scene):
        rows_by_split_scene[split] = {
            scene_id: rows
            for scene_id, rows in rows_by_split_scene[split].items()
            if scene_id in common
        }

    scene_to_split_metrics = _scene_metrics(rows_by_split_scene)

    results: Dict[str, Dict] = {}
    for split in args.splits:
        rows = [
            row
            for scene_rows in rows_by_split_scene[split].values()
            for row in scene_rows
        ]
        results[f"fixed_{split}"] = _aggregate(rows)

    policies = {
        "oracle_dyn_rare_intrec": _choose_oracle(
            scene_to_split_metrics, "dyn_rare_intrec",
        ),
        "oracle_collision": _choose_oracle(scene_to_split_metrics, "collision"),
        "oracle_pareto": _choose_oracle(scene_to_split_metrics, "pareto"),
        "adaptive_risk_density": _choose_adaptive_heuristic(
            rows_by_split_scene, args.splits,
        ),
    }
    for policy_name, choices in policies.items():
        metrics = _aggregate(_rows_for_policy(rows_by_split_scene, choices))
        metrics["choice_histogram"] = _choice_histogram(choices)
        results[policy_name] = metrics

    output_json = args.experiment_dir / "oracle_adaptive_split_summary.json"
    output_md = args.experiment_dir / "oracle_adaptive_split_summary.md"
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    output_md.write_text(
        "# Stage-0 Oracle / Adaptive Split Summary\n\n"
        f"Common scenes: {len(common)}\n\n"
        + _format_table(results)
        + "\n\n"
        "Oracle policies choose the split per scene and then aggregate the "
        "selected per-sample numerators/denominators. `adaptive_risk_density` "
        "uses only scene token statistics: high risky-relation density -> "
        "`10/6`, high dynamic density -> `14/2`, otherwise `12/4`.\n"
    )
    print(output_md)


if __name__ == "__main__":
    main()
