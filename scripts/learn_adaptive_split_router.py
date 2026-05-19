#!/usr/bin/env python3
"""Learn a tiny deployable router for adaptive DOOR split selection.

This is deliberately not a new model. It learns a two-threshold rule from
scene-level token statistics:

    if risky_relation_density >= threshold: choose A
    elif dynamic_density >= threshold: choose B
    else: choose C

The rule is trained on a subset of validation scenes to minimize the same
per-scene Pareto cost used by the oracle summary, then evaluated on held-out
scenes by aggregating the original per-sample metric numerators/denominators.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


LOWER_IS_BETTER = ("dyn_rollout_mse", "rare_ade")
HIGHER_IS_BETTER = ("interaction_recall_at_1m", "collision_f1")


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _group_by(rows: Iterable[Mapping], key: str) -> Dict[str, List[Mapping]]:
    grouped: Dict[str, List[Mapping]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


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


def _pareto_cost_by_scene(
    scene_to_split_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    costs: Dict[str, Dict[str, float]] = {}
    for scene_id, metrics_by_split in scene_to_split_metrics.items():
        score = {split: 0.0 for split in metrics_by_split}
        for metric in LOWER_IS_BETTER:
            norm = _normalize(
                {split: float(metrics[metric]) for split, metrics in metrics_by_split.items()},
                higher_is_better=False,
            )
            for split, value in norm.items():
                score[split] += value
        for metric in HIGHER_IS_BETTER:
            norm = _normalize(
                {split: float(metrics[metric]) for split, metrics in metrics_by_split.items()},
                higher_is_better=True,
            )
            for split, value in norm.items():
                score[split] += value
        costs[scene_id] = score
    return costs


def _oracle_choices(costs: Mapping[str, Mapping[str, float]]) -> Dict[str, str]:
    return {scene_id: min(split_cost, key=split_cost.get) for scene_id, split_cost in costs.items()}


def _scene_features(reference_by_scene: Mapping[str, Sequence[Mapping]]) -> Dict[str, Dict[str, float]]:
    features: Dict[str, Dict[str, float]] = {}
    for scene_id, rows in reference_by_scene.items():
        n = max(1, len(rows))
        dyn = sum(int(r.get("num_dynamic_tokens") or 0) for r in rows) / n
        rel = sum(int(r.get("num_relation_tokens") or 0) for r in rows) / n
        risky = sum(int(r.get("num_risky_relation_tokens") or 0) for r in rows) / n
        valid = sum(int(r.get("num_valid_tokens") or 0) for r in rows) / n
        features[scene_id] = {
            "dynamic_density": dyn / max(valid, 1.0),
            "relation_density": rel / max(valid, 1.0),
            "risky_relation_density": risky / max(rel, 1.0),
            "num_dynamic_tokens": dyn,
            "num_relation_tokens": rel,
            "num_risky_relation_tokens": risky,
            "num_valid_tokens": valid,
        }
    return features


def _quantile_grid(values: Sequence[float], max_points: int = 15) -> List[float]:
    vals = sorted(set(float(v) for v in values))
    if not vals:
        return [0.0]
    if len(vals) <= max_points:
        return vals
    return [
        vals[round(i * (len(vals) - 1) / (max_points - 1))]
        for i in range(max_points)
    ]


def _choose_with_rule(
    feature: Mapping[str, float],
    rule: Mapping[str, object],
) -> str:
    risk_high = float(feature["risky_relation_density"]) >= float(rule["risk_threshold"])
    dyn_high = float(feature["dynamic_density"]) >= float(rule["dynamic_threshold"])
    if rule["order"] == "risk_first":
        if risk_high:
            return str(rule["risk_split"])
        if dyn_high:
            return str(rule["dynamic_split"])
    else:
        if dyn_high:
            return str(rule["dynamic_split"])
        if risk_high:
            return str(rule["risk_split"])
    return str(rule["default_split"])


def _policy_rows(
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    choices: Mapping[str, str],
) -> List[Mapping]:
    rows: List[Mapping] = []
    for scene_id, split in choices.items():
        rows.extend(rows_by_split_scene[split][scene_id])
    return rows


def _hist(choices: Mapping[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for split in choices.values():
        out[split] = out.get(split, 0) + 1
    return dict(sorted(out.items()))


def _rule_cost(
    scenes: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    costs: Mapping[str, Mapping[str, float]],
    rule: Mapping[str, object],
) -> float:
    return sum(costs[scene][_choose_with_rule(features[scene], rule)] for scene in scenes) / max(1, len(scenes))


def _learn_rule(
    train_scenes: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    costs: Mapping[str, Mapping[str, float]],
    splits: Sequence[str],
) -> Dict[str, object]:
    risk_grid = _quantile_grid([features[s]["risky_relation_density"] for s in train_scenes])
    dyn_grid = _quantile_grid([features[s]["dynamic_density"] for s in train_scenes])
    best_rule: Dict[str, object] | None = None
    best_cost = float("inf")
    for order in ("risk_first", "dynamic_first"):
        for risk_thr in risk_grid:
            for dyn_thr in dyn_grid:
                for risk_split in splits:
                    for dyn_split in splits:
                        for default_split in splits:
                            rule = {
                                "order": order,
                                "risk_threshold": risk_thr,
                                "dynamic_threshold": dyn_thr,
                                "risk_split": risk_split,
                                "dynamic_split": dyn_split,
                                "default_split": default_split,
                            }
                            cost = _rule_cost(train_scenes, features, costs, rule)
                            if cost < best_cost:
                                best_cost = cost
                                best_rule = rule
    assert best_rule is not None
    best_rule["train_pareto_cost"] = best_cost
    return best_rule


def _evaluate_rule(
    scenes: Sequence[str],
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    features: Mapping[str, Mapping[str, float]],
    rule: Mapping[str, object],
) -> Dict[str, object]:
    choices = {scene: _choose_with_rule(features[scene], rule) for scene in scenes}
    metrics = _aggregate(_policy_rows(rows_by_split_scene, choices))
    metrics["choice_histogram"] = _hist(choices)
    return metrics


def _evaluate_fixed(
    scenes: Sequence[str],
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    split: str,
) -> Dict[str, object]:
    choices = {scene: split for scene in scenes}
    metrics = _aggregate(_policy_rows(rows_by_split_scene, choices))
    metrics["choice_histogram"] = _hist(choices)
    return metrics


def _evaluate_oracle(
    scenes: Sequence[str],
    rows_by_split_scene: Mapping[str, Mapping[str, Sequence[Mapping]]],
    oracle_choices: Mapping[str, str],
) -> Dict[str, object]:
    choices = {scene: oracle_choices[scene] for scene in scenes}
    metrics = _aggregate(_policy_rows(rows_by_split_scene, choices))
    metrics["choice_histogram"] = _hist(choices)
    return metrics


def _format_results(results: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "| Policy | Dyn MSE ↓ | Rare ADE ↓ | IntRec@1m ↑ | Coll F1 ↑ | Action MSE ↓ | Scenes/Split |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in results.items():
        hist = metrics.get("choice_histogram", {})
        hist_text = ", ".join(f"{k}:{v}" for k, v in hist.items()) if hist else "-"
        lines.append(
            f"| `{name}` | {float(metrics['dyn_rollout_mse']):.4f} | "
            f"{float(metrics['rare_ade']):.4f} | "
            f"{float(metrics['interaction_recall_at_1m']):.4f} | "
            f"{float(metrics['collision_f1']):.4f} | "
            f"{float(metrics['action_mse']):.4f} | {hist_text} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--variant", default="object_relation_decoupled")
    parser.add_argument("--splits", nargs="+", default=["14/2", "12/4", "10/6"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()

    rows_by_split_scene: Dict[str, Dict[str, List[Mapping]]] = {}
    common_scenes = None
    for split in args.splits:
        path = (
            args.experiment_dir
            / f"split_{split.replace('/', '_')}"
            / args.variant
            / "per_sample_metrics.jsonl"
        )
        by_scene = _group_by(_load_jsonl(path), "scene_id")
        rows_by_split_scene[split] = by_scene
        scenes = set(by_scene)
        common_scenes = scenes if common_scenes is None else common_scenes & scenes

    if not common_scenes:
        raise RuntimeError("No common scenes across split outputs.")
    common = sorted(common_scenes)
    for split in list(rows_by_split_scene):
        rows_by_split_scene[split] = {
            scene: rows_by_split_scene[split][scene] for scene in common
        }

    rng = random.Random(args.seed)
    shuffled = list(common)
    rng.shuffle(shuffled)
    n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * args.train_ratio))))
    train_scenes = sorted(shuffled[:n_train])
    test_scenes = sorted(shuffled[n_train:])

    scene_to_split_metrics = _scene_metrics(rows_by_split_scene)
    costs = _pareto_cost_by_scene(scene_to_split_metrics)
    oracle_choices = _oracle_choices(costs)
    features = _scene_features(rows_by_split_scene[args.splits[0]])
    rule = _learn_rule(train_scenes, features, costs, args.splits)

    train_results: Dict[str, Dict[str, object]] = {}
    test_results: Dict[str, Dict[str, object]] = {}
    for split in args.splits:
        train_results[f"fixed_{split}"] = _evaluate_fixed(train_scenes, rows_by_split_scene, split)
        test_results[f"fixed_{split}"] = _evaluate_fixed(test_scenes, rows_by_split_scene, split)
    train_results["oracle_pareto"] = _evaluate_oracle(train_scenes, rows_by_split_scene, oracle_choices)
    test_results["oracle_pareto"] = _evaluate_oracle(test_scenes, rows_by_split_scene, oracle_choices)
    train_results["learned_router"] = _evaluate_rule(train_scenes, rows_by_split_scene, features, rule)
    test_results["learned_router"] = _evaluate_rule(test_scenes, rows_by_split_scene, features, rule)

    output = {
        "setup": {
            "experiment_dir": str(args.experiment_dir),
            "splits": args.splits,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "n_common_scenes": len(common),
            "n_train_scenes": len(train_scenes),
            "n_test_scenes": len(test_scenes),
        },
        "learned_rule": rule,
        "train_results": train_results,
        "test_results": test_results,
    }
    out_json = args.experiment_dir / "adaptive_split_learned_router_summary.json"
    out_md = args.experiment_dir / "adaptive_split_learned_router_summary.md"
    out_json.write_text(json.dumps(output, indent=2, sort_keys=True))

    rule_text = (
        f"order=`{rule['order']}`, "
        f"risk_threshold={float(rule['risk_threshold']):.4f}, "
        f"dynamic_threshold={float(rule['dynamic_threshold']):.4f}, "
        f"risk_split=`{rule['risk_split']}`, "
        f"dynamic_split=`{rule['dynamic_split']}`, "
        f"default_split=`{rule['default_split']}`"
    )
    out_md.write_text(
        "# Adaptive Split Learned Router\n\n"
        "This is a deployable proxy router trained only on scene token statistics, "
        "not per-split outcome metrics at test time.\n\n"
        f"Common scenes: {len(common)}; train scenes: {len(train_scenes)}; "
        f"test scenes: {len(test_scenes)}.\n\n"
        f"Learned rule: {rule_text}.\n\n"
        "## Train\n\n"
        + _format_results(train_results)
        + "\n\n## Held-Out Test\n\n"
        + _format_results(test_results)
        + "\n\nReading: `oracle_pareto` is an upper bound that chooses the best split "
        "per scene using outcome metrics. `learned_router` uses only deployable "
        "scene statistics (`risky_relation_density`, `dynamic_density`) and the "
        "learned thresholds above.\n"
    )
    print(out_md)


if __name__ == "__main__":
    main()
