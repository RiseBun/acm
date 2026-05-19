"""Aggregate nuPlan official closed-loop P0 runs and paired bootstrap tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

KEY_METRICS = [
    "score",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "ego_is_comfortable",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "driving_direction_compliance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="nuPlan run dirs containing aggregator_metric/")
    parser.add_argument("--name", default="nuplan_official_closed_loop_p0")
    parser.add_argument("--primary", default="doorrl_wm_decoupled_no_vis")
    parser.add_argument(
        "--comparators",
        nargs="*",
        default=["doorrl_wm_object", "doorrl_bc", "PDMClosedPlanner", "IDMPlanner"],
    )
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--paper-assets-data",
        default=str(ROOT / "期刊" / "paper_assets" / "data"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "nuplan_official_closed_loop_p0" / "summary"),
    )
    return parser.parse_args()


def _read_aggregator(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "aggregator_metric").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No aggregator parquet under {run_dir / 'aggregator_metric'}")
    df = pd.read_parquet(files[-1])
    df["run_dir"] = str(run_dir)
    df["aggregator_metric_path"] = str(files[-1])
    return df


def _read_runner(run_dir: Path) -> list[dict[str, Any]]:
    runner_path = run_dir / "runner_report.parquet"
    if not runner_path.exists():
        return []
    return pd.read_parquet(runner_path).to_dict(orient="records")


def _scenario_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario" not in df.columns:
        return df.iloc[0:0].copy()
    rows = df[df["scenario"].astype(str) != "final_score"].copy()
    if rows.empty:
        # Some nuPlan aggregator configs only write final rows. Keep summary usable
        # but paired tests will be marked unavailable.
        return rows
    return rows


def _metric_mean(df: pd.DataFrame, planner: str, metric: str) -> float | None:
    rows = df[(df["planner_name"] == planner) & (df["scenario"].astype(str) == "final_score")]
    if rows.empty:
        rows = df[df["planner_name"] == planner]
    if metric not in rows.columns:
        return None
    vals = pd.to_numeric(rows[metric], errors="coerce").dropna()
    return None if vals.empty else float(vals.mean())


def _planner_summary(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    planners = sorted(str(x) for x in df["planner_name"].dropna().unique())
    return {
        planner: {metric: _metric_mean(df, planner, metric) for metric in KEY_METRICS}
        for planner in planners
    }


def _paired_bootstrap(
    scenario_df: pd.DataFrame,
    primary: str,
    comparator: str,
    metric: str,
    iters: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if metric not in scenario_df.columns:
        return {"available": False, "reason": f"metric {metric} missing"}
    pivot = scenario_df.pivot_table(index="scenario", columns="planner_name", values=metric, aggfunc="mean")
    if primary not in pivot.columns or comparator not in pivot.columns:
        return {"available": False, "reason": "planner missing from scenario-level rows"}
    paired = pivot[[primary, comparator]].dropna()
    if paired.empty:
        return {"available": False, "reason": "no paired scenario-level rows"}
    delta = paired[primary].to_numpy(dtype=float) - paired[comparator].to_numpy(dtype=float)
    idx = rng.integers(0, len(delta), size=(iters, len(delta)))
    samples = delta[idx].mean(axis=1)
    return {
        "available": True,
        "n_paired_scenarios": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "prob_delta_gt_0": float((samples > 0).mean()),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['name']}",
        "",
        payload["note"],
        "",
        "| planner | score | no-coll | drivable | progress | progress ratio | comfort | TTC | speed | direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for planner, metrics in payload["summary"].items():
        def fmt(metric: str) -> str:
            value = metrics.get(metric)
            return "NA" if value is None else f"{value:.3f}"

        lines.append(
            f"| {planner} | {fmt('score')} | {fmt('no_ego_at_fault_collisions')} | "
            f"{fmt('drivable_area_compliance')} | {fmt('ego_is_making_progress')} | "
            f"{fmt('ego_progress_along_expert_route')} | {fmt('ego_is_comfortable')} | "
            f"{fmt('time_to_collision_within_bound')} | {fmt('speed_limit_compliance')} | "
            f"{fmt('driving_direction_compliance')} |"
        )
    lines.extend(["", "## Paired Bootstrap", ""])
    for key, result in payload["paired_bootstrap"].items():
        if not result.get("available"):
            lines.append(f"- {key}: unavailable ({result.get('reason')})")
            continue
        lo, hi = result["ci95"]
        lines.append(
            f"- {key}: delta={result['mean_delta']:.4f}, "
            f"95% CI [{lo:.4f}, {hi:.4f}], P(delta>0)={result['prob_delta_gt_0']:.3f}, "
            f"n={result['n_paired_scenarios']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_assets_data)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    dfs = [_read_aggregator(Path(run_dir)) for run_dir in args.run_dirs]
    df = pd.concat(dfs, ignore_index=True)
    scenario_df = _scenario_rows(df)
    rng = np.random.default_rng(args.seed)
    paired = {}
    for comparator in args.comparators:
        for metric in ["score", "no_ego_at_fault_collisions", "time_to_collision_within_bound", "ego_progress_along_expert_route"]:
            key = f"{args.primary}__minus__{comparator}__{metric}"
            paired[key] = _paired_bootstrap(
                scenario_df, args.primary, comparator, metric, args.bootstrap_iters, rng
            )

    payload = {
        "name": args.name,
        "date": "2026-04-28",
        "note": "nuPlan official closed-loop P0 aggregate; paired bootstrap uses scenario-level rows when available.",
        "run_dirs": args.run_dirs,
        "aggregator_metric_paths": sorted(df["aggregator_metric_path"].dropna().unique().tolist()),
        "summary": _planner_summary(df),
        "paired_bootstrap": paired,
        "runner_report": [_to_jsonable(row) for run_dir in args.run_dirs for row in _read_runner(Path(run_dir))],
    }
    json_text = json.dumps(payload, indent=2, default=_to_jsonable)
    (out_dir / f"{args.name}.json").write_text(json_text)
    (paper_dir / f"{args.name}.json").write_text(json_text)
    _write_markdown(out_dir / f"{args.name}.md", payload)
    _write_markdown(paper_dir / f"{args.name}.md", payload)
    print(f"wrote {out_dir / f'{args.name}.json'}")
    print(f"wrote {paper_dir / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
