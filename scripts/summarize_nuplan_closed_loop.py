"""Summarize nuPlan closed-loop MVP parquet outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


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
    parser.add_argument("run_dir", help="nuPlan run directory containing aggregator_metric/")
    return parser.parse_args()


def _read_aggregator(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "aggregator_metric").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No aggregator parquet under {run_dir / 'aggregator_metric'}")
    return pd.read_parquet(files[-1])


def _metric_mean(df: pd.DataFrame, planner: str, metric: str) -> float | None:
    rows = df[(df["planner_name"] == planner) & (df["scenario"] == "final_score")]
    if rows.empty:
        rows = df[df["planner_name"] == planner]
    if metric not in rows.columns:
        return None
    vals = pd.to_numeric(rows[metric], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def summarize(run_dir: Path) -> Dict:
    agg = _read_aggregator(run_dir)
    planners = sorted(str(x) for x in agg["planner_name"].dropna().unique())
    summary: Dict[str, Dict] = {}
    for planner in planners:
        block = {}
        for metric in KEY_METRICS:
            block[metric] = _metric_mean(agg, planner, metric)
        summary[planner] = block

    runner_path = run_dir / "runner_report.parquet"
    runners = []
    if runner_path.exists():
        runner_df = pd.read_parquet(runner_path)
        runners = runner_df.to_dict(orient="records")

    return {
        "run_dir": str(run_dir),
        "n_planners": len(planners),
        "planners": planners,
        "summary": summary,
        "runner_report": runners,
        "note": "nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.",
    }


def write_markdown(run_dir: Path, payload: Dict) -> None:
    lines = [
        "# nuPlan Closed-Loop MVP Summary",
        "",
        payload["note"],
        "",
        "| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for planner, metrics in payload["summary"].items():
        def fmt(name: str) -> str:
            value = metrics.get(name)
            return "NA" if value is None else f"{value:.3f}"

        lines.append(
            f"| {planner} | {fmt('score')} | "
            f"{fmt('no_ego_at_fault_collisions')} | "
            f"{fmt('drivable_area_compliance')} | "
            f"{fmt('ego_is_making_progress')} | "
            f"{fmt('ego_progress_along_expert_route')} | "
            f"{fmt('ego_is_comfortable')} | "
            f"{fmt('time_to_collision_within_bound')} | "
            f"{fmt('speed_limit_compliance')} | "
            f"{fmt('driving_direction_compliance')} |"
        )
    lines.extend([
        "",
        "## Runner Report",
        "",
        "| planner | succeeded | scenario | runtime mean | duration |",
        "|---|---:|---|---:|---:|",
    ])
    for row in payload.get("runner_report", []):
        lines.append(
            f"| {row.get('planner_name')} | {row.get('succeeded')} | "
            f"{row.get('scenario_name')} | "
            f"{float(row.get('compute_trajectory_runtimes_mean', 0.0)):.4f} | "
            f"{float(row.get('duration', 0.0)):.1f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    payload = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    write_markdown(run_dir, payload)
    print(f"wrote {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
