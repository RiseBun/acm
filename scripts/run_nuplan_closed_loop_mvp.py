"""Run a small nuPlan closed-loop MVP with DOOR-RL planners.

This script is a thin bridge around the official nuPlan devkit. It expects the
local devkit checkout under ``cangku/nuplan-devkit`` plus official nuPlan DB/map
paths. If dependencies or data are missing, it fails early with actionable
messages rather than silently falling back to offline evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuplan-devkit-root",
        default=str(ROOT / "cangku" / "nuplan-devkit"),
    )
    parser.add_argument(
        "--nuplan-data-root",
        default="/mnt/datasets/e2e-nuplan/20260302/val",
        help="Folder containing official nuPlan .db files for the subset to simulate.",
    )
    parser.add_argument(
        "--nuplan-maps-root",
        default="/mnt/datasets/e2e-nuplan/20260302/maps",
    )
    parser.add_argument(
        "--nuplan-sensor-root",
        default="/mnt/datasets/e2e-nuplan/20260302/original/sensor_blobs",
    )
    parser.add_argument(
        "--db-files",
        nargs="*",
        default=None,
        help="Optional explicit nuPlan .db files. Defaults to the first non-empty DB under --nuplan-data-root.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument(
        "--stage1-root",
        default=str(ROOT / "experiments" / "nuplan_stage1_50k"),
    )
    parser.add_argument(
        "--bc-stage1-root",
        default=str(ROOT / "experiments" / "nuplan_bc_baseline_50k"),
        help="Stage-1 root for the BC / planner-imitation baseline.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["wm_object", "wm_decoupled_no_vis"],
        choices=["wm_object", "wm_decoupled_no_vis", "bc"],
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        choices=["pdm", "idm"],
        help="Additional non-DOOR planners to run in the same official wrapper.",
    )
    parser.add_argument("--scenario-builder", default="nuplan")
    parser.add_argument("--scenario-filter", default="closed_loop_few_training_scenario")
    parser.add_argument(
        "--scenario-tokens-json",
        default=None,
        help="Optional fixed scenario-token JSON from prepare_nuplan_closed_loop_subset.py.",
    )
    parser.add_argument("--limit-total-scenarios", type=int, default=50)
    parser.add_argument(
        "--worker",
        default="single_machine_thread_pool",
        help="nuPlan worker config. Thread pool uses the strong local CPU for larger subsets.",
    )
    parser.add_argument("--worker-max-workers", type=int, default=None)
    parser.add_argument("--observation", default="box_observation")
    parser.add_argument("--ego-controller", default="perfect_tracking_controller")
    parser.add_argument(
        "--simulation-metric",
        default="simulation_closed_loop_nonreactive_agents",
        help="Use simulation_closed_loop_reactive_agents after IDM observation is configured.",
    )
    parser.add_argument(
        "--metric-aggregator",
        default="closed_loop_nonreactive_agents_weighted_average",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "nuplan_closed_loop_mvp"),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="nuPlan job name. If omitted, includes the challenge name so metrics aggregate.",
    )
    parser.add_argument("--horizon-seconds", type=float, default=8.0)
    parser.add_argument("--sampling-time", type=float, default=0.25)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=2.0,
        help="Scale policy action[0] before adding it to current ego speed.",
    )
    parser.add_argument(
        "--yaw-rate-scale",
        type=float,
        default=0.6,
        help="Scale policy action[1] into yaw-rate for trajectory rollout.",
    )
    parser.add_argument("--max-accel", type=float, default=2.5)
    parser.add_argument("--max-decel", type=float, default=4.0)
    parser.add_argument(
        "--disable-safety-projection",
        action="store_true",
        help="Use nominal action rollout without collision/TTC/drivable candidate scoring.",
    )
    parser.add_argument("--collision-weight", type=float, default=80.0)
    parser.add_argument("--ttc-weight", type=float, default=25.0)
    parser.add_argument("--drivable-weight", type=float, default=60.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=12.0)
    parser.add_argument("--lane-center-weight", type=float, default=8.0)
    parser.add_argument("--max-comfortable-decel", type=float, default=2.0)
    parser.add_argument("--max-comfortable-jerk", type=float, default=2.5)
    parser.add_argument(
        "--min-progress-speed",
        type=float,
        default=2.5,
        help="Minimum target speed candidate used when safety scoring permits progress.",
    )
    parser.add_argument(
        "--disable-corridor-projection",
        action="store_true",
        help="Disable route/lane corridor projection and use free-space kinematic rollout.",
    )
    parser.add_argument(
        "--corridor-candidate-limit",
        type=int,
        default=3,
        help="Number of sorted lane/route corridor candidates evaluated per planner step.",
    )
    parser.add_argument(
        "--enable-lead-vehicle-controller",
        action="store_true",
        help="Enable smoother lead-vehicle following speed caps on lane corridors.",
    )
    parser.add_argument("--lead-time-headway", type=float, default=2.0)
    parser.add_argument("--lead-min-gap", type=float, default=8.0)
    parser.add_argument("--lead-max-speed-drop", type=float, default=2.0)
    parser.add_argument("--lead-ttc-threshold", type=float, default=4.0)
    parser.add_argument("--lead-gap-margin", type=float, default=0.0)
    parser.add_argument(
        "--enable-ttc-proxy",
        action="store_true",
        help="Penalize near-future low-clearance candidates before collision overlap.",
    )
    parser.add_argument("--ttc-proxy-clearance", type=float, default=5.0)
    parser.add_argument("--ttc-proxy-horizon", type=float, default=4.0)
    parser.add_argument("--pdm-map-radius", type=float, default=100.0)
    parser.add_argument("--pdm-fallback-target-velocity", type=float, default=15.0)
    parser.add_argument("--pdm-lateral-offsets", type=float, nargs="*", default=[-1.0, 1.0])
    return parser.parse_args()


def _require_path(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _resolve_db_files(args: argparse.Namespace) -> list[str]:
    if args.db_files:
        files = [_require_path(p, "nuPlan db file") for p in args.db_files]
    else:
        root = Path(args.nuplan_data_root)
        files = [p for p in sorted(root.glob("*.db")) if p.stat().st_size > 0]
        if not files:
            raise FileNotFoundError(f"No non-empty .db files under {root}")
        files = files[:1]
    return [str(p) for p in files]


def _load_scenario_subset(path: str | None, limit: int | None) -> tuple[list[str] | None, list[str] | None]:
    if path is None:
        return None, None
    payload = json.loads(Path(path).read_text())
    rows = payload.get("scenarios", payload.get("tokens", payload.get("scenario_tokens"))) if isinstance(payload, dict) else payload
    if rows is None:
        raise ValueError(f"No scenario tokens found in {path}")
    tokens: list[str] = []
    for row in rows:
        if isinstance(row, str):
            token = row
        elif isinstance(row, dict):
            token = row.get("token") or row.get("scenario_token")
        elif isinstance(row, (list, tuple)) and row:
            token = row[-1]
        else:
            token = None
        if not token:
            raise ValueError(f"Unsupported scenario-token entry in {path}: {row!r}")
        tokens.append(str(token))
    db_files = payload.get("db_files") if isinstance(payload, dict) else None
    return (tokens[:limit] if limit and limit > 0 else tokens), db_files


def _door_checkpoint(args: argparse.Namespace, condition: str) -> Path:
    root = Path(args.bc_stage1_root if condition == "bc" else args.stage1_root)
    ckpt = root / f"seed{args.seed}" / condition / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found for {condition}: {ckpt}")
    return ckpt


def _build_idm_planner(devkit_root: Path):
    sys.path.insert(0, str(devkit_root))
    from nuplan.planning.simulation.planner.idm_planner import IDMPlanner

    return IDMPlanner(
        target_velocity=10.0,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=3.0,
        planned_trajectory_samples=16,
        planned_trajectory_sample_interval=0.5,
        occupancy_map_radius=40.0,
    )


def _build_pdm_planner(args: argparse.Namespace, devkit_root: Path):
    navsim_root = ROOT / "cangku" / "navsim"
    sys.path.insert(0, str(devkit_root))
    sys.path.insert(0, str(navsim_root))
    try:
        from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
        from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    except Exception as exc:
        raise RuntimeError(
            "PDM baseline requested but local navsim PDM dependencies are not importable. "
            "Install navsim/nuPlan-compatible dependencies or run with --baselines idm."
        ) from exc

    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=args.sampling_time)
    future_sampling = TrajectorySampling(
        num_poses=max(1, int(round(args.horizon_seconds / args.sampling_time))),
        interval_length=args.sampling_time,
    )
    return PDMClosedPlanner(
        trajectory_sampling=future_sampling,
        proposal_sampling=proposal_sampling,
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
            fallback_target_velocity=args.pdm_fallback_target_velocity,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
        ),
        lateral_offsets=args.pdm_lateral_offsets,
        map_radius=args.pdm_map_radius,
    )


def main() -> None:
    args = parse_args()
    devkit_root = _require_path(args.nuplan_devkit_root, "nuPlan devkit root")
    _require_path(args.nuplan_data_root, "nuPlan data root")
    _require_path(args.nuplan_maps_root, "nuPlan maps root")

    sys.path.insert(0, str(devkit_root))
    os.environ.setdefault("NUPLAN_DATA_ROOT", str(Path(args.nuplan_data_root).parent))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", args.nuplan_maps_root)
    os.environ.setdefault("NUPLAN_EXP_ROOT", args.output_dir)

    try:
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from nuplan.planning.script.run_simulation import run_simulation
    except Exception as exc:
        raise RuntimeError(
            "nuPlan devkit dependencies are not importable. Install the local "
            "`cangku/nuplan-devkit/requirements.txt` into the active environment "
            "or run this script inside a nuPlan-compatible env."
        ) from exc

    from doorrl.closed_loop.nuplan_oracle_planner import DoorRLNuPlanPlanner

    scenario_tokens, subset_db_files = _load_scenario_subset(args.scenario_tokens_json, args.limit_total_scenarios)
    db_files = subset_db_files or _resolve_db_files(args)
    db_files_override = json.dumps(db_files, separators=(",", ":"))
    challenge_name = (
        "closed_loop_nonreactive_agents"
        if "nonreactive" in args.simulation_metric
        else "closed_loop_reactive_agents"
    )
    job_name = args.job_name or f"seed{args.seed}_{challenge_name}"

    planners = []
    for condition in args.conditions:
        ckpt = _door_checkpoint(args, condition)
        planners.append(
            DoorRLNuPlanPlanner(
                config_path=args.config,
                checkpoint_path=ckpt,
                condition=condition,
                horizon_seconds=args.horizon_seconds,
                sampling_time=args.sampling_time,
                speed_scale=args.speed_scale,
                yaw_rate_scale=args.yaw_rate_scale,
                max_accel=args.max_accel,
                max_decel=args.max_decel,
                safety_projection=not args.disable_safety_projection,
                collision_weight=args.collision_weight,
                ttc_weight=args.ttc_weight,
                drivable_weight=args.drivable_weight,
                progress_weight=args.progress_weight,
                smoothness_weight=args.smoothness_weight,
                lane_center_weight=args.lane_center_weight,
                max_comfortable_decel=args.max_comfortable_decel,
                max_comfortable_jerk=args.max_comfortable_jerk,
                min_progress_speed=args.min_progress_speed,
                corridor_projection=not args.disable_corridor_projection,
                corridor_candidate_limit=args.corridor_candidate_limit,
                lead_vehicle_controller=args.enable_lead_vehicle_controller,
                lead_time_headway=args.lead_time_headway,
                lead_min_gap=args.lead_min_gap,
                lead_max_speed_drop=args.lead_max_speed_drop,
                lead_ttc_threshold=args.lead_ttc_threshold,
                lead_gap_margin=args.lead_gap_margin,
                ttc_proxy=args.enable_ttc_proxy,
                ttc_proxy_clearance=args.ttc_proxy_clearance,
                ttc_proxy_horizon=args.ttc_proxy_horizon,
            )
        )
    for baseline in args.baselines:
        if baseline == "pdm":
            planners.append(_build_pdm_planner(args, devkit_root))
        elif baseline == "idm":
            planners.append(_build_idm_planner(devkit_root))

    config_dir = devkit_root / "nuplan" / "planning" / "script" / "config" / "simulation"
    GlobalHydra.instance().clear()
    try:
        hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    except TypeError:
        hydra.initialize_config_dir(config_dir=str(config_dir))
    overrides = [
        f"group={args.output_dir}",
        "experiment_name=doorrl_closed_loop_mvp",
        f"job_name={job_name}",
        "experiment=${experiment_name}/${job_name}",
        f"worker={args.worker}",
        f"scenario_builder={args.scenario_builder}",
        f"scenario_builder.data_root={args.nuplan_data_root}",
        f"scenario_builder.map_root={args.nuplan_maps_root}",
        f"scenario_builder.sensor_root={args.nuplan_sensor_root}",
        f"scenario_builder.db_files={db_files_override}",
        f"scenario_filter={args.scenario_filter}",
        f"ego_controller={args.ego_controller}",
        f"observation={args.observation}",
        f"simulation_metric={args.simulation_metric}",
        f"metric_aggregator={args.metric_aggregator}",
        "output_dir=${group}/${experiment}",
    ]
    if scenario_tokens:
        overrides.extend(
            [
                "scenario_filter.scenario_tokens="
                + json.dumps(scenario_tokens, separators=(",", ":")),
                "scenario_filter.limit_total_scenarios=null",
                "scenario_filter.shuffle=false",
            ]
        )
    else:
        overrides.append(f"scenario_filter.limit_total_scenarios={args.limit_total_scenarios}")
    if args.worker_max_workers is not None:
        overrides.append(f"worker.max_workers={args.worker_max_workers}")
    cfg = hydra.compose(
        config_name="default_simulation",
        overrides=overrides,
    )
    run_simulation(cfg=cfg, planners=planners)


if __name__ == "__main__":
    main()
