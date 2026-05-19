"""Create fixed nuPlan scenario-token subsets for official closed-loop sweeps."""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuplan-devkit-root", default=str(ROOT / "cangku" / "nuplan-devkit"))
    parser.add_argument("--nuplan-data-root", default="/mnt/datasets/e2e-nuplan/20260302/val")
    parser.add_argument("--nuplan-maps-root", default="/mnt/datasets/e2e-nuplan/20260302/maps")
    parser.add_argument("--nuplan-sensor-root", default="/mnt/datasets/e2e-nuplan/20260302/original/sensor_blobs")
    parser.add_argument("--db-files", nargs="*", default=None)
    parser.add_argument("--scenario-builder", default="nuplan")
    parser.add_argument("--scenario-filter", default="closed_loop_few_training_scenario")
    parser.add_argument("--pool-size", type=int, default=500)
    parser.add_argument(
        "--direct-db-scan",
        action="store_true",
        help="Read lidar_pc/scenario_tag tokens directly from sqlite DBs instead of constructing devkit scenarios.",
    )
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "nuplan_official_closed_loop_p0" / "scenario_subsets"),
    )
    return parser.parse_args()


def _resolve_db_files(args: argparse.Namespace) -> list[str]:
    if args.db_files:
        return [str(Path(p)) for p in args.db_files]
    root = Path(args.nuplan_data_root)
    files = [p for p in sorted(root.glob("*.db")) if p.stat().st_size > 0]
    if not files:
        raise FileNotFoundError(f"No non-empty .db files under {root}")
    return [str(p) for p in files]


def _token_hex(value: bytes) -> str:
    return bytes(value).hex()


def _scan_db_tokens(db_files: list[str], pool_size: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for db_file in db_files:
        log_name = Path(db_file).stem
        con = sqlite3.connect(db_file)
        try:
            query = """
                SELECT lp.token, COALESCE(st.type, 'unknown') AS scenario_type
                FROM lidar_pc AS lp
                LEFT OUTER JOIN scenario_tag AS st ON lp.token = st.lidar_pc_token
                WHERE lp.token IS NOT NULL
                ORDER BY lp.timestamp
                LIMIT 1 OFFSET 10
            """
            found = list(con.execute(query))
            if not found:
                query = """
                    SELECT token, 'unknown'
                    FROM lidar_pc
                    WHERE token IS NOT NULL
                    ORDER BY timestamp
                    LIMIT 1
                """
                found = list(con.execute(query))
            for token_blob, scenario_type in found:
                token = _token_hex(token_blob)
                if token in seen:
                    continue
                seen.add(token)
                rows.append(
                    {
                        "index": len(rows),
                        "token": token,
                        "log_name": log_name,
                        "db_file": db_file,
                        "scenario_name": token,
                        "scenario_type": str(scenario_type),
                    }
                )
                if len(rows) >= pool_size:
                    return rows
        finally:
            con.close()
    return rows


def main() -> None:
    args = parse_args()
    db_files = _resolve_db_files(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.direct_db_scan:
        rows = _scan_db_tokens(db_files, max(args.subset_sizes))
        scenario_filter_name = "direct_db_scan"
    else:
        rows = []
        scenario_filter_name = args.scenario_filter
    devkit_root = Path(args.nuplan_devkit_root)
    sys.path.insert(0, str(devkit_root))
    os.environ.setdefault("NUPLAN_DATA_ROOT", str(Path(args.nuplan_data_root).parent))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", args.nuplan_maps_root)

    if not args.direct_db_scan:
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
        from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
        from nuplan.planning.script.builders.worker_pool_builder import build_worker

        db_files_override = json.dumps(db_files, separators=(",", ":"))
        config_dir = devkit_root / "nuplan" / "planning" / "script" / "config" / "simulation"
        GlobalHydra.instance().clear()
        try:
            hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
        except TypeError:
            hydra.initialize_config_dir(config_dir=str(config_dir))
        cfg = hydra.compose(
            config_name="default_simulation",
            overrides=[
                f"scenario_builder={args.scenario_builder}",
                f"scenario_builder.data_root={args.nuplan_data_root}",
                f"scenario_builder.map_root={args.nuplan_maps_root}",
                f"scenario_builder.sensor_root={args.nuplan_sensor_root}",
                f"scenario_builder.db_files={db_files_override}",
                f"scenario_filter={args.scenario_filter}",
                f"scenario_filter.limit_total_scenarios={args.pool_size}",
                "scenario_filter.shuffle=false",
                "worker=single_machine_thread_pool",
                "worker.max_workers=16",
                f"output_dir={out_dir}",
            ],
        )
        scenarios = build_scenario_builder(cfg).get_scenarios(build_scenario_filter(cfg.scenario_filter), build_worker(cfg))
        rows = [
            {
                "index": i,
                "token": str(s.token),
                "log_name": str(s.log_name),
                "scenario_name": str(s.scenario_name),
                "scenario_type": str(s.scenario_type),
            }
            for i, s in enumerate(scenarios)
        ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    pool = rows[: max(args.subset_sizes)]
    for size in args.subset_sizes:
        subset = pool[:size]
        payload = {
            "name": f"nuplan_official_closed_loop_{size}",
            "date": "2026-04-28",
            "seed": args.seed,
            "requested_size": size,
            "actual_size": len(subset),
            "db_files": sorted({row.get("db_file", "") for row in subset if row.get("db_file")}),
            "scenario_filter": scenario_filter_name,
            "tokens": [row["token"] for row in subset],
            "scenarios": subset,
        }
        path = out_dir / f"nuplan_official_closed_loop_{size}_scenarios.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path} ({len(subset)} scenarios)")


if __name__ == "__main__":
    main()
