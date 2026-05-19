#!/usr/bin/env python
"""Aggregate Stage-1 pilot X results across seeds into a mean+/-std table.

Reads ``experiments/stage1_pilot_x/seed{S}/{cond}/stage1_metrics.json`` for
all (seed, cond) combinations and prints a markdown-ready summary.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List


_METRICS = [
    ("latent_return_mean", "Return"),
    ("imagined_collision_rate", "CollRate"),
    ("collision_mean", "CollMean"),
    ("rollout_stability", "Stab(ego-cos)"),
    ("rollout_stability_global", "Stab(L2)"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="experiments/stage1_pilot_x")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    p.add_argument(
        "--conds", type=str, nargs="+",
        default=["wm_object", "wm_decoupled", "wm_decoupled_no_vis"],
    )
    p.add_argument("--out-md", type=str, default="experiments/stage1_pilot_x/X_summary.md")
    p.add_argument("--out-json", type=str, default="experiments/stage1_pilot_x/X_summary.json")
    args = p.parse_args()

    root = Path(args.root)
    summary: Dict[str, Dict[str, List[float]]] = {c: {k: [] for k, _ in _METRICS} for c in args.conds}
    missing: List[str] = []
    for s in args.seeds:
        for c in args.conds:
            f = root / f"seed{s}" / c / "stage1_metrics.json"
            if not f.is_file():
                missing.append(str(f))
                continue
            with f.open() as fh:
                d = json.load(fh)
            m = d.get("metrics", d)
            for k, _ in _METRICS:
                if k in m:
                    summary[c][k].append(float(m[k]))

    out_lines: List[str] = []
    out_lines.append("# Stage 1 Pilot X: 3-seed verification of v3 ranking\n")
    out_lines.append(f"Seeds: {args.seeds}  Conditions: {args.conds}\n")
    if missing:
        out_lines.append(f"Missing files ({len(missing)}): {missing}\n")
    out_lines.append("")

    header = "| condition | " + " | ".join(short for _, short in _METRICS) + " |"
    sep = "|" + "|".join(["---"] * (len(_METRICS) + 1)) + "|"
    out_lines.append(header)
    out_lines.append(sep)

    json_dump: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in args.conds:
        cells = [c]
        json_dump[c] = {}
        for k, _ in _METRICS:
            vals = summary[c][k]
            if not vals:
                cells.append("n/a")
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{m:.3f} \u00b1 {s:.3f}")
            json_dump[c][k] = {"mean": m, "std": s, "n": len(vals), "values": vals}
        out_lines.append("| " + " | ".join(cells) + " |")

    out_lines.append("")
    out_lines.append("## Per-seed raw values")
    out_lines.append("")
    for c in args.conds:
        out_lines.append(f"### {c}")
        out_lines.append("")
        out_lines.append("| seed | " + " | ".join(short for _, short in _METRICS) + " |")
        out_lines.append("|" + "|".join(["---"] * (len(_METRICS) + 1)) + "|")
        for i, s in enumerate(args.seeds):
            cells = [str(s)]
            for k, _ in _METRICS:
                vs = summary[c][k]
                cells.append(f"{vs[i]:.3f}" if i < len(vs) else "n/a")
            out_lines.append("| " + " | ".join(cells) + " |")
        out_lines.append("")

    out_md = "\n".join(out_lines)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(out_md)
    Path(args.out_json).write_text(json.dumps(json_dump, indent=2))

    print(out_md)
    print(f"\nWrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
