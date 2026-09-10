#!/usr/bin/env python3
"""Run your route planner on every scenario and print the scoreboard.

    python run.py                          # all scenarios, seed 0
    python run.py --scenario rush_hour     # just one (repeatable)
    python run.py --seed 3                 # a different town and different trips
    python run.py --trip 5                 # plot trip 5 instead of trip 1
    python run.py --planner other.py       # grade a different route_planner.py

Results (plots, summary.json, scoreboard.md) are written to ./results/.
A run with --scenario writes plots only and leaves summary.json alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `common`
sys.path.insert(0, HERE)

from sim.runner import run_scenario  # noqa: E402
from sim.scenarios import SCENARIOS  # noqa: E402


def load_planner(path: str):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_route_planner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def strict(obj):
    """JSON-safe copy: NaN and infinity become null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [strict(v) for v in obj]
    return obj


def non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("trips are numbered from 1")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="run only this scenario")
    ap.add_argument("--planner", default=os.path.join(HERE, "route_planner.py"), help="planner file")
    ap.add_argument("--seed", type=non_negative, default=0, help="town and trip seed (0 or more)")
    ap.add_argument("--trip", type=positive, default=1, help="which trip to plot (from 1)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    planner = load_planner(args.planner)
    names = args.scenario or list(SCENARIOS)
    full_run = not args.scenario
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    if not args.no_plot:
        from sim.plotting import plot_scenario

    header = (f"{'scenario':<14} {'set':<7} {'result':<8} {'score':>6} {'trips ok':>9} "
              f"{'cost/best':>9} {'time [s]':>9}   {'A*/Dijkstra explored (info)'}")
    summary = {"planner": os.path.abspath(args.planner), "seed": args.seed, "scenarios": {}}
    t_start = time.time()
    printed_header = False
    for name in names:
        scn = SCENARIOS[name](args.seed)
        res = run_scenario(scn, planner)
        if not printed_header:
            if res.missing_parts:
                print("Not implemented yet: " + ", ".join(res.missing_parts)
                      + ".  Run `python check.py` to work through them.\n")
            print(header)
            print("-" * len(header))
            printed_header = True
        ok = [t for t in res.trips if t.ok]
        ratio = sum(t.ratio for t in ok) / len(ok) if ok else math.nan
        ex = [t.explored for t in res.trips if t.explored and t.explored.get("astar")]
        explored = "-"
        if ex:
            n_a, n_d = sum(e["astar"] for e in ex), sum(e["dijkstra"] for e in ex)
            explored = f"{n_a}/{n_d} = {n_a / max(1, n_d):.2f}"
        print(f"{name:<14} {'core' if scn.core else 'stretch':<7} {res.status:<8} {res.score:>6.1f} "
              f"{len(ok):>4}/{len(res.trips):<4} {fmt(ratio, 3):>9} {fmt(res.total_time):>9}   {explored}")
        bad = [t for t in res.trips if not t.ok]
        if res.status == "TIMEOUT":
            print(f"    planning took more than {scn.time_limit:.0f} s in total, so this scenario scores 0")
        if bad and all(t.message == bad[0].message for t in bad):
            which = "every trip" if len(bad) == len(res.trips) else f"{len(bad)} trips"
            print(f"    {which}: {bad[0].message}")
        else:
            for t in bad[:2]:
                print(f"    trip {t.index + 1} (to {t.destination}): {t.message}")
            if len(bad) > 2:
                where = summary_path if full_run else "the per-trip list above"
                print(f"    ... and {len(bad) - 2} more failed trips (see {where})")
        summary["scenarios"][name] = {
            "core": scn.core, "status": res.status, "score": res.score, "planning_time_s": res.total_time,
            "trips": [{"trip": t.index + 1, "destination": t.destination, "ok": t.ok, "message": t.message,
                       "cost_over_best": t.ratio, "points": t.points, "routes": t.routes,
                       "explored_dijkstra": (t.explored or {}).get("dijkstra"),
                       "explored_astar": (t.explored or {}).get("astar")} for t in res.trips]}
        if not args.no_plot:
            k = min(args.trip, len(scn.trips)) - 1
            if args.trip > len(scn.trips):
                print(f"    ({name} has only {len(scn.trips)} trips: plotting trip {k + 1})")
            plot_scenario(res, planner, os.path.join(args.out, f"{name}.png"), trip_index=k)

    def avg(core):
        vals = [e["score"] for e in summary["scenarios"].values() if e["core"] == core]
        return sum(vals) / len(vals) if vals else math.nan

    summary["core_average"], summary["stretch_average"] = avg(True), avg(False)
    print("-" * len(header))
    print(f"core average {fmt(summary['core_average'], 1)} | stretch average "
          f"{fmt(summary['stretch_average'], 1)} | {time.time() - t_start:.1f} s")
    if not full_run:
        print(f"(only some scenarios ran, so {summary_path} and scoreboard.md were not updated)")
        return 0
    with open(summary_path, "w") as f:
        json.dump(strict(summary), f, indent=2, allow_nan=False)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
        f.write("| scenario | set | result | score |\n|---|---|---|---|\n")
        for name, e in summary["scenarios"].items():
            f.write(f"| {name} | {'core' if e['core'] else 'stretch'} | {e['status']} | {e['score']:.1f} |\n")
        f.write(f"\ncore average: {fmt(summary['core_average'], 1)}, "
                f"stretch average: {fmt(summary['stretch_average'], 1)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
