#!/usr/bin/env python3
"""Run your parking planner on every scenario and print the scoreboard.

    python run.py                        # all scenarios, seed 0
    python run.py --scenario parallel    # just one (repeatable)
    python run.py --seed 3               # a different lot layout
    python run.py --planner my_other_planner.py

Results (plots, summary.json, scoreboard.md) are written to ./results/.
A run of only some scenarios (--scenario) writes their plots but leaves
summary.json and scoreboard.md from your last full run alone.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `common`
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from sim.checker import PLAN_TIME_LIMIT, CheckResult, check  # noqa: E402
from sim.grading import score  # noqa: E402
from sim.harness import TimeLimit, describe_exception, strict_json, time_limit  # noqa: E402
from sim.scenarios import PAR, SCENARIOS  # noqa: E402


def load_planner(path: str):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_planner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report_parts(mod) -> None:
    """For the guided planner.py: say which of Parts 1 and 2 are still TODO."""
    if not all(hasattr(mod, n) for n in ("drive", "path_is_free", "min_clearance", "plan_path")):
        return
    probe = SCENARIOS["empty_lot"](0)
    xs, ys, hs = (np.array([v]) for v in (probe.start.x, probe.start.y, probe.start.yaw))
    parts = []
    for label, fn in (("Part 1 drive", lambda: mod.drive((0.0, 0.0, 0.0), 0.0, 1.0)),
                      ("Part 2 path_is_free", lambda: mod.path_is_free(probe, xs, ys, hs)),
                      ("Part 2 min_clearance", lambda: mod.min_clearance(probe, xs, ys, hs))):
        try:
            fn()
            parts.append(f"{label}: written")
        except NotImplementedError:
            parts.append(f"{label}: TODO")
        except Exception as e:  # report, but let the run continue
            parts.append(f"{label}: raises {type(e).__name__}")
    print(" | ".join(parts))


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def run_one(Planner, problem, planner_file):
    """Plan on a copy of the problem (so the planner cannot edit the lot) and check it."""
    t0 = time.perf_counter()
    try:
        with time_limit(PLAN_TIME_LIMIT + 0.5):
            plan = Planner().plan(copy.deepcopy(problem))
    except TimeLimit:
        return CheckResult().fail(f"planning took longer than {PLAN_TIME_LIMIT:.0f} s (stopped)"), ""
    except Exception as e:
        tb = traceback.format_exc()
        return CheckResult().fail("plan() raised " + describe_exception(e, planner_file)), tb
    return check(problem, plan, time.perf_counter() - t0), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="run only this scenario")
    ap.add_argument("--planner", default=os.path.join(HERE, "planner.py"), help="planner file")
    ap.add_argument("--seed", type=int, default=0, help="lot layout seed, 0 or more (0 = public layout)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be 0 or more")
    partial = bool(args.scenario)

    mod = load_planner(args.planner)
    Planner = mod.Planner
    report_parts(mod)
    names = args.scenario or list(SCENARIOS)
    os.makedirs(args.out, exist_ok=True)
    if not args.no_plot:
        from sim.plotting import plot_run

    header = (f"{'scenario':<15} {'set':<7} {'result':<6} {'score':>6} {'length':>7} {'par':>5} "
              f"{'cusps':>5} {'min_clr':>7} {'lat':>5} {'head':>5} {'steer':>5} {'time':>6}")
    print(header)
    print("-" * len(header))
    summary = {"planner": os.path.relpath(args.planner, HERE), "seed": args.seed, "scenarios": {}}
    t_start = time.time()
    for name in names:
        problem = SCENARIOS[name](args.seed)
        par_len, par_cusps = PAR[name]
        result, tb = run_one(Planner, problem, args.planner)
        pts = score(result, par_len, par_cusps, core=problem.core)
        m = result.metrics
        print(f"{name:<15} {'core' if problem.core else 'stretch':<7} {'PASS' if result.passed else 'FAIL':<6} "
              f"{pts['total']:>6.1f} {fmt(m.get('length_m'), 1):>7} {par_len:>5.0f} "
              f"{m.get('cusps', '-'):>5} {fmt(m.get('min_clearance_m')):>7} {fmt(m.get('lateral_m')):>5} "
              f"{fmt(m.get('heading_deg'), 1):>5} {m.get('steer_events', '-'):>5} "
              f"{fmt(m.get('plan_time_s')):>6}")
        if not result.passed:
            print(f"    -> {result.reason}")
            if tb:
                print("       (full traceback in results/summary.json)" if not partial
                      else "       (run all scenarios to get the full traceback in results/summary.json)")
        summary["scenarios"][name] = {"core": problem.core, "passed": result.passed,
                                      "reason": result.reason, "score": pts["total"], "points": pts,
                                      "metrics": m, "par_length_m": par_len, "par_cusps": par_cusps,
                                      "traceback": tb}
        if not args.no_plot:
            plot_run(problem, result, pts["total"], os.path.join(args.out, f"{name}.png"))

    def avg(core):
        vals = [e["score"] for e in summary["scenarios"].values() if e["core"] == core]
        return float(np.mean(vals)) if vals else math.nan

    summary["core_average"], summary["stretch_average"] = avg(True), avg(False)
    print("-" * len(header))
    print(f"core average {fmt(summary['core_average'], 1)} | stretch average "
          f"{fmt(summary['stretch_average'], 1)} | {time.time() - t_start:.1f} s")
    fallback = sorted(getattr(mod, "FALLBACK_USED", ()) or ())
    if fallback:
        print("Used the plain Dubins path (it ignores obstacles) because: " + "; ".join(fallback)
              + ". Run `python check.py` to test each part.")
    if partial:
        where = "results" if os.path.abspath(args.out) == os.path.join(HERE, "results") else args.out
        print(f"Only some scenarios ran, so {where}/summary.json and scoreboard.md were left as they were "
              "(plots were updated).")
        return 0
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(strict_json(summary), f, indent=2, allow_nan=False)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
        f.write(f"planner: `{summary['planner']}`, seed {args.seed}\n\n")
        f.write("| scenario | set | result | score |\n|---|---|---|---|\n")
        for name, e in summary["scenarios"].items():
            res = "pass" if e["passed"] else "FAIL: " + e["reason"]
            f.write(f"| {name} | {'core' if e['core'] else 'stretch'} | {res} | {e['score']:.1f} |\n")
        f.write(f"\ncore average: {fmt(summary['core_average'], 1)}, "
                f"stretch average: {fmt(summary['stretch_average'], 1)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
