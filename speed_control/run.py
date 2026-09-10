#!/usr/bin/env python3
"""Drive every scenario with your controller and print the scoreboard.

    python run.py                          # all scenarios
    python run.py --scenario hill          # just one (repeatable)
    python run.py --perturb 5              # also 5 randomly perturbed cars per scenario

Plots, summary.json and scoreboard.md are written to ./results/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from common.messages import CarState  # noqa: E402
from sim.car import perturbed  # noqa: E402
from sim.grading import metrics, score  # noqa: E402
from sim.road import SpeedTarget  # noqa: E402
from sim.runner import run  # noqa: E402
from sim.scenarios import SCENARIOS  # noqa: E402


def load(path):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parts_status(mod) -> str:
    """Which TODOs are written, by calling each once."""
    try:
        ctrl = mod.SpeedController()
    except Exception as e:  # noqa: BLE001
        return f"SpeedController() could not be created ({type(e).__name__}: {e})"
    probes = [("1", lambda: (mod.to_tbs(1.0), ctrl.p_term(1.0))), ("2", lambda: ctrl.i_term(0.0)),
              ("3", lambda: ctrl.feedforward(SpeedTarget(v=1.0, a=0.0), CarState(v=1.0))),
              ("4", lambda: ctrl.stop_logic(0.0, CarState(v=1.0), SpeedTarget(v=1.0, a=0.0)))]
    out = []
    for name, fn in probes:
        try:
            fn()
            out.append(f"part {name} done")
        except NotImplementedError:
            out.append(f"part {name} TODO")
        except Exception as e:
            out.append(f"part {name} ERROR ({type(e).__name__})")
    return ", ".join(out)


def _strict(o):
    """Make an object strict-JSON safe: NaN/inf become null, numpy scalars become floats."""
    if isinstance(o, dict):
        return {str(k): _strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strict(v) for v in o]
    if isinstance(o, (bool, str)) or o is None:
        return o
    try:
        f = float(o)
    except (TypeError, ValueError):
        return str(o)
    return f if math.isfinite(f) else None


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS))
    ap.add_argument("--controller", default=os.path.join(HERE, "controller.py"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=int, default=0, help="perturbed cars per scenario")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    mod = load(args.controller)
    print(f"Your controller: {parts_status(mod)}\n")
    os.makedirs(args.out, exist_ok=True)
    if not args.no_plot:
        from sim.plotting import plot_run
    head = (f"{'scenario':<18} {'set':<8} {'result':<15} {'score':>5} {'v_rms':>6} {'speeding':>8} "
            f"{'stop_err':>8} {'jerk':>5} {'cmd':>5} {'time':>5} {'gap_s':>6}")
    print(head)
    print("-" * len(head))
    summary = {"seed": args.seed, "parts": parts_status(mod), "scenarios": {}}
    for name in args.scenario or list(SCENARIOS):
        scn = SCENARIOS[name]()
        log = run(scn, mod.SpeedController, seed=args.seed)
        m = metrics(log)
        pts = score(log, m)
        result = "ok" if log.completed else log.status.upper()
        print(f"{name:<18} {'core' if scn.core else 'stretch':<8} {result:<15} {pts['total']:>5.0f} "
              f"{fmt(m.get('speed_rms')):>6} {fmt(m.get('speeding')):>8} {fmt(m.get('stop_error')):>8} "
              f"{fmt(m.get('jerk_rms'), 1):>5} {fmt(m.get('cmd_rate'), 0):>5} {fmt(m.get('time_ratio')):>5} "
              f"{fmt(m.get('min_time_gap')):>6}")
        if log.message:
            print(f"    -> {log.message.strip().splitlines()[-1]}")
        entry = {"core": scn.core, "status": log.status, "completed": log.completed, "score": pts["total"],
                 "points": pts, "metrics": m, "stops": log.stops, "message": log.message}
        if args.perturb:
            rng = np.random.default_rng(500 + args.seed)
            scores = [score(lg, metrics(lg))["total"] for lg in
                      (run(scn, mod.SpeedController, perturbed(rng, scn.params), seed=args.seed + k + 1)
                       for k in range(args.perturb))]
            entry["perturbed_scores"] = scores
            print(f"    perturbed x{args.perturb}: mean {np.mean(scores):.0f}, min {np.min(scores):.0f}")
        if not args.no_plot:
            plot_run(log, pts["total"], os.path.join(args.out, f"{name}.png"))
        summary["scenarios"][name] = entry

    def avg(core):
        vals = [e["score"] for e in summary["scenarios"].values() if e["core"] == core]
        return float(np.mean(vals)) if vals else math.nan

    summary["core_average"], summary["stretch_average"] = avg(True), avg(False)
    print("-" * len(head))
    print(f"core average {fmt(summary['core_average'], 0)} | stretch average {fmt(summary['stretch_average'], 0)}")
    print("columns: stop_err [m], jerk and cmd (how jumpy your throttle/brake requests are) [m/s^3], "
          "time = finish time / planned time, gap_s = smallest time gap to the car ahead [s]")
    if args.scenario:
        print(f"(partial run: plots updated, but summary.json and scoreboard.md in {args.out} were not; "
              "run without --scenario for your final results)")
        return 0
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(_strict(summary), f, indent=2, allow_nan=False)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
        f.write(f"Parts: {summary['parts']}\n\n| scenario | set | result | score |\n|---|---|---|---|\n")
        for name, e in summary["scenarios"].items():
            f.write(f"| {name} | {'core' if e['core'] else 'stretch'} | "
                    f"{'ok' if e['completed'] else e['status']} | {e['score']:.0f} |\n")
        f.write(f"\ncore average {fmt(summary['core_average'], 0)}, stretch average "
                f"{fmt(summary['stretch_average'], 0)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
