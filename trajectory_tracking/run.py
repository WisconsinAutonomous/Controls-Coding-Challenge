#!/usr/bin/env python3
"""Run your controller on every scenario and print the scoreboard.

    python run.py                       # all scenarios, nominal car
    python run.py --scenario hairpin    # just one (repeatable)
    python run.py --perturb 5           # also 5 randomly perturbed cars per scenario
    python run.py --backend chrono      # optional: PyChrono vehicle instead of NumPy
    python run.py --controller my_other_controller.py

Results (plots, summary.json, scoreboard.md) are written to ./results/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `common`
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from sim.grading import metrics, score  # noqa: E402
from sim.plant import NOMINAL, perturbed_params  # noqa: E402
from sim.runner import run  # noqa: E402
from sim.scenarios import SCENARIOS  # noqa: E402


def load_controller(path: str):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


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
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="run only this scenario")
    ap.add_argument("--controller", default=os.path.join(HERE, "controller.py"), help="controller file")
    ap.add_argument("--perturb", type=int, default=0, help="perturbed cars per scenario")
    ap.add_argument("--seed", type=int, default=0, help="noise / perturbation seed")
    ap.add_argument("--backend", choices=["numpy", "chrono"], default="numpy")
    ap.add_argument("--plant-params", help="JSON file overriding the nominal car parameters")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    Controller = load_controller(args.controller)
    params = NOMINAL
    if args.plant_params:
        with open(args.plant_params) as f:
            overrides = json.load(f)
        known = set(NOMINAL.__dataclass_fields__)
        bad = sorted(set(overrides) - known)
        if bad:
            ap.error(f"unknown key(s) {bad} in {args.plant_params}; allowed keys: {sorted(known)}")
        params = replace(NOMINAL, **overrides)
    names = args.scenario or list(SCENARIOS)
    os.makedirs(args.out, exist_ok=True)
    if not args.no_plot:
        from sim.plotting import plot_run

    header = (f"{'scenario':<12} {'set':<7} {'result':<15} {'score':>6} {'lat_rms':>8} {'lat_max':>8} "
              f"{'v_rms':>6} {'stop_err':>8} {'jerk':>6} {'steer_rt':>8} {'time':>6} {'ms_p95':>6}")
    print(header)
    print("-" * len(header))
    summary = {"controller": os.path.relpath(args.controller, HERE), "backend": args.backend,
               "seed": args.seed, "scenarios": {}}
    t_start = time.time()
    for name in names:
        scn = SCENARIOS[name]()
        if args.backend == "chrono" and not scn.chrono_ok:
            print(f"{name:<12} skipped on the chrono backend")
            continue
        log = run(scn, Controller, params, seed=args.seed, backend=args.backend)
        m = metrics(log)
        pts = score(log, m)
        result = "ok" if log.completed else log.status.upper()
        print(f"{name:<12} {'core' if scn.core else 'stretch':<7} {result:<15} {pts['total']:>6.1f} "
              f"{fmt(m.get('lat_rms_m'), 3):>8} {fmt(m.get('lat_max_m'), 3):>8} "
              f"{fmt(m.get('speed_rms_mps')):>6} {fmt(m.get('stop_err_m')):>8} "
              f"{fmt(m.get('jerk_rms')):>6} {fmt(m.get('steer_rate_rms'), 3):>8} "
              f"{fmt(m.get('time_ratio')):>6} {fmt(m.get('compute_ms_p95'), 1):>6}")
        if log.message:
            print(f"    -> {log.message.strip().splitlines()[-1]}")
        entry = {"core": scn.core, "status": log.status, "completed": log.completed,
                 "score": pts["total"], "points": pts, "metrics": m, "stops": log.stops,
                 "message": log.message}
        if not args.no_plot:
            plot_run(log, pts["total"], os.path.join(args.out, f"{name}.png"))

        if args.perturb > 0:
            rng = np.random.default_rng(1000 + args.seed)
            scores = []
            for k in range(args.perturb):
                p = perturbed_params(rng, params)
                lg = run(scn, Controller, p, seed=args.seed + k + 1, backend=args.backend)
                scores.append(score(lg, metrics(lg))["total"])
            entry["perturbed_scores"] = scores
            print(f"    perturbed x{args.perturb}: mean {np.mean(scores):.1f}, min {np.min(scores):.1f}, "
                  f"failed {sum(1 for s in scores if s == 0)}")
        summary["scenarios"][name] = entry

    def avg(core):
        vals = [e["score"] for e in summary["scenarios"].values() if e["core"] == core]
        return float(np.mean(vals)) if vals else math.nan

    summary["core_average"], summary["stretch_average"] = avg(True), avg(False)
    print("-" * len(header))
    print(f"core average {fmt(summary['core_average'], 1)} | stretch average "
          f"{fmt(summary['stretch_average'], 1)} | {time.time() - t_start:.1f} s")
    if args.scenario:
        print(f"(partial run: plots updated, but summary.json and scoreboard.md in {args.out} were not; "
              "run without --scenario for your final results)")
        return 0
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(_strict(summary), f, indent=2, allow_nan=False)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
        f.write("| scenario | set | result | score |\n|---|---|---|---|\n")
        for name, e in summary["scenarios"].items():
            res = "ok" if e["completed"] else e["status"]
            f.write(f"| {name} | {'core' if e['core'] else 'stretch'} | {res} | {e['score']:.1f} |\n")
        f.write(f"\ncore average: {fmt(summary['core_average'], 1)}, "
                f"stretch average: {fmt(summary['stretch_average'], 1)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
