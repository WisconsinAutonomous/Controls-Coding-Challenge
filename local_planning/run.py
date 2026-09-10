"""Run your planner through the scenarios and print a scoreboard.

    python run.py                          # all scenarios, seed 0, with plots
    python run.py --scenario barrels       # just one (repeatable)
    python run.py --seed 3                 # a different random layout
    python run.py --no-plot --quiet        # fast, table only
    python run.py --animate --scenario barrels   # also write a GIF
    python run.py --raise                  # let planner exceptions crash (for debugging)

Results go to local_planning/results: one PNG per scenario, summary.json and
scoreboard.md.  Running only some scenarios (--scenario) updates their plots but
leaves summary.json and scoreboard.md alone, so those always describe a full run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))   # repo root, for `common`

from sim.grading import grade                # noqa: E402
from sim.runner import run_scenario          # noqa: E402
from sim.scenarios import SCENARIOS          # noqa: E402


def load_planner(path: str):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_planner"] = mod
    spec.loader.exec_module(mod)
    return mod.Planner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="run only these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", default=os.path.join(HERE, "planner.py"))
    ap.add_argument("--plot", dest="plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--animate", action="store_true", help="also write results/<scenario>.gif")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--raise", dest="raise_errors", action="store_true")
    args = ap.parse_args(argv)

    try:
        Planner = load_planner(args.planner)
    except Exception:
        err = traceback.format_exc(limit=0).strip()
        print(f"Could not load {os.path.relpath(args.planner)}; Python says:\n\n    "
              + err.replace("\n", "\n    "))
        return 1
    names = args.scenario or list(SCENARIOS)
    os.makedirs(args.out, exist_ok=True)
    grades, t_wall, missing = [], time.time(), set()
    for name in names:
        sc = SCENARIOS[name](args.seed)
        res = run_scenario(sc, Planner, raise_errors=args.raise_errors)
        g = grade(res)
        grades.append(g)
        missing.update(res.planner_missing)
        if res.first_exception and not args.quiet:
            print(f"\n[{name}] your planner raised an exception (first one shown; "
                  f"run with --raise for the full traceback):\n  {res.first_exception}")
        if args.plot or args.animate:
            from sim import plotting
            if args.plot:
                plotting.plot_run(res, g, os.path.join(args.out, f"{name}.png"))
            if args.animate:
                plotting.animate_run(res, g, os.path.join(args.out, f"{name}.gif"))
        if not args.quiet:
            print(f"  ran {name:14s} {g.status:4s} {g.score:5.1f}"
                  + (f"   ({g.reason})" if g.status == "FAIL" else ""))

    table = scoreboard(grades, args.seed)
    print("\n" + table)
    if missing:
        check = os.path.relpath(os.path.join(HERE, "check.py"))
        print("\nNot done yet (the planner used its naive fallback): " + ", ".join(sorted(missing))
              + f".\nRun `python {check}` to check your parts one at a time.")
    print(f"(wall time {time.time() - t_wall:.1f} s)")
    if args.scenario:
        print("(ran a subset of the scenarios, so summary.json and scoreboard.md were not "
              "updated; run without --scenario for your final results)")
    else:
        with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
            f.write(table + "\n")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(_strict([g.__dict__ for g in grades]), f, indent=2, allow_nan=False)
    return 0


def _strict(x):
    """Make values strict-JSON safe: NaN and inf become null, numpy numbers become floats."""
    if isinstance(x, dict):
        return {k: _strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    return f if math.isfinite(f) else None


def scoreboard(grades, seed: int) -> str:
    rows = [f"Local planning scoreboard (seed {seed})", "",
            "| Scenario | Tier | Result | Score | Time [s] | Top penalties / failure |",
            "|---|---|---|---:|---:|---|"]
    for g in grades:
        note = g.reason if g.status == "FAIL" else (g.top_penalties() or "-")
        rows.append(f"| {g.name} | {g.tier} | {g.status} | {g.score:.1f} | {g.time:.1f} | {note} |")
    rows.append("")
    for tier in ("core", "stretch"):
        sel = [g.score for g in grades if g.tier == tier]
        if sel:
            rows.append(f"{tier.capitalize()} average: {sum(sel) / len(sel):.1f} "
                        f"({sum(g.status == 'PASS' for g in grades if g.tier == tier)}/{len(sel)} passed)")
    slow = [g for g in grades if g.metrics.get("plan_ms_mean", 0) > 100]
    if slow:
        rows.append("Note: your planner averaged over 100 ms per call in "
                    + ", ".join(g.name for g in slow) + " (the real one runs at 10 Hz).")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
