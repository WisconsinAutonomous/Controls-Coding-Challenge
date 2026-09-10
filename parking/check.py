#!/usr/bin/env python3
"""Checkpoints for the parking walkthrough.

    python check.py            # every part
    python check.py part1      # Part 1: drive()
    python check.py part2      # Part 2: path_is_free() and min_clearance()
    python check.py part3      # Part 3: plan_path()
    python check.py part4      # the core scenarios, start to finish
    python check.py --planner my_planner.py part1

Each check prints PASS, FAIL (with what we expected, what you returned, and
a hint) or TODO (not written yet).  Exits with 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import numbers
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from sim.checker import PLAN_TIME_LIMIT, check as grade  # noqa: E402
from sim.harness import TimeLimit, describe_exception, time_limit  # noqa: E402
from sim.grading import score  # noqa: E402
from sim.lot import OrientedBox, ParkingProblem, ParkingSpot, Pose  # noqa: E402
from sim.scenarios import PAR, SCENARIOS  # noqa: E402
from tools import to_segments  # noqa: E402

POS_TOL, YAW_TOL = 0.02, math.radians(0.5)


class Fail(Exception):
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.hint = hint


def _deg(a):
    return math.degrees(math.remainder(a, 2 * math.pi))


def _where(x, y, yaw):
    return f"({x:.2f}, {y:.2f}) facing {_deg(yaw):.1f} deg"


INTEGRATION_HINT = ("you are close, so this looks like integration error, not a wrong formula: taking "
                    "10 cm steps with the heading from the start of each step drifts about 7 cm over a "
                    "quarter circle. Use the exact arc formula, or integrate in much smaller steps "
                    "(1 cm) and keep every 10th pose.")


def _end_pose_check(out, want, what, hint=""):
    x, y, yaw = _arrays(out)
    ex, ey, eyaw = want
    dpos = math.hypot(x[-1] - ex, y[-1] - ey)
    dyaw = abs(math.remainder(yaw[-1] - eyaw, 2 * math.pi))
    if dpos > POS_TOL or dyaw > YAW_TOL:
        close = dpos <= 0.2 and dyaw <= math.radians(1.0)
        raise Fail(f"{what}: expected the car at {_where(ex, ey, eyaw)}, "
                   f"yours ends at {_where(x[-1], y[-1], yaw[-1])} "
                   f"({dpos:.2f} m and {math.degrees(dyaw):.1f} deg off)",
                   INTEGRATION_HINT if close else hint)


def _truth(value, fn):
    """A True/False answer (Python or numpy bool); anything else is a clear failure."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise Fail(f"{fn}() should return True or False; it returned {type(value).__name__} {value!r}",
               "return a single True/False for the whole path (for example `not collides.any()`)")


def _number(value, fn):
    if isinstance(value, (numbers.Real, np.floating, np.integer)) and not isinstance(value, bool):
        v = float(value)
        if math.isfinite(v):
            return v
        raise Fail(f"{fn}() returned {v}; it should be a finite distance in meters")
    raise Fail(f"{fn}() should return one number (meters); it returned {type(value).__name__}",
               "take the smallest value over the whole path, for example float(clearances.min())")


def _hint_for(reason: str) -> str:
    """A hint that matches why the grader rejected a path."""
    if "planning took" in reason:
        return (f"the whole plan must be ready within {PLAN_TIME_LIMIT:.0f} s. Try fewer staging "
                "candidates, stop early once you have a good path, or build the checker once per problem.")
    if "collision" in reason:
        return ("is your path_is_free() right (python check.py part2)? did you keep a path that was not free? "
                "run.py --scenario ... draws the car where it first touches something.")
    if "slides sideways" in reason:
        return ("the car moves sideways compared with its yaw: are x, y and yaw from the same path, "
                "and is yaw in radians?")
    if "tighter than the car" in reason:
        return "build turns from dubins_path() or drive() with |curvature| <= MAX_CURVATURE"
    if "inside the spot" in reason or "final heading" in reason:
        return "aim at problem.spot.goal_pose(), which parks the car centered and facing the right way"
    if "first point" in reason:
        return "start the path exactly at problem.start"
    if "does not start where" in reason:
        return "each piece must start where the previous one ended: join() checks that for you"
    return "run `python run.py` and look at the plot in results/"


def _arrays(out):
    if all(hasattr(out, a) for a in ("x", "y", "yaw")):       # a Path from tools
        out = (out.x, out.y, out.yaw)
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise Fail(f"expected three arrays (x, y, yaw) or a Path, got {type(out).__name__}",
                   "return x, y, yaw  (three numpy arrays of the same length), or the Path from join()")
    x, y, yaw = (np.atleast_1d(np.asarray(a, dtype=float)) for a in out)
    if not (len(x) == len(y) == len(yaw)) or len(x) < 2:
        raise Fail(f"x, y, yaw must have the same length (at least 2); got {len(x)}, {len(y)}, {len(yaw)}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(yaw))):
        raise Fail("your arrays contain NaN or inf")
    return x, y, yaw


# ---------------------------------------------------------------------- #
def part1(mod):
    drive = mod.drive
    MAX = getattr(mod, "MAX_CURVATURE", 0.2145)
    R = 5.0
    q = 0.5 * math.pi * R          # a quarter circle of radius 5

    def straight_ahead():
        out = drive((0.0, 0.0, 0.0), 0.0, 5.0)
        _end_pose_check(out, (5.0, 0.0, 0.0), "5 m straight ahead from (0, 0) facing east")
        x, y, yaw = _arrays(out)
        if math.hypot(x[0], y[0]) > 1e-6 or abs(yaw[0]) > 1e-6:
            raise Fail(f"the first pose should be the start pose (0, 0, 0); yours is {_where(x[0], y[0], yaw[0])}",
                       "include the starting pose as element 0")

    def spacing():
        x, y, yaw = _arrays(drive((0.0, 0.0, 0.0), 0.2, 3.0, step=0.1))
        d = np.hypot(np.diff(x), np.diff(y))
        if d.max() > 0.1 + 1e-6:
            raise Fail(f"poses should be at most step = 0.1 m apart; yours are up to {d.max():.2f} m apart",
                       "use n = ceil(|distance| / step) steps")
        if len(x) < 31:
            raise Fail(f"3 m at step 0.1 m needs at least 31 poses; you returned {len(x)}")

    def straight_back():
        _end_pose_check(drive((0.0, 0.0, 0.0), 0.0, -3.0), (-3.0, 0.0, 0.0),
                        "3 m straight BACK from (0, 0) facing east",
                        "a negative distance moves the car opposite to where it points; yaw stays the same")

    def quarter_left():
        _end_pose_check(drive((0.0, 0.0, 0.0), 1.0 / R, q), (5.0, 5.0, 0.5 * math.pi),
                        "a quarter circle of radius 5 m to the LEFT (should end facing north)",
                        "yaw changes by curvature * distance; the rear axle ends one radius left and one radius ahead")

    def quarter_right():
        _end_pose_check(drive((0.0, 0.0, 0.0), -1.0 / R, q), (5.0, -5.0, -0.5 * math.pi),
                        "a quarter circle of radius 5 m to the RIGHT (should end facing south)",
                        "negative curvature turns right: the heading decreases")

    def any_start():
        _end_pose_check(drive((10.0, 2.0, 0.5 * math.pi), 1.0 / R, q), (5.0, 7.0, math.pi),
                        "a left quarter circle starting at (10, 2) facing north (should end facing west)",
                        "your formula must use the starting yaw, not assume the car faces east")

    def backing_left():
        _end_pose_check(drive((0.0, 0.0, 0.0), 1.0 / R, -q), (-5.0, 5.0, -0.5 * math.pi),
                        "BACKING UP a quarter circle with the wheels turned left",
                        "use the same rules with a negative ds: yaw changes by curvature * (negative distance), "
                        "so the nose swings right while the rear goes back and to the left")

    def full_circle():
        _end_pose_check(drive((1.0, 1.0, 0.3), 1.0 / R, 2 * math.pi * R), (1.0, 1.0, 0.3),
                        "a full circle of radius 5 m (should come back to where it started)")

    def too_tight():
        try:
            drive((0.0, 0.0, 0.0), 0.3, 1.0)
        except ValueError:
            return
        raise Fail("curvature 0.3 1/m (a 3.3 m circle) is tighter than the car can turn, "
                   "but drive() did not raise ValueError",
                   "if abs(curvature) > MAX_CURVATURE: raise ValueError(...)")

    def full_lock():
        for k in (MAX, -MAX):
            try:
                drive((0.0, 0.0, 0.0), k, 2.0)
            except ValueError:
                raise Fail(f"curvature {k:+.4f} is exactly the car's limit (full lock), and drive() refused it",
                           "only refuse turns TIGHTER than MAX_CURVATURE: use >, not >=")

    def uses_step():
        x, y, yaw = _arrays(drive((0.0, 0.0, 0.0), 0.1, 1.2, step=0.03))
        d = np.hypot(np.diff(x), np.diff(y))
        if d.max() > 0.03 + 1e-6 or len(x) < 41:
            raise Fail(f"with step=0.03 the poses should be at most 3 cm apart (41 or more poses for 1.2 m); "
                       f"yours are up to {100 * d.max():.1f} cm apart ({len(x)} poses)",
                       "use the `step` argument instead of a fixed spacing")

    def pose_object():
        _end_pose_check(drive(Pose(2.0, 3.0, 0.0), 0.0, 1.0), (3.0, 3.0, 0.0),
                        "starting from problem.start-style object Pose(2, 3, 0)",
                        "accept both a tuple (x, y, yaw) and an object with .x .y .yaw")

    return [("drives straight ahead and starts at the start pose", straight_ahead),
            ("returns poses at most `step` apart", spacing),
            ("drives straight backward", straight_back),
            ("turns left", quarter_left),
            ("turns right", quarter_right),
            ("works from any starting pose", any_start),
            ("backs up while turning", backing_left),
            ("comes back after a full circle", full_circle),
            ("refuses turns tighter than the car can do", too_tight),
            ("accepts a full-lock turn", full_lock),
            ("uses the step you give it", uses_step),
            ("accepts problem.start as the pose", pose_object)]


def _box_problem():
    """A small lot with one parked car centered at (15, 6), facing east."""
    car = OrientedBox(15.0, 6.0, 0.0, 4.6, 1.9, "car")        # spans x 12.7..17.3, y 5.05..6.95
    spot = ParkingSpot(26.0, 6.0, 0.0, 3.0, 5.5)
    return ParkingProblem("checkpoint", "", Pose(5.0, 3.0, 0.0), spot, [car], (0.0, 30.0, 0.0, 12.0))


def _line(x0, x1, y, yaw=0.0, n=None):
    n = n or int(round(abs(x1 - x0) / 0.1)) + 1
    x = np.linspace(x0, x1, n)
    return x, np.full(n, y), np.full(n, yaw)


def part2(mod):
    P = _box_problem()

    def free_path():
        if _truth(mod.path_is_free(P, *_line(5.0, 25.0, 3.0)), "path_is_free") is not True:
            raise Fail("driving east along y = 3 passes 1.15 m below the parked car, but you said it collides",
                       "return True when nothing is hit")

    def side_overlap():
        if _truth(mod.path_is_free(P, *_line(5.0, 25.0, 4.3)), "path_is_free") is not False:
            raise Fail("driving east along y = 4.3: the rear axle stays 0.75 m from the parked car, "
                       "but the car's left side overlaps it by 15 cm. You said the path is free.",
                       "check the whole 4.5 m x 1.8 m footprint, not just the rear axle point")

    def front_bumper():
        if _truth(mod.path_is_free(P, *_line(5.0, 9.2, 6.0)), "path_is_free") is not False:
            raise Fail("driving toward the parked car and stopping with the rear axle 3.5 m from it: "
                       "the front bumper (3.7 m ahead of the rear axle) is already inside it. You said free.",
                       "the car sticks out 3.7 m ahead of and 0.8 m behind the rear axle")

    def leaves_lot():
        x = np.full(46, 25.0)
        y = np.linspace(6.0, 10.5, 46)
        if _truth(mod.path_is_free(P, x, y, np.full(46, 0.5 * math.pi)), "path_is_free") is not False:
            raise Fail("driving north until the front bumper is 2 m past the edge of the lot: you said free",
                       "leaving the lot counts as a collision (FastChecker and problem.collides_many handle it)")

    def between_poses():
        x, y, yaw = np.array([5.0, 20.0]), np.array([6.0, 6.0]), np.array([0.0, 0.0])
        if _truth(mod.path_is_free(P, x, y, yaw), "path_is_free") is not False:
            raise Fail("a path of just two poses, before and after the parked car, drives right through it "
                       "between them. You said free.",
                       "check between poses too: get_checker(problem).path_is_free(x, y, yaw) does, "
                       "or interpolate the path every few cm yourself")

    def clearance_value():
        c = _number(mod.min_clearance(P, *_line(5.0, 25.0, 3.0)), "min_clearance")
        if abs(c - 1.15) > 0.1:
            raise Fail(f"the path along y = 3 passes 1.15 m from the parked car; min_clearance says {c:.2f} m",
                       "the smallest distance from the car's footprint (not its rear axle) to anything")

    def clearance_touching():
        c = _number(mod.min_clearance(P, *_line(5.0, 25.0, 4.3)), "min_clearance")
        if c > 0.05:
            raise Fail(f"a path that overlaps the parked car should have clearance 0; yours says {c:.2f} m")

    return [("says a clear path is free", free_path),
            ("sees the side of the car, not just the rear axle", side_overlap),
            ("sees the front bumper", front_bumper),
            ("treats leaving the lot as a collision", leaves_lot),
            ("checks between the poses you give it", between_poses),
            ("measures clearance", clearance_value),
            ("reports zero clearance when touching", clearance_touching)]


def _plan_path_check(mod, name, seed):
    P = SCENARIOS[name](seed)
    t0 = time.perf_counter()
    out = mod.plan_path(P)
    dt = time.perf_counter() - t0
    x, y, yaw = _arrays(out)
    st, g = P.start, P.spot.goal_pose()
    if math.hypot(x[0] - st.x, y[0] - st.y) > 0.1 or abs(math.remainder(yaw[0] - st.yaw, 2 * math.pi)) > math.radians(3):
        raise Fail(f"your path should start at problem.start {_where(st.x, st.y, st.yaw)}; "
                   f"it starts at {_where(x[0], y[0], yaw[0])}")
    if math.hypot(x[-1] - g.x, y[-1] - g.y) > 0.1 or abs(math.remainder(yaw[-1] - g.yaw, 2 * math.pi)) > math.radians(3):
        raise Fail(f"your path should end at the goal pose {_where(g.x, g.y, g.yaw)}; "
                   f"it ends at {_where(x[-1], y[-1], yaw[-1])}",
                   "aim at problem.spot.goal_pose()")
    back = (np.diff(x) * np.cos(yaw[1:]) + np.diff(y) * np.sin(yaw[1:])) < -1e-9
    if back.any():
        raise Fail("your path backs up somewhere; Part 3 is forward only (backing up is the stretch part)")
    r = grade(P, to_segments(x, y, yaw), dt)
    if not r.passed:
        raise Fail(f"the grader rejects your path on {name}: {r.reason}", _hint_for(r.reason))
    return f"{r.metrics['length_m']:.1f} m, {r.metrics['min_clearance_m']:.2f} m clearance, {dt:.2f} s"


def part3(mod):
    return [(f"parks in {name} (layout {seed})", (lambda n=name, s=seed: _plan_path_check(mod, n, s)))
            for name, seed in (("empty_lot", 0), ("perpendicular", 0), ("angled", 0), ("perpendicular", 3))]


def part4(mod):
    def one(name, seed):
        P = SCENARIOS[name](seed)
        fallback = getattr(mod, "FALLBACK_USED", None)
        if fallback is not None:
            fallback.clear()
        t0 = time.perf_counter()
        plan = mod.Planner().plan(P)
        r = grade(P, plan, time.perf_counter() - t0)
        pts = score(r, *PAR[name], core=P.core)
        if not r.passed:
            if fallback:
                why = "; ".join(sorted(fallback))
                then = ("which is exactly what Part 3 fixes" if "Part 3" in why
                        else "and once that part is written your plan_path() takes over")
                raise Fail(f"{name} (layout {seed}) fails: {r.reason}",
                           f"this used the plain Dubins path because: {why}. That path cuts into a parked "
                           f"car here, {then}.")
            raise Fail(f"{name} (layout {seed}) fails: {r.reason}", _hint_for(r.reason))
        return f"score {pts['total']:.1f}"
    return [(f"core scenario {name}, layout {seed}", (lambda n=name, s=seed: one(n, s)))
            for name in ("empty_lot", "angled", "perpendicular") for seed in (0, 1, 2)]


PARTS = {"part1": ("Part 1: drive()", part1), "part2": ("Part 2: path_is_free() and min_clearance()", part2),
         "part3": ("Part 3: plan_path()", part3), "part4": ("Part 4: the core scenarios", part4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", help="which parts: " + ", ".join(PARTS) + " (default: all)")
    ap.add_argument("--planner", default=os.path.join(HERE, "planner.py"))
    args = ap.parse_args()
    unknown = [p for p in args.parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown part(s) {', '.join(unknown)}; choose from {', '.join(PARTS)}")
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(args.planner))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_planner", args.planner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    totals = {"PASS": 0, "FAIL": 0, "TODO": 0}
    limit = PLAN_TIME_LIMIT + 5.0
    for key in (args.parts or list(PARTS)):
        title, make = PARTS[key]
        print(f"\n{title}")
        timed_out = False
        for label, fn in make(mod):
            if timed_out:
                totals["FAIL"] += 1
                print(f"  [FAIL] {label}: skipped, because the previous check timed out")
                continue
            try:
                with time_limit(limit):
                    info = fn()
                totals["PASS"] += 1
                print(f"  [PASS] {label}" + (f"  ({info})" if info else ""))
            except NotImplementedError as e:
                totals["TODO"] += 1
                print(f"  [TODO] {label}: {e}")
            except Fail as e:
                totals["FAIL"] += 1
                print(f"  [FAIL] {label}\n         {e}")
                if e.hint:
                    print(f"         hint: {e.hint}")
            except TimeLimit:
                timed_out = True
                totals["FAIL"] += 1
                print(f"  [FAIL] {label}\n         stopped after {limit:.0f} s without an answer\n"
                      f"         hint: {_hint_for('planning took')}")
            except Exception as e:
                totals["FAIL"] += 1
                print(f"  [FAIL] {label}\n         your code raised {describe_exception(e, args.planner)}")
    print(f"\n{totals['PASS']} passed, {totals['FAIL']} failed, {totals['TODO']} not written yet")
    return 0 if totals["FAIL"] == 0 and totals["TODO"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
