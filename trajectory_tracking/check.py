#!/usr/bin/env python3
"""Checkpoints for Trajectory Tracking.  Run after each part:

    python check.py            # every part
    python check.py part3      # just one part

Each check prints what it expected, what your code did, and a hint.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from common import vehicle  # noqa: E402
from common.geometry import TrackBuilder  # noqa: E402
from common.messages import CarState, ReferenceTrajectory  # noqa: E402

GREEN, RED, DIM, END = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = END = ""


class CheckFailed(Exception):
    pass


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def load(path: str):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_from(track, v=5.0, s0=0.0, length=50.0) -> dict:
    s = np.arange(s0, min(s0 + length, track.length), 0.5)
    x, y, yaw = track.interp(s)
    n = len(s)
    return ReferenceTrajectory.from_arrays(x, y, yaw, np.full(n, v), np.zeros(n),
                                           track.curvature_at(s)).as_arrays()


def require(mod, upto: int) -> None:
    ctrl = mod.Controller()
    pts = ref_from(TrackBuilder().straight(60).build())
    st = CarState(x=1.0, y=0.0, v=3.0, psi=0.0)
    steps = [lambda: ctrl.path_errors(st, pts), lambda: ctrl.speed_accel(st, pts, 2),
             lambda: ctrl.pure_pursuit(st, pts, 2)]
    for k, fn in enumerate(steps[:upto], start=1):
        try:
            fn()
        except NotImplementedError as e:
            raise NotImplementedError(f"finish part {k} first ({e})")


def simple_steering(ctrl):
    """The stand-in steering from the glue, built on YOUR Part 1 (used to check Part 2 on its own)."""
    def steer(state, pts, i):
        _, e_lat, e_yaw = ctrl.path_errors(state, pts)
        return -0.3 * e_lat - 1.0 * e_yaw
    return steer


def _pose_on(track, s, d=0.0, dpsi=0.0, v=5.0):
    x, y, yaw = (float(a) for a in track.interp(s))
    return CarState(x=x - d * math.sin(yaw), y=y + d * math.cos(yaw), v=v, psi=yaw + dpsi)


# -----------------------------------------------------------------------------
def part1(mod):
    ctrl = mod.Controller()
    heading = math.radians(30)
    line = TrackBuilder(yaw0=heading).straight(60).build()
    pts = ref_from(line)
    # 10 m along the line, 0.8 m to its LEFT, heading 5 deg more to the left
    i, e_lat, e_yaw = ctrl.path_errors(_pose_on(line, 10.0, d=0.8, dpsi=math.radians(5)), pts)
    expect(abs(pts["x"][i] - 10 * math.cos(heading)) < 0.3 and abs(pts["y"][i] - 10 * math.sin(heading)) < 0.3,
           f"The closest point should be near 10 m along the path (index ~20); you returned index {i}.")
    expect(abs(e_lat - 0.8) < 0.05, f"A car 0.8 m LEFT of the path should give e_lat = +0.8; you returned {e_lat:+.3f}. "
           "Hint: left of a path with heading yaw is the direction (-sin(yaw), cos(yaw)).")
    expect(abs(e_yaw - math.radians(5)) < 0.01,
           f"The car points 5 deg left of the path, so e_yaw should be +0.087 rad; you returned {e_yaw:+.3f}.")
    _, e_lat, _ = ctrl.path_errors(_pose_on(line, 10.0, d=-0.8), pts)
    expect(abs(e_lat + 0.8) < 0.05, f"A car 0.8 m RIGHT of the path should give e_lat = -0.8; you returned {e_lat:+.3f}.")
    # The car pointing 60 deg off the path: e_lat is measured along the PATH's left, not the car's.
    _, e_lat, _ = ctrl.path_errors(_pose_on(line, 10.0, d=0.8, dpsi=math.radians(60)), pts)
    expect(abs(e_lat - 0.8) < 0.05, f"A car 0.8 m left of the path but pointing 60 deg off it should still give "
           f"e_lat = +0.8; you returned {e_lat:+.3f}. Hint: use the path's heading at the closest point, not the car's.")
    # heading wrap: path heading +179 deg, car heading -179 deg -> 2 deg to the left, not 358
    wrap = TrackBuilder(yaw0=math.radians(179)).straight(40).build()
    pts = ref_from(wrap)
    car = _pose_on(wrap, 5.0, dpsi=math.radians(2))
    car.psi = (car.psi + math.pi) % (2 * math.pi) - math.pi      # localization reports -179 deg
    _, _, e_yaw = ctrl.path_errors(car, pts)
    expect(abs(e_yaw - math.radians(2)) < 0.01,
           f"Path heading +179 deg, car heading -179 deg: the car is 2 deg LEFT (+0.035 rad), but you returned "
           f"{e_yaw:+.3f} rad. Hint: wrap angle differences with wrap_to_pi.")
    return "closest point, lateral error sign, heading error and angle wrapping all correct"


def part2(mod):
    require(mod, 1)
    ctrl = mod.Controller()
    pts = ref_from(TrackBuilder().straight(60).build(), v=8.0)
    slow = ctrl.speed_accel(CarState(x=0.0, v=5.0), pts, 0)
    fast = mod.Controller().speed_accel(CarState(x=0.0, v=11.0), pts, 0)
    expect(slow > 0.5 and fast < -0.5, f"At 5 m/s with an 8 m/s target you asked for {slow:+.2f} m/s^2, "
           f"and at 11 m/s for {fast:+.2f}. Too slow should accelerate, too fast should brake.")
    from sim.grading import metrics
    from sim.runner import run
    from sim.scenarios import SCENARIOS

    def factory():
        c = mod.Controller()
        c.pure_pursuit = simple_steering(c)     # Part 2 is checked with the simple steering
        return c
    log = run(SCENARIOS["stop_sign"](), factory)
    expect(log.status not in ("not_implemented", "exception", "bad_output"),
           "stop_sign crashed: " + (log.message.strip().splitlines()[-1] if log.message else log.status))
    expect(log.status != "off_road", f"stop_sign: {log.message}. This check steers with the simple stand-in "
           "built on your Part 1, so check path_errors() (the signs of e_lat and e_yaw).")
    expect(log.completed, f"stop_sign did not finish: {log.message or log.status}. "
           "Hint: from rest the speed right at the car is ~0; look a few points ahead.")
    m = metrics(log)
    expect(m["speed_rms_mps"] < 0.8, f"Speed error RMS on stop_sign is {m['speed_rms_mps']:.2f} m/s (want < 0.8). "
           "Hint: add the reference acceleration as feedforward and some integral action.")
    return f"stop_sign completes with speed error RMS {m['speed_rms_mps']:.2f} m/s"


def part3(mod):
    require(mod, 1)
    ctrl = mod.Controller()
    ctrl.pure_pursuit(CarState(x=5.0, y=0.0, v=5.0, psi=0.0), ref_from(TrackBuilder().straight(60).build()), 10)
    # Pin the lookahead so the check does not depend on your tuning: ld = 6 m.
    ctrl.lookahead_min, ctrl.lookahead_gain = 6.0, 0.0
    for heading_deg in (0.0, 150.0, -100.0):
        h = math.radians(heading_deg)
        straight = TrackBuilder(yaw0=h).straight(60).build()
        pts = ref_from(straight)
        d0 = ctrl.pure_pursuit(_pose_on(straight, 5.0), pts, 10)
        expect(abs(d0) < 0.01, f"On a straight path heading {heading_deg:.0f} deg, on the path and pointing along it, "
               f"the steering should be ~0; you returned {d0:+.3f} rad. Hint: alpha is the angle to the goal "
               "point measured FROM THE CAR'S HEADING (subtract psi).")
        d_right = ctrl.pure_pursuit(_pose_on(straight, 5.0, d=-1.0), pts, 10)
        expect(d_right > 0.01, f"On a path heading {heading_deg:.0f} deg, 1 m to the RIGHT of it, you should steer LEFT "
               f"(positive); you returned {d_right:+.3f} rad.")
        d_left = ctrl.pure_pursuit(_pose_on(straight, 5.0, d=1.0), pts, 10)
        expect(d_left < -0.01, f"On a path heading {heading_deg:.0f} deg, 1 m to the LEFT of it, you should steer "
               f"RIGHT (negative); you returned {d_left:+.3f} rad.")
    R = 12.0
    for heading_deg in (0.0, 135.0):
        circle = TrackBuilder(yaw0=math.radians(heading_deg)).arc(R, 300).build()
        pts = ref_from(circle, v=5.0)
        want = math.atan(vehicle.WHEELBASE / R)
        got = ctrl.pure_pursuit(_pose_on(circle, 0.0), pts, 0)
        expect(abs(got - want) < 0.03, f"Sitting on a circle of radius {R:.0f} m (turning left, starting heading "
               f"{heading_deg:.0f} deg), pure pursuit should steer about atan(L/R) = {want:.3f} rad; you returned "
               f"{got:+.3f}. Hint: alpha is measured from the car's heading to the goal point, and k = 2 sin(alpha) / D "
               "with D the distance to the goal point.")
    return f"steers 0 on the path, toward it when off it, at any heading, and {got:.3f} rad on a 12 m circle (ideal {want:.3f})"


def part4(mod_path):
    require(load(mod_path), 3)
    from sim.grading import metrics, score
    from sim.runner import run
    from sim.scenarios import SCENARIOS
    mod = load(mod_path)
    lines, ok = [], True
    for name, fn in SCENARIOS.items():
        scn = fn()
        if not scn.core:
            continue
        log = run(scn, mod.Controller)
        s = score(log, metrics(log))["total"]
        passed = log.completed and s > 0
        ok &= passed
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'} ({s:.0f}/100){'' if passed else ' ' + log.message.strip().splitlines()[-1] if log.message else ''}")
    expect(ok, "Not every core scenario passes yet:\n      " + "\n      ".join(lines))
    return "all core scenarios pass: " + "; ".join(lines)


PARTS = {"part1": part1, "part2": part2, "part3": part3, "part4": part4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", help="which parts, e.g. part1 part3 (default: all)")
    ap.add_argument("--controller", default=os.path.join(HERE, "controller.py"))
    args = ap.parse_args()
    names = args.parts or list(PARTS)
    bad = [n for n in names if n not in PARTS]
    if bad:
        ap.error(f"unknown part(s) {bad}; choose from {list(PARTS)}")
    n_ok = 0
    for name in names:
        try:
            msg = PARTS[name](args.controller if name == "part4" else load(args.controller))
            print(f"{GREEN}PASS{END} {name}: {msg}")
            n_ok += 1
        except NotImplementedError as e:
            print(f"{DIM}TODO{END} {name}: {e}")
        except CheckFailed as e:
            print(f"{RED}FAIL{END} {name}: {e}")
        except Exception as e:
            print(f"{RED}ERROR{END} {name}: {type(e).__name__}: {e}")
    print(f"\n{n_ok}/{len(names)} checks passed.")
    return 0 if n_ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
