#!/usr/bin/env python3
"""Checkpoints for Speed Control.  Run after each part:

    python check.py            # every part
    python check.py part2      # just one part

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

from common.messages import CarState, CarTBS  # noqa: E402
from sim.car import NOMINAL, Car  # noqa: E402
from sim.road import Road, SpeedTarget, Stop  # noqa: E402

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


# Later parts are switched off with neutral stand-ins, so a check only sees the
# parts it is about (and works even if you rewrote compute()).
NEUTRAL = {"i_term": (2, lambda error: 0.0),
           "feedforward": (3, lambda target, state: 0.0),
           "stop_logic": (4, lambda accel, state, target: accel)}


def make(mod, upto: int):
    """A controller with only Parts 1..upto switched on."""
    ctrl = mod.SpeedController()
    for name, (part, stand_in) in NEUTRAL.items():
        if part > upto:
            setattr(ctrl, name, stand_in)
    return ctrl


def drive(ctrl, target_fn, v0=0.0, grade=None, T=25.0, seed=1):
    """Run a controller against the car with a scripted target. Returns arrays."""
    car = Car(v0, NOMINAL, grade=grade, seed=seed)
    ts, vs, accs = [], [], []
    t = 0.0
    while t < T:
        meas = car.measure()
        car.apply(ctrl.compute(meas, target_fn(t, car)))
        car.step(0.02)
        t += 0.02
        ts.append(t)
        vs.append(car.v)
        accs.append(car.accel)
    return np.array(ts), np.array(vs), np.array(accs)


def require(mod, upto: int) -> None:
    """Earlier parts must be written before a later check makes sense."""
    ctrl = mod.SpeedController()
    steps = [lambda: (mod.to_tbs(1.0), ctrl.p_term(1.0)), lambda: ctrl.i_term(0.0),
             lambda: ctrl.feedforward(SpeedTarget(v=1.0, a=0.0), CarState(v=1.0))]
    for k, fn in enumerate(steps[:upto], start=1):
        try:
            fn()
        except NotImplementedError as e:
            raise NotImplementedError(f"finish part {k} first ({e})")


# -----------------------------------------------------------------------------
def part1(mod):
    for a, (t, b) in [(2.0, (2.0, 0.0)), (-3.0, (0.0, -3.0)), (9.0, (5.0, 0.0)),
                      (-12.0, (0.0, -10.0)), (0.0, (0.0, 0.0))]:
        out = mod.to_tbs(a)
        expect(isinstance(out, CarTBS), f"to_tbs({a}) should return a CarTBS, got {type(out).__name__}")
        expect(abs(out.t - t) < 1e-9 and abs(out.b - b) < 1e-9,
               f"to_tbs({a}) should give t={t}, b={b}; got t={out.t}, b={out.b}. "
               "Hint: positive requests go to the throttle, negative to the brake (which is negative), "
               "and each is clipped to its range.")
    ctrl = mod.SpeedController()
    expect(ctrl.kp > 0, "self.kp is still 0: pick a positive proportional gain in __init__.")
    expect(abs(ctrl.p_term(1.0) - ctrl.kp) < 1e-9 and ctrl.p_term(-2.0) < 0,
           "p_term(error) should be kp * error (positive when too slow, negative when too fast).")
    ts, vs, accs = drive(make(mod, 1), lambda t, car: SpeedTarget(v=8.0, a=0.0))
    late = ts > 20
    v_end = float(np.mean(vs[late]))
    wobble = float(np.std(accs[late]))
    expect(wobble < 0.3, f"With P control only, the car's acceleration keeps swinging in the last 5 s "
           f"(std {wobble:.2f} m/s^2): it is oscillating. Hint: kp is too large for a car whose brakes "
           "and throttle respond with a delay.")
    expect(7.0 <= v_end <= 8.6, f"With P control only and an 8 m/s target, the car settled at {v_end:.2f} m/s. "
           "Hint: if it is well below 8, kp is too small; P control needs an error to push at all.")
    return f"to_tbs works; with P only (kp = {ctrl.kp:g}) the car settles at {v_end:.2f} m/s for an 8 m/s target"


def part2(mod):
    require(mod, 1)
    ctrl = mod.SpeedController()
    ctrl.i_term(0.0)
    expect(ctrl.ki > 0, "self.ki is still 0: pick a positive integral gain in __init__.")
    # Windup, tested directly: 10 s of a large error must not produce a huge push.
    probe = mod.SpeedController()
    push = 0.0
    for _ in range(500):
        push = probe.i_term(5.0)
    expect(abs(push) <= 3.0, f"After 10 s of a 5 m/s error, i_term() asks for {push:.1f} m/s^2 and is still "
           "growing: that is integral windup. Clamp self.integral so the integral term stays a few m/s^2 at most.")
    ts, vs, accs = drive(make(mod, 2), lambda t, car: SpeedTarget(v=8.0, a=0.0), T=30.0)
    err = 8.0 - float(np.mean(vs[ts > 25]))
    if err > 0.06:
        expect(False, f"After 25 s at an 8 m/s target (P and I only) the car is still {err:.2f} m/s too slow. "
               "Hint: the integral should keep growing while there is an error, until the error is gone.")
    expect(err >= -0.06, f"After 25 s at an 8 m/s target (P and I only) the car is {-err:.2f} m/s too fast. "
           "Hint: the integral is too large or does not come back down; check your clamp and the sign.")
    ts, vs, accs = drive(make(mod, 2), lambda t, car: SpeedTarget(v=13.0, a=0.0), T=30.0)
    over = float(np.max(vs)) - 13.0
    expect(over < 1.5, f"Asked to go from 0 to 13 m/s (P and I only), the car overshot to {13 + over:.1f} m/s. "
           "Hint: integral windup; clamp the integral, or only add to it while the error is small.")
    wobble = float(np.std(accs[ts > 25]))
    expect(wobble < 0.3, f"The car oscillates at the end (acceleration std {wobble:.2f} m/s^2). Hint: ki may be too large.")
    return (f"no steady-state error ({err:+.3f} m/s), integral term bounded ({push:.1f} m/s^2 after 10 s), "
            f"overshoot {max(over, 0):.2f} m/s from 0 to 13 m/s")


def part3(mod):
    require(mod, 2)
    ctrl = mod.SpeedController()
    ff = ctrl.feedforward(SpeedTarget(v=5.0, a=1.0), CarState(v=5.0, pitch=0.08))
    want = 1.0 + 9.81 * math.sin(0.08) + 0.15
    expect(isinstance(ff, (int, float)) or hasattr(ff, "__float__"),
           f"feedforward() returned {ff!r}; it must return a number")
    expect(abs(ff - want) < 0.12, f"feedforward(a=1.0, pitch=0.08 rad, moving) returned {ff:.2f} m/s^2; "
           f"about {want:.2f} was expected. Hint: plan acceleration + gravity on the slope + rolling resistance.")
    grade = lambda x: 0.08  # noqa: E731
    ts, vs, _ = drive(make(mod, 3), lambda t, car: SpeedTarget(v=6.0, a=0.0), v0=6.0, grade=grade, T=15.0)
    worst = float(np.max(np.abs(vs - 6.0)))
    expect(worst < 0.35, f"Holding 6 m/s up an 8% hill, the speed dipped {worst:.2f} m/s off target. "
           "Hint: with gravity in the feedforward the controller should not need an error to push uphill.")
    return f"feedforward {ff:.2f} m/s^2 for the test case; on an 8% hill the speed stays within {worst:.2f} m/s"


def _stop_test(mod, grade=None, v0=8.0):
    """Approach a stop sign at 40 m, hold 3 s, drive off. Returns a dict of what happened."""
    road = Road(70.0, lambda x: v0, [Stop(40.0, 3.0, "stop sign")], v_start=v0)
    ctrl = mod.SpeedController()
    car = Car(v0, NOMINAL, grade=grade, seed=3)
    t, drove_off, near_since, creep = 0.0, None, None, 0.0
    while t < 40.0:
        meas = car.measure()
        car.apply(ctrl.compute(meas, road.target(car.x, meas.v)))
        car.step(0.02)
        t += 0.02
        stop = road.stops[0]
        road.update(car.x, car.v, 0.02)
        if not stop.released and near_since is None and abs(car.x - 40.0) <= 0.3 and car.v < 0.05:
            near_since = car.x
        if near_since is not None and not stop.released:
            creep = max(creep, abs(car.x - near_since))
        if stop.released and drove_off is None and car.v > 1.0:
            drove_off = t
    return {"stop": road.stops[0], "drove_off": drove_off, "creep": creep, "x": car.x}


def part4(mod):
    require(mod, 3)
    mod.SpeedController().stop_logic(0.0, CarState(v=1.0), SpeedTarget(v=1.0, a=0.0, stop_distance=3.0))
    for label, grade in (("on flat ground", None), ("on an 8% hill", lambda x: 0.08)):
        r = _stop_test(mod, grade=grade, v0=8.0 if grade is None else 6.0)
        stop = r["stop"]
        expect(not stop.overrun, f"{label}: the car rolled through the stop point at 40 m.")
        expect(stop.released, f"{label}: the car never came to rest within 1.5 m of the stop point "
               f"(it ended at {r['x']:.2f} m). Hint: if you stop short, creep forward; on a hill, the creep "
               "request must also beat gravity.")
        err = stop.error
        expect(abs(err) < 0.3, f"{label}: the car was released {err:+.2f} m from the stop point at 40 m "
               "(want within 0.3 m). Hint: near the line, slow at v^2 / (2 d), counting what gravity and "
               "rolling resistance already do, and creep up if you stop short.")
        expect(r["creep"] < 0.15, f"{label}: while waiting at the line the car moved {r['creep']:.2f} m. "
               "Hint: hold the brake while stopped at the line, and do not let the integral wind up while you wait.")
        expect(r["drove_off"] is not None, f"{label}: after the stop was released the car never drove off again.")
    return "stops within 0.3 m of the line on flat ground and on an 8% hill, holds still, and drives off"


def part5(mod_path):
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
        log = run(scn, mod.SpeedController)
        s = score(log, metrics(log))["total"]
        passed = log.completed and s > 0
        ok &= passed
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'} ({s:.0f}/100){'' if passed else ' ' + log.message}")
    expect(ok, "Not every core scenario passes yet:\n      " + "\n      ".join(lines))
    return "all core scenarios pass: " + "; ".join(lines)


PARTS = {"part1": part1, "part2": part2, "part3": part3, "part4": part4, "part5": part5}


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
            msg = PARTS[name](args.controller if name == "part5" else load(args.controller))
            print(f"{GREEN}PASS{END} {name}: {msg}")
            n_ok += 1
        except NotImplementedError as e:
            print(f"{DIM}TODO{END} {name}: {e}")
        except CheckFailed as e:
            print(f"{RED}FAIL{END} {name}: {e}")
        except Exception as e:  # a bug in their code: show it plainly
            print(f"{RED}ERROR{END} {name}: {type(e).__name__}: {e}")
    print(f"\n{n_ok}/{len(names)} checks passed.")
    return 0 if n_ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
