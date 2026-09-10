"""Checkpoints for the three parts of planner.py.

    python check.py              # check every part
    python check.py part1        # just one part: part1, part2, part3, or part4
    python check.py part1 part3  # several

Parts 1 to 3 are quick unit tests of your functions (no simulation).  Part 4
drives the three core scenarios with your whole planner and tells you
PASS/FAIL for each.  When a check fails you get what we expected, what you
returned, and a hint.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

import numpy as np                                                        # noqa: E402

from common.geometry import TrackBuilder                                   # noqa: E402
from common.messages import ObjClass                                       # noqa: E402
from sim.world import (DOUBLE_YELLOW, SOLID_WHITE, LaneLineSpan, RouteMap,  # noqa: E402
                       StopLine)
from tools import TrackedObject                                            # noqa: E402

TOL = 0.02
LANE_OFFSET_MAX = 0.7    # [m] how far from the lane center the car's center may go


class NotStarted(Exception):
    pass


def load_planner(path):
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(path))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


def straight_route(length=220.0, stops=()):
    track = TrackBuilder(yaw0=0.3).straight(length).build()
    lines = [LaneLineSpan("left", 0.0, length + 10, DOUBLE_YELLOW),
             LaneLineSpan("right", 0.0, length + 10, SOLID_WHITE)]
    return RouteMap(reference=track, lane_width=3.6, lane_lines=lines,
                    left_lane="opposite_direction", speed_limits=[],
                    stop_lines=list(stops), goal_s=length - 10.0)


def call(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        raise NotStarted()


def as_array(value, n, name, what):
    """(array, None) if ``value`` is n numbers, else (None, (message, hint))."""
    if value is None:
        return None, (f"{name} returned None", "did you forget to `return` your array?")
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, (f"{name} returned a {type(value).__name__}", f"return a numpy array of {what}")
    if arr.ndim == 0:
        return None, (f"{name} returned a single number, not one {what[:-1]} per station",
                      f"return a numpy array with one {what[:-1]} per station")
    if arr.shape != (n,):
        return None, (f"{name} returned {arr.shape[0]} {what} for {n} stations",
                      f"return one {what[:-1]} per station")
    return arr, None


# ---------------------------------------------------------------------- #
# Part 1
# ---------------------------------------------------------------------- #
def _profile_rules(mod, s, v_limit, curv, v0, stop_s, v, ideal=None):
    """Return (message, hint) for the first rule broken, or None."""
    v, bad = as_array(v, len(s), "speed_profile", "speeds")
    if bad:
        return bad
    if not np.all(np.isfinite(v)):
        hint = ("curvature is negative in right turns, so use abs(curvature), and do not divide by a "
                "curvature of 0" if np.any(np.asarray(curv) < 0) else
                "a curvature of 0 means a straight road: do not divide by it (use a tiny floor), "
                "and never take sqrt of a negative number")
        return ("your speeds contain NaN or inf", hint)
    if abs(v[0] - v0) > 1e-6:
        return (f"v[0] is {v[0]:.2f} but the car is doing v0 = {v0:.2f} m/s", "rule 1: set v[0] = v0 at the end")
    if np.any(v < -1e-9):
        i = int(np.argmin(v))
        return (f"negative speed {v[i]:.2f} at s={s[i]:.1f}", "speeds can never be negative")
    lim = np.minimum(v_limit, np.sqrt(mod.A_LAT_MAX / np.maximum(np.abs(curv), 1e-9)))
    over = v[1:] - lim[1:]
    if np.any(over > TOL):
        i = int(np.argmax(over)) + 1
        if v[i] > v_limit[i] + TOL:
            return (f"at s={s[i]:.1f} you planned {v[i]:.2f} m/s but the speed limit there is {v_limit[i]:.1f}",
                    "rule 2: cap every station at v_limit")
        return (f"at s={s[i]:.1f} the curvature is {curv[i]:.3f} 1/m, so the car can go at most "
                f"sqrt({mod.A_LAT_MAX} / {abs(curv[i]):.3f}) = {lim[i]:.2f} m/s there; you planned {v[i]:.2f}",
                "rule 2: cap each station at sqrt(A_LAT_MAX / |curvature|)")
    ds = np.diff(s)
    a = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)
    hi = np.nonzero(a > mod.ACCEL + TOL)[0]
    if len(hi):
        i = int(hi[0])
        return (f"your profile speeds up at {a[i]:.2f} m/s^2 between s={s[i]:.1f} and s={s[i + 1]:.1f}; "
                f"the limit is ACCEL = {mod.ACCEL}",
                "rule 3: forward pass, v[i] <= sqrt(v[i-1]**2 + 2*ACCEL*ds)")
    lo = np.nonzero(-a[1:] > mod.DECEL + TOL)[0]      # the first step may be infeasible
    if len(lo):
        i = int(lo[0]) + 1
        return (f"your profile slows down at {-a[i]:.2f} m/s^2 between s={s[i]:.1f} and s={s[i + 1]:.1f}; "
                f"the comfortable limit is DECEL = {mod.DECEL}",
                "rule 3: start braking earlier with a backward pass, v[i] <= sqrt(v[i+1]**2 + 2*DECEL*ds)")
    if stop_s is not None:
        after = s >= stop_s - 1e-9
        if np.any(v[after] > 1e-6):
            i = int(np.nonzero(after & (v > 1e-6))[0][0])
            return (f"the car must be stopped at s={stop_s:.1f}, but you planned {v[i]:.2f} m/s at s={s[i]:.1f}",
                    "rule 4: speed is 0 at stop_s and at every station after it")
    if ideal is not None:
        slow = ideal - v
        if np.any(slow > 0.3 + 0.05 * ideal):
            i = int(np.argmax(slow))
            return (f"at s={s[i]:.1f} you planned {v[i]:.2f} m/s, but the car could safely do {ideal[i]:.2f} there",
                    "rule 5: go as fast as the rules allow (brake as late as you can, speed up as early as you can)")
    return None


def part1(mod):
    A, D, L = mod.ACCEL, mod.DECEL, mod.A_LAT_MAX
    cases = []

    s = np.arange(0.0, 60.0 + 1e-9, 0.5)
    lim = np.full(len(s), 10.0)
    ideal = np.minimum(10.0, np.sqrt(2 * A * s))
    cases.append(("flat road, starting from a standstill", s, lim, np.zeros(len(s)), 0.0, None, ideal))

    s = np.arange(0.0, 40.0 + 1e-9, 0.5)
    lim = np.full(len(s), 10.0)
    ideal = np.minimum.reduce([lim, np.sqrt(64 + 2 * A * s), np.sqrt(2 * D * (40.0 - s))])
    ideal[0] = 8.0
    cases.append(("stop line 40 m ahead at 8 m/s", s, lim, np.zeros(len(s)), 8.0, 40.0, ideal))

    s = np.arange(0.0, 70.0 + 1e-9, 0.5)
    curv = np.where((s >= 30) & (s <= 45), -0.1, 0.0)             # a right turn: curvature < 0
    vc = math.sqrt(L / 0.1)
    ideal = np.minimum.reduce([np.full(len(s), 10.0), np.sqrt(64 + 2 * A * s),
                               np.where(s < 30, np.sqrt(vc ** 2 + 2 * D * np.maximum(30 - s, 0)), np.inf),
                               np.where(s > 45, np.sqrt(vc ** 2 + 2 * A * np.maximum(s - 45, 0)), np.inf),
                               np.where((s >= 30) & (s <= 45), vc, np.inf)])
    cases.append(("a tight right turn (radius 10 m) from s=30 to 45", s, np.full(len(s), 10.0), curv, 8.0, None, ideal))

    s = np.arange(0.0, 60.0 + 1e-9, 0.5)
    lim = np.where(s < 30, 10.0, 5.0)
    ideal = np.minimum.reduce([lim, np.where(s < 30, np.sqrt(25 + 2 * D * np.maximum(30 - s, 0)), 5.0)])
    ideal[0] = 10.0
    cases.append(("the speed limit drops from 10 to 5 m/s at s=30", s, lim, np.zeros(len(s)), 10.0, None, ideal))

    s = np.cumsum(np.r_[0.0, np.tile([0.3, 0.7, 0.45, 0.55], 12)])   # stations are not always 0.5 m apart
    lim = np.full(len(s), 10.0)
    ideal = np.minimum.reduce([lim, np.sqrt(2 * A * s), np.sqrt(2 * D * (s[-1] - s))])
    cases.append(("uneven station spacing, stopping at the end (use each ds, not 0.5)", s, lim,
                  np.zeros(len(s)), 0.0, float(s[-1]), ideal))

    s = np.array([0.0, 0.5, 1.0])
    cases.append(("already stopped at the stop point (must not creep)", s, np.full(3, 10.0), np.zeros(3), 0.0, 0.0, None))

    s = np.arange(0.0, 5.0 + 1e-9, 0.5)
    cases.append(("a stop point only 5 m away at 6 m/s (too close to be comfortable)", s,
                  np.full(len(s), 10.0), np.zeros(len(s)), 6.0, 5.0, None))

    passed = 0
    for name, s, lim, curv, v0, stop_s, ideal in cases:
        v = call(mod.speed_profile, s.copy(), lim.copy(), curv.copy(), v0, stop_s)
        if name.startswith("a stop point only"):
            v, bad = as_array(v, len(s), "speed_profile", "speeds")
            if bad:
                pass
            elif not np.all(np.isfinite(v)):
                bad = ("your speeds contain NaN or inf", "check your square roots and divisions")
            elif abs(v[0] - v0) > 1e-6:
                bad = (f"v[0] is {v[0]:.2f} but the car is doing {v0}",
                       "rule 1: the first speed is always the car's current speed; set v[0] = v0 at the end")
            elif v[-1] > 1e-6:
                bad = (f"the car must be stopped at s=5.0 but you planned {v[-1]:.2f} m/s there",
                       "rule 4 wins even when you cannot stop comfortably: the car brakes harder")
        else:
            bad = _profile_rules(mod, s, lim, curv, v0, stop_s, v, ideal)
        passed += report(name, bad)
    return passed, len(cases)


# ---------------------------------------------------------------------- #
# Part 2
# ---------------------------------------------------------------------- #
def _check_stop_point(r, line, front):
    if r is None:
        return (f"you returned None while a stop sign is {line - 10.0:.0f} m ahead",
                "return the s where the rear axle should stop, for example line.s - 1.0 - FRONT")
    if not isinstance(r, (int, float, np.floating)) or not math.isfinite(r):
        return (f"you returned {r!r}", "return a number (the rear axle's stop s) or None")
    gap = line - (r + front)
    where = f"{gap:.1f} m before the line" if gap >= 0 else f"{-gap:.1f} m past the line"
    if not (0.0 <= gap <= 3.0):
        return (f"your stop point puts the front bumper {where}; it must be 0 to 3 m before it "
                "(the grader takes points off beyond 3 m and fails you beyond 4 m)",
                "the car's position is the rear axle: stop_s = line.s - 1.0 - FRONT puts the bumper 1 m back")
    return None


def part2(mod):
    FRONT = getattr(mod, "FRONT", 3.7)
    lines = [StopLine(1, 60.0, "stop_sign"), StopLine(2, 110.0, "traffic_light"),
             StopLine(3, 150.0, "stop_sign")]
    route = straight_route(stops=lines)
    passed, total = 0, 5

    # 1. approaching
    logic = mod.StopSignLogic()
    r = call(logic.update, 0.0, 10.0, 8.0, route)
    bad = _check_stop_point(r, 60.0, FRONT)
    passed += report("gives a stop point for the stop sign ahead", bad)
    if bad:
        return passed, total

    # 2. approach, creep the last few cm (like a real car), stop, hold 3 s, release
    t, s_car, v = 0.0, 10.0, 8.0
    bad = None
    creep_from = r - 0.4
    while s_car < r - 1e-9 and bad is None:
        t += 0.1
        if s_car < creep_from:
            v = max(0.3, math.sqrt(max(2 * 1.5 * (creep_from - s_car), 0.0)))
            s_car = min(creep_from, s_car + v * 0.1)
        else:
            v = 0.2                                   # rolling slowly, but not stopped
            s_car = min(r, s_car + v * 0.1)
        out = logic.update(t, s_car, v, route)
        if out is None or abs(out - r) > 0.3:
            bad = (f"while approaching (car at s={s_car:.2f}, {v:.1f} m/s) your stop point changed from "
                   f"{r:.1f} to {out}", "keep returning the same stop point until the stop is done "
                   "(and remember 'stopped' means below 0.05 m/s)")
    t_still, released_after = t + 0.1, None          # the next call is the first with v = 0
    while bad is None and t - t_still < 6.0:
        t += 0.1
        out = logic.update(t, r, 0.0, route)
        if out is None or out > r + 1.0:              # released (maybe already aiming at the next sign)
            released_after = t - t_still               # seconds of standing still before the release
            break
    if bad is None:
        if released_after is None:
            bad = ("the car has been standing still at the line for 6 s and you still tell it to stop",
                   f"after STOP_HOLD_S = {getattr(mod, 'STOP_HOLD_S', 3.0)} s of standing still, return None")
        elif released_after < 3.0 - 1e-6:
            bad = (f"you released the stop after {released_after:.1f} s of standing still; the rule is 3 s",
                   "start the clock at the first update where v < 0.05 m/s (it crept in at 0.2 m/s "
                   "before that), and release once t minus that time reaches STOP_HOLD_S")
    passed += report("holds a full stop for 3 s, then lets the car go", bad)

    # 3. drive on: no second stop for the same sign, ignore the traffic light, next sign works
    bad = None
    if released_after is not None:
        s_car, v = r, 0.0
        while s_car < 100.0 and bad is None:
            t += 0.1
            v = min(8.0, v + 0.15)
            s_car += v * 0.1
            out = logic.update(t, s_car, v, route)
            if out is not None and out < 140.0:
                bad = (f"after the stop, with the car at s={s_car:.1f}, you asked it to stop at s={out:.1f}",
                       "remember finished signs (a set of ids) and ignore lines behind you and "
                       "lines of other kinds, like the traffic light at s=110")
        if bad is None:
            out = logic.update(t + 0.1, 101.0, 8.0, route)
            if out is None:
                bad = ("with the car at s=101 you returned None, but there is another stop sign at s=150",
                       "keep looking for the next stop sign ahead that is not done yet")
            elif _check_stop_point(out, 150.0, FRONT):
                bad = (f"for the second stop sign (line at s=150) you asked to stop at s={out:.1f}",
                       "stop_s = line.s - 1.0 - FRONT, for the nearest unfinished stop sign ahead")
    else:
        bad = ("skipped because the previous check failed", "fix that one first")
    passed += report("drives on afterward, and handles the next stop sign", bad)

    # 4. stopping far from the line does not count
    logic = mod.StopSignLogic()
    call(logic.update, 0.0, 10.0, 6.0, route)
    far = 60.0 - FRONT - 8.0
    bad = None
    t = 0.0
    for _ in range(50):
        t += 0.1
        out = logic.update(t, far, 0.0, route)
        if out is None:
            bad = ("the car waited 8 m before the line and you let it go; that stop does not count",
                   "only count time while stopped with the front bumper within 4 m of the line")
            break
    passed += report("does not count a stop made 8 m before the line", bad)

    # 5. moving again restarts the count
    logic = mod.StopSignLogic()
    r = call(logic.update, 0.0, 40.0, 5.0, route)
    bad = _check_stop_point(r, 60.0, FRONT)
    if bad is None:
        t = 0.0
        seq = [(r - 1.0, 0.0)] * 15 + [(r - 1.0 + 0.05 * k, 0.5) for k in range(1, 11)] + [(r, 0.0)] * 25
        for i, (sc, vv) in enumerate(seq):
            t += 0.1
            out = logic.update(t, sc, vv, route)
            if out is None:
                bad = ("the car stopped 1.5 s, crept forward, then stopped again, and you released it "
                       f"after only {max(0.0, (i - 24) * 0.1):.1f} s of the second stop",
                       "if the car starts moving, the 3 s count starts over")
                break
    passed += report("restarts the 3 s count if the car moves", bad)
    return passed, total


# ---------------------------------------------------------------------- #
# Part 3
# ---------------------------------------------------------------------- #
def obstacle(oid, cls, s_mid, d_mid, length, width):
    return TrackedObject(oid, cls, 0.0, 0.0, length, width, s=s_mid, d=d_mid,
                         s_min=s_mid - length / 2, s_max=s_mid + length / 2,
                         d_min=d_mid - width / 2, d_max=d_mid + width / 2, hits=10)


def _offset_rules(mod, s, d, obstacles, center_ranges):
    FRONT, REAR = getattr(mod, "FRONT", 3.7), getattr(mod, "REAR", 0.8)
    d, bad = as_array(d, len(s), "choose_offset", "offsets")
    if bad:
        return bad
    if not np.all(np.isfinite(d)):
        return ("your offsets contain NaN or inf", "check your divisions")
    if np.max(np.abs(d)) > LANE_OFFSET_MAX + 1e-6:
        i = int(np.argmax(np.abs(d)))
        side = "outside the lane" if abs(d[i]) > 0.9 else "too close to the lane line"
        return (f"at s={s[i]:.1f} your offset is {d[i]:+.2f} m, so part of the car is {side}",
                f"rule 1: keep |d| <= {LANE_OFFSET_MAX} so the car's corners stay in the lane even mid-shift")
    for ob in obstacles:
        if ob.is_pedestrian or ob.d_max < -1.8 or ob.d_min > 1.8:
            continue
        beside = (s + FRONT >= ob.s_min) & (s - REAR <= ob.s_max)
        for i in np.nonzero(beside)[0]:
            gap = max(ob.d_min - (d[i] + 0.9), (d[i] - 0.9) - ob.d_max)
            if gap < 0.5 - 1e-6:
                return (f"with the car at s={s[i]:.1f} (d={d[i]:+.2f}) its side is {gap:.2f} m from the "
                        f"{ob.name} (at s={ob.s_min:.1f} to {ob.s_max:.1f}); it needs 0.5 m or more",
                        "rule 2: be fully shifted before the front of the car reaches s_min, and stay "
                        "shifted until the rear is past s_max")
    ds = np.diff(s)
    slope = np.diff(d) / ds
    if np.any(np.abs(slope) > 0.3):
        i = int(np.argmax(np.abs(slope)))
        return (f"between s={s[i]:.1f} and s={s[i + 1]:.1f} your offset changes by {slope[i] * ds[i]:+.2f} m, "
                "a sideways jump the car cannot steer",
                "rule 4: use lateral_transition() over several meters")
    curv = np.diff(d, 2) / (ds[:-1] * ds[1:])
    if np.any(np.abs(curv) > 0.1):
        i = int(np.argmax(np.abs(curv))) + 1
        return (f"at s={s[i]:.1f} your offsets bend with curvature {abs(curv[i]):.2f} 1/m; the car's limit "
                "is 0.21 and anything above 0.1 is a harsh swerve",
                "rule 4: make the shift longer (transition_length() suggests how long)")
    for a, b in center_ranges:
        sel = (s >= a) & (s <= b)
        if np.any(np.abs(d[sel]) > 0.05):
            i = int(np.nonzero(sel & (np.abs(d) > 0.05))[0][0])
            return (f"at s={s[i]:.1f} your offset is {d[i]:+.2f} m, but nothing there needs a nudge",
                    "rules 3 and 5: ignore pedestrians and anything outside your lane, and come back "
                    "to d = 0 once you are past an obstacle")
    return None


def part3(mod):
    route = straight_route()
    s = np.arange(0.0, 110.0 + 1e-9, 0.5)
    right = obstacle(101, ObjClass.BARREL, 40.0, -1.05, 0.6, 0.6)      # 1.05 m into the lane
    left = obstacle(102, ObjClass.CONE, 40.0, 1.2, 0.4, 0.4)           # 0.8 m into the lane
    shoulder = obstacle(103, ObjClass.CONE, 40.0, -2.5, 0.4, 0.4)
    ped = obstacle(104, ObjClass.PEDESTRIAN, 60.0, -0.5, 0.5, 0.5)   # in the lane: not Part 3's job
    b1 = obstacle(105, ObjClass.BARREL, 30.0, -1.15, 0.6, 0.6)
    b2 = obstacle(106, ObjClass.BARREL, 95.0, 1.15, 0.6, 0.6)
    cases = [
        ("no obstacles: stay in the middle", [], [(0, 110)]),
        ("a barrel 1.05 m into the lane on the right (clip to the lane margin)", [right], [(0, 5), (80, 110)]),
        ("a cone 0.8 m into the lane on the left", [left], [(0, 5), (80, 110)]),
        ("a cone on the shoulder and a pedestrian: ignore them", [shoulder, ped], [(0, 110)]),
        ("two barrels far apart, one on each side", [b1, b2], [(0, 2), (60, 64)]),
    ]
    passed = 0
    for name, obs, centers in cases:
        d = call(mod.choose_offset, s.copy(), list(obs), route)
        passed += report(name, _offset_rules(mod, s, d, obs, centers))
    # the same barrel, planned again once the car has moved on to s=30
    s_late = np.arange(30.0, 140.0 + 1e-9, 0.5)
    a1, bad = as_array(call(mod.choose_offset, s.copy(), [right], route), len(s), "choose_offset", "offsets")
    a2, bad2 = as_array(call(mod.choose_offset, s_late.copy(), [right], route), len(s_late),
                        "choose_offset", "offsets")
    bad = bad or bad2
    if bad is None:
        both = s_late[s_late <= s[-1]]
        diff = np.abs(np.interp(both, s, a1) - np.interp(both, s_late, a2))
        if diff.max() > 0.02:
            i = int(np.argmax(diff))
            bad = (f"at s={both[i]:.1f} you planned {np.interp(both[i], s, a1):+.2f} m when the car was at "
                   f"s=0, but {np.interp(both[i], s_late, a2):+.2f} m once it reached s=30",
                   "the plan must not depend on where the car is: tie each shift to the obstacle's "
                   "s_min and s_max, not to s[0]")
        else:
            bad = _offset_rules(mod, s_late, a2, [right], [(80, 140)])
    passed += report("the same plan when the car is further along", bad)
    # sensitivity to perception noise
    moved = obstacle(101, ObjClass.BARREL, 40.05, -1.0, 0.62, 0.62)
    d1, b1 = as_array(call(mod.choose_offset, s.copy(), [right], route), len(s), "choose_offset", "offsets")
    d2, b2 = as_array(call(mod.choose_offset, s.copy(), [moved], route), len(s), "choose_offset", "offsets")
    bad = b1 or b2
    change = float(np.mean(np.abs(d2 - d1))) if bad is None else 0.0
    if bad is None and change > 0.05:
        bad = (f"when the barrel's estimate moves by 5 cm, your offsets move by {change:.2f} m on average",
               "small changes in the input should give small changes in the plan: the grader "
               "penalizes a plan that moves more than 5 cm per tick")
    passed += report("small perception noise gives a small change", bad)
    return passed, len(cases) + 2


# ---------------------------------------------------------------------- #
# Part 4
# ---------------------------------------------------------------------- #
def part4(planner_path, seed=0):
    from sim.grading import grade
    from sim.runner import run_scenario
    from sim.scenarios import SCENARIOS
    mod = load_planner(planner_path)
    names = [n for n, fn in SCENARIOS.items() if fn(0).tier == "core"]
    passed, missing = 0, set()
    for name in names:
        res = run_scenario(SCENARIOS[name](seed), mod.Planner)
        g = grade(res)
        missing.update(res.planner_missing)
        ok = g.status == "PASS"
        passed += ok
        note = f"score {g.score:.1f}" + (f", biggest penalties: {g.top_penalties(2)}" if g.top_penalties(2) else "")
        run = os.path.relpath(os.path.join(HERE, "run.py"))
        png = os.path.relpath(os.path.join(HERE, "results", f"{name}.png"))
        report(f"{name} (seed {seed})", None if ok else (g.reason, f"run `python {run} --scenario "
                                                                f"{name}` and look at {png}"),
               extra=note if ok else None)
    if missing:
        print("       (still using the naive fallback for: " + ", ".join(sorted(missing)) + ")")
    return passed, len(names)


# ---------------------------------------------------------------------- #
def report(name, bad, extra=None) -> int:
    if bad is None:
        print(f"  [ok] {name}" + (f"   ({extra})" if extra else ""))
        return 1
    msg, hint = bad
    print(f"  [X]  {name}\n       {msg}\n       hint: {hint}")
    return 0


PARTS = {"part1": ("Part 1: speed_profile", part1),
         "part2": ("Part 2: StopSignLogic", part2),
         "part3": ("Part 3: choose_offset", part3),
         "part4": ("Part 4: the core scenarios, with your whole planner", None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("parts", nargs="*", help="part1, part2, part3, part4 (default: all)")
    ap.add_argument("--planner", default=os.path.join(HERE, "planner.py"))
    ap.add_argument("--seed", type=int, default=0, help="seed for part4's scenarios (default 0)")
    args = ap.parse_args(argv)
    todo = args.parts or list(PARTS)
    unknown = [p for p in todo if p not in PARTS]
    if unknown:
        ap.error(f"unknown part {unknown[0]!r}; choose from {', '.join(PARTS)}")
    try:
        mod = load_planner(args.planner)
    except Exception:
        print(f"Could not load {os.path.relpath(args.planner)}; Python says:\n")
        print("    " + traceback.format_exc(limit=0).strip().replace("\n", "\n    "))
        return 1
    all_ok = True
    for key in todo:
        title, fn = PARTS[key]
        print(f"\n{title}")
        needed = {"part1": "speed_profile", "part2": "StopSignLogic", "part3": "choose_offset"}.get(key)
        if needed and not hasattr(mod, needed):
            print(f"  planner.py has no `{needed}`, so this part cannot be checked "
                  "(fine if you restructured it for the stretch; part4 still works)")
            all_ok = False
            continue
        try:
            if key == "part4":
                p, n = part4(args.planner, args.seed)
            else:
                p, n = fn(mod)
        except NotStarted:
            print("  TODO: not started yet (it still raises NotImplementedError).  Read its docstring in planner.py.")
            all_ok = False
            continue
        except Exception:
            print("  your code raised an exception:\n")
            print("    " + traceback.format_exc().strip().replace("\n", "\n    "))
            all_ok = False
            continue
        all_ok &= p == n
        print(f"  {p}/{n} checks passed" + (", nice!" if p == n else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
