"""Checks a plan against the rules and measures it.

A plan is a list of ReferenceTrajectory segments.  Each segment is driven in
one direction (``reverse`` flag) and point ``yaw`` is the car's heading.
Consecutive segments must share a pose, consecutive segments in the same
direction are merged into one run, and every run is replayed every 0.05 m.

Kinematic rules (a car cannot slide sideways or turn on the spot):
* over every 0.5 m of a run (or the whole run if shorter), the car moves
  along its heading (or against it when reversing) within 4 deg;
* over every 0.5 m of TOTAL travel, across segment boundaries and direction
  changes too, the heading changes no faster than the steering allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from common import vehicle as V
from common.geometry import wrap_to_pi
from common.messages import ReferenceTrajectory
from .lot import CENTER_OFFSET, ParkingProblem, _sat_overlap, footprints

STEP = 0.05                 # [m] replay resolution
MAX_SPACING = 0.5           # [m] between consecutive points
START_TOL = (0.10, 3.0)     # [m], [deg]
JOIN_TOL = (0.01, 0.5)      # [m], [deg]: consecutive segments must share a pose
WINDOW = 0.5                # [m] window for direction and curvature checks
SLIP_TOL_DEG = 4.0          # motion direction vs heading (worst honest case ~3 deg: 0.5 m spacing on a full-lock arc)
SLIP_SLACK = 0.001          # [m] lateral allowance per window (numerical noise on tiny runs)
HEADING_SLACK = math.radians(0.5)   # heading allowance per window of the global heading-rate check
CURV_LIMIT = V.MAX_CURVATURE * 1.03
FINAL_HEADING_TOL_DEG = 5.0
PLAN_TIME_LIMIT = 30.0      # [s]
STEER_JUMP = 0.08           # [1/m] curvature change within WINDOW that counts as steering in place


@dataclass
class CheckResult:
    passed: bool = False
    reason: str = ""
    fail_xy: Optional[Tuple[float, float]] = None
    fail_pose: Optional[Tuple[float, float, float]] = None   # the car pose at first contact
    dense: List[dict] = field(default_factory=list)      # per segment: x, y, yaw, s, reverse
    cusps: List[Tuple[float, float]] = field(default_factory=list)
    final_pose: Optional[Tuple[float, float, float]] = None
    metrics: dict = field(default_factory=dict)

    def fail(self, reason: str, x: Optional[float] = None, y: Optional[float] = None) -> "CheckResult":
        self.passed, self.reason = False, reason
        if x is not None:
            self.fail_xy = (float(x), float(y))
        return self


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersection of two convex counter-clockwise polygons (Sutherland-Hodgman)."""
    out = list(subject)
    for k in range(len(clip)):
        a, b = clip[k], clip[(k + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        for m in range(len(inp)):
            p, q = inp[m], inp[(m + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
    return np.array(out)


def _contact_point(problem: ParkingProblem, x: float, y: float, yaw: float) -> Tuple[float, float]:
    """Where the car at this pose touches something: the middle of the overlap."""
    car = footprints([x], [y], [yaw])[0]
    if len(problem._boxes):
        hits = np.nonzero(_sat_overlap(car[None], problem._boxes)[0])[0]
        for j in hits:
            poly = _clip(car, problem._boxes[j])
            if len(poly):
                c = poly.mean(axis=0)
                return float(c[0]), float(c[1])
    xmin, xmax, ymin, ymax = problem.bounds
    out = [p for p in car if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)]
    c = np.mean(out, axis=0) if out else car.mean(axis=0)
    return float(c[0]), float(c[1])


def _densify(x, y, yaw) -> Tuple[np.ndarray, ...]:
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    if s[-1] < 1e-9:
        return x[:1], y[:1], yaw[:1], s[:1]
    sd = np.append(np.arange(0.0, s[-1], STEP), s[-1])
    yu = np.unwrap(yaw)
    return np.interp(sd, s, x), np.interp(sd, s, y), np.interp(sd, s, yu), sd


def _windows(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) with s[j] - s[i] just >= WINDOW (or the whole segment if shorter)."""
    if s[-1] < WINDOW:
        return np.array([0]), np.array([len(s) - 1])
    j = np.searchsorted(s, s + WINDOW - 1e-9)
    ok = j < len(s)
    return np.nonzero(ok)[0], j[ok]


def _curvature_profile(s: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Curvature from heading change over a centered WINDOW (NaN near the ends)."""
    h = 0.5 * WINDOW
    k = np.full(len(s), np.nan)
    if s[-1] < WINDOW:
        return k
    lo = np.searchsorted(s, s - h)
    hi = np.searchsorted(s, s + h)
    ok = (s - h >= -1e-9) & (s + h <= s[-1] + 1e-9) & (hi < len(s))
    idx = np.nonzero(ok)[0]
    k[idx] = (yaw[hi[idx]] - yaw[lo[idx]]) / np.maximum(s[hi[idx]] - s[lo[idx]], 1e-9)
    return k


def _count_steer_events(s: np.ndarray, kappa: np.ndarray) -> int:
    valid = ~np.isnan(kappa)
    if valid.sum() < 2:
        return 0
    sv, kv = s[valid], kappa[valid]
    j = np.searchsorted(sv, sv + WINDOW - 1e-9)
    ok = j < len(sv)
    jump = np.zeros(len(sv), dtype=bool)
    jump[np.nonzero(ok)[0]] = np.abs(kv[j[ok]] - kv[ok]) > STEER_JUMP
    # count runs of consecutive True as one event
    return int(np.sum(jump[1:] & ~jump[:-1]) + (1 if jump[0] else 0))


def check(problem: ParkingProblem, plan, plan_time: float) -> CheckResult:
    r = CheckResult()
    r.metrics["plan_time_s"] = float(plan_time)
    if plan_time > PLAN_TIME_LIMIT:
        return r.fail(f"planning took {plan_time:.1f} s (limit {PLAN_TIME_LIMIT:.0f} s)")
    if isinstance(plan, ReferenceTrajectory):
        plan = [plan]
    if not isinstance(plan, (list, tuple)) or not plan:
        return r.fail("plan() must return a non-empty list of ReferenceTrajectory")

    # ---------------- parse segments ----------------
    segs = []
    for k, seg in enumerate(plan):
        if not isinstance(seg, ReferenceTrajectory):
            return r.fail(f"segment {k} is a {type(seg).__name__}, not a ReferenceTrajectory")
        if len(seg.points) < 2:
            return r.fail(f"segment {k} has fewer than 2 points")
        a = seg.as_arrays()
        x, y, yaw = a["x"], a["y"], a["yaw"]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(yaw))):
            return r.fail(f"segment {k} contains NaN or inf")
        step = np.hypot(np.diff(x), np.diff(y))
        if np.any(step > MAX_SPACING + 1e-9):
            i = int(np.argmax(step))
            return r.fail(f"segment {k}: points {step[i]:.2f} m apart (max {MAX_SPACING} m)", x[i], y[i])
        keep = np.concatenate(([True], step > 1e-6))
        x, y, yaw = x[keep], y[keep], yaw[keep]
        if len(x) < 2:
            # zero-length segment: only its heading matters for continuity
            segs.append({"x": x, "y": y, "yaw": yaw, "reverse": bool(seg.reverse), "empty": True})
        else:
            segs.append({"x": x, "y": y, "yaw": yaw, "reverse": bool(seg.reverse), "empty": False})

    # ---------------- start and continuity ----------------
    st = problem.start
    x0, y0, h0 = segs[0]["x"][0], segs[0]["y"][0], segs[0]["yaw"][0]
    d0 = math.hypot(x0 - st.x, y0 - st.y)
    e0 = abs(math.degrees(float(wrap_to_pi(h0 - st.yaw))))
    if d0 > START_TOL[0] or e0 > START_TOL[1]:
        return r.fail(f"first point is {d0:.2f} m / {e0:.1f} deg from the start pose", x0, y0)
    for k in range(len(segs) - 1):
        a, b = segs[k], segs[k + 1]
        dj = math.hypot(a["x"][-1] - b["x"][0], a["y"][-1] - b["y"][0])
        ej = abs(math.degrees(float(wrap_to_pi(a["yaw"][-1] - b["yaw"][0]))))
        if dj > JOIN_TOL[0] or ej > JOIN_TOL[1]:
            return r.fail(f"segment {k + 1} does not start where segment {k} ends ({dj:.3f} m, {ej:.2f} deg; "
                          f"they must share a pose within {JOIN_TOL[0] * 100:.0f} cm and {JOIN_TOL[1]} deg)",
                          b["x"][0], b["y"][0])
    segs = [sg for sg in segs if not sg["empty"]]
    if not segs:
        return r.fail("the plan does not move")

    # ---------------- merge same-direction segments into runs ----------------
    runs = []
    for sg in segs:
        if runs and runs[-1]["reverse"] == sg["reverse"]:
            prev = runs[-1]
            yaw = sg["yaw"] + (prev["yaw"][-1] - sg["yaw"][0]) - float(wrap_to_pi(prev["yaw"][-1] - sg["yaw"][0]))
            same = math.hypot(sg["x"][0] - prev["x"][-1], sg["y"][0] - prev["y"][-1]) < 1e-6
            k0 = 1 if same else 0
            prev["x"] = np.concatenate((prev["x"], sg["x"][k0:]))
            prev["y"] = np.concatenate((prev["y"], sg["y"][k0:]))
            prev["yaw"] = np.concatenate((prev["yaw"], yaw[k0:]))
        else:
            runs.append({"x": sg["x"], "y": sg["y"], "yaw": np.unwrap(sg["yaw"]), "reverse": sg["reverse"]})

    # ---------------- per run: the car moves along its heading ----------------
    steer_events = 0
    tan_tol = math.tan(math.radians(SLIP_TOL_DEG))
    for run in runs:
        x, y, yaw, s = _densify(run["x"], run["y"], run["yaw"])
        run.update(x=x, y=y, yaw=yaw, s=s)
        r.dense.append({"x": x, "y": y, "yaw": yaw, "s": s, "reverse": run["reverse"]})
        if s[-1] < 1e-6:
            continue
        i, j = _windows(s)
        dx, dy = x[j] - x[i], y[j] - y[i]
        ymid = np.interp(0.5 * (s[i] + s[j]), s, yaw)
        sign = -1.0 if run["reverse"] else 1.0
        lon = sign * (dx * np.cos(ymid) + dy * np.sin(ymid))
        lat = np.abs(-dx * np.sin(ymid) + dy * np.cos(ymid))
        moved = np.hypot(dx, dy) > 1e-6
        bad = moved & ((lon <= 0) | (lat > tan_tol * np.maximum(lon, 0.0) + SLIP_SLACK))
        if bad.any():
            angles = np.degrees(np.arctan2(lat, lon))
            w = int(np.argmax(np.where(bad, angles, -np.inf)))      # report the worst window
            ang = float(angles[w])
            m = int(i[w] + j[w]) // 2
            direction = "backward" if run["reverse"] else "forward"
            hint = " (wrong reverse flag?)" if ang > 150 else ""
            return r.fail(f"car slides sideways: moves {ang:.0f} deg off its heading "
                          f"while driving {direction}{hint}", x[m], y[m])
        steer_events += _count_steer_events(s, _curvature_profile(s, yaw))

    # ---------------- whole path: heading changes only while driving ----------------
    # Travel adds up across runs (forward and backward both count); the
    # heading may not change faster than the steering allows per meter of
    # travel, including across segment boundaries and direction changes.
    ST, HT, XT, YT = [], [], [], []
    s_off, h_prev = 0.0, None
    for run in runs:
        x, y, yaw, s = run["x"], run["y"], run["yaw"], run["s"]
        if h_prev is not None:
            yaw = yaw + (h_prev - yaw[0]) - float(wrap_to_pi(h_prev - yaw[0]))
            s_off += math.hypot(x[0] - XT[-1][-1], y[0] - YT[-1][-1])
        ST.append(s + s_off)
        HT.append(yaw)
        XT.append(x)
        YT.append(y)
        s_off = float(ST[-1][-1])
        h_prev = float(yaw[-1])
    ST, HT, XT, YT = (np.concatenate(a) for a in (ST, HT, XT, YT))
    i, j = _windows(ST)
    turn = np.abs(HT[j] - HT[i])
    allowed = CURV_LIMIT * (ST[j] - ST[i]) + HEADING_SLACK
    if np.any(turn > allowed):
        w = int(np.argmax(turn - allowed))
        m = int(i[w] + j[w]) // 2
        kap = turn[w] / max(ST[j[w]] - ST[i[w]], 1e-9)
        return r.fail(f"turns tighter than the car can: heading changes {math.degrees(turn[w]):.1f} deg over "
                      f"{ST[j[w]] - ST[i[w]]:.2f} m of travel (curvature {kap:.3f} 1/m, max "
                      f"{V.MAX_CURVATURE:.3f}; the car cannot turn while standing still)", XT[m], YT[m])
    segs = runs

    # ---------------- collisions ----------------
    X = np.concatenate([sg["x"] for sg in segs])
    Y = np.concatenate([sg["y"] for sg in segs])
    H = np.concatenate([sg["yaw"] for sg in segs])
    hit = problem.collides_many(X, Y, H)
    if hit.any():
        i = int(np.argmax(hit))
        r.fail_pose = (float(X[i]), float(Y[i]), float(H[i]))
        return r.fail("collision", *_contact_point(problem, X[i], Y[i], H[i]))

    # ---------------- final pose ----------------
    fx, fy, fh = float(X[-1]), float(Y[-1]), float(H[-1])
    r.final_pose = (fx, fy, fh)
    for k in range(len(segs) - 1):
        if segs[k]["reverse"] != segs[k + 1]["reverse"]:
            r.cusps.append((float(segs[k]["x"][-1]), float(segs[k]["y"][-1])))
    pe = problem.spot.placement_error(fx, fy, fh)
    if not problem.spot.contains(fx, fy, fh):
        return r.fail("the car does not end fully inside the spot",
                      fx + CENTER_OFFSET * math.cos(fh), fy + CENTER_OFFSET * math.sin(fh))
    if abs(pe["heading_deg"]) > FINAL_HEADING_TOL_DEG:
        return r.fail(f"final heading is {abs(pe['heading_deg']):.1f} deg off the spot "
                      f"(max {FINAL_HEADING_TOL_DEG:.0f})", fx, fy)

    clear = problem.clearance_many(X, Y, H)
    r.passed = True
    r.metrics.update({
        "length_m": float(sum(sg["s"][-1] for sg in segs)),
        "segments": len(segs),
        "cusps": len(r.cusps),
        "min_clearance_m": float(np.min(clear)),
        "lateral_m": abs(pe["lateral_m"]),
        "heading_deg": abs(pe["heading_deg"]),
        "steer_events": int(steer_events),
    })
    return r
