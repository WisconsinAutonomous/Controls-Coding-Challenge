"""Scoring.  Hard rules make a scenario FAIL (score 0).  Otherwise the score
starts at 100 and loses points for discomfort, risk, and sloppiness.

Every penalty is capped so that one bad habit cannot zero a run on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common import vehicle as V

from . import geom
from .runner import DT, STOP_SPEED, RunResult

# (limit, weight, cap) for each penalty; weights are points per unit of excess
LAT_ACCEL_LIMIT = 2.0          # [m/s^2]
ACCEL_LIMIT = 2.0              # [m/s^2]
DECEL_LIMIT = 3.0              # [m/s^2]
JERK_RMS_LIMIT = 3.5           # [m/s^3] a plain trapezoidal speed profile lands near 3
CURV_RATE_LIMIT = 0.2          # [1/(m s)]
CLEARANCE_TARGET = 0.5         # [m]
SPEED_TOL = 0.3                # [m/s]
TIMID_GAP = 3.0                # [m] stopping further than this before a line
CLOSURE_TIMID_GAP = 6.0        # [m]
INSTABILITY_LIMIT = 0.05       # [m] mean lateral change between consecutive plans
CENTER_TOL = 0.5               # [m]
CENTER_BEHIND, CENTER_AHEAD = 30.0, 40.0   # [m] obstacles this close excuse being off center
CURVATURE_FAIL = 1.05 * V.MAX_CURVATURE

PENALTIES = {
    # key: (label, cap)
    "lat_accel": ("lateral accel > 2.0 m/s^2", 15.0),
    "long_accel": ("accel > 2.0 or decel > 3.0 m/s^2", 15.0),
    "jerk": ("jerky speed changes", 10.0),
    "curv_rate": ("steering faster than the car can", 10.0),
    "clearance": ("closer than 0.5 m to an obstacle", 20.0),
    "speeding": ("over the speed limit", 15.0),
    "time": ("slower than par", 15.0),
    "instability": ("plan flip-flops between ticks", 15.0),
    "discontinuity": ("plan does not start at the car", 10.0),
    "invalid": ("unusable plans (emergency stops)", 10.0),
    "centering": ("off lane center for no reason", 10.0),
    "timid_stop": ("stopped too far from the line", 10.0),
    "needless_stop": ("stopped for a pedestrian who was not crossing", 10.0),
}


@dataclass
class Grade:
    name: str
    tier: str
    seed: int
    status: str
    reason: str
    score: float
    time: float
    penalties: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    fail_xy: Optional[tuple] = None
    events: List[str] = field(default_factory=list)

    def top_penalties(self, n: int = 3) -> str:
        items = sorted(((v, k) for k, v in self.penalties.items() if v >= 0.05), reverse=True)[:n]
        return ", ".join(f"{PENALTIES[k][0]} -{v:.1f}" for v, k in items)


def _capped(raw: float, key: str) -> float:
    return float(min(max(raw, 0.0), PENALTIES[key][1]))


def _stop_runs(v: np.ndarray, min_len: int = 5):
    """Index ranges [a, b) where v < STOP_SPEED for at least min_len ticks."""
    runs, a = [], None
    for i, vi in enumerate(v):
        if vi < STOP_SPEED and a is None:
            a = i
        elif vi >= STOP_SPEED and a is not None:
            if i - a >= min_len:
                runs.append((a, i))
            a = None
    if a is not None and len(v) - a >= min_len:
        runs.append((a, len(v)))
    return runs


def grade(res: RunResult) -> Grade:
    sc = res.scenario
    route = sc.route
    ref = route.reference
    tr = res.trace
    ticks = res.ticks
    g = Grade(sc.name, sc.tier, sc.seed, "PASS", "", 100.0, res.t_end, events=list(res.events))
    fail = dict(res.fail) if res.fail else None
    m, p = g.metrics, {}

    t, x, y, v = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 4]

    # ---- curvature / lateral accel / steering rate from the driven path
    s_r, kap, s_tr = geom.path_curvature(x, y)
    if len(kap):
        mask = np.concatenate(([True], np.diff(s_tr) > 1e-6))
        v_r = np.interp(s_r, s_tr[mask], v[mask])
        t_r = np.interp(s_r, s_tr[mask], t[mask])
        dt_r = 0.25 / np.maximum(v_r, 0.3)
        m["max_curvature"] = float(np.abs(kap).max())
        bad = np.nonzero(np.abs(kap) > CURVATURE_FAIL)[0]
        if len(bad):
            i = bad[0]
            if fail is None or t_r[i] < fail["t"]:
                fail = {"t": float(t_r[i]), "x": float(np.interp(s_r[i], s_tr[mask], x[mask])),
                        "y": float(np.interp(s_r[i], s_tr[mask], y[mask])),
                        "reason": f"impossible steering: path curvature {abs(kap[i]):.3f} 1/m "
                                  f"is beyond the car's limit {V.MAX_CURVATURE:.3f} 1/m "
                                  "(a kink or a jump in your path?)"}
        alat = v_r ** 2 * np.abs(kap)
        m["max_lat_accel"] = float(alat.max())
        p["lat_accel"] = 8.0 * float(np.sum(np.maximum(alat - LAT_ACCEL_LIMIT, 0.0) * dt_r))
        dk = np.zeros_like(kap)
        if len(kap) > 9:
            dk[4:-4] = (kap[8:] - kap[:-8]) / 2.0
        krate = np.abs(dk) * v_r
        m["max_curvature_rate"] = float(krate.max())
        p["curv_rate"] = 20.0 * float(np.sum(np.maximum(krate - CURV_RATE_LIMIT, 0.0) * dt_r))

    # ---- longitudinal, from the 10 Hz states
    vt = np.array([tk.v for tk in ticks] + [tr[-1, 4]]) if ticks else np.zeros(1)
    if len(vt) > 2:
        a = np.diff(vt) / DT
        m["max_accel"], m["max_decel"] = float(a.max()), float(-a.min())
        ex = np.maximum(a - ACCEL_LIMIT - 0.05, 0.0) + np.maximum(-a - DECEL_LIMIT - 0.05, 0.0)
        p["long_accel"] = 5.0 * float(np.sum(ex) * DT)
        j = np.diff(a) / DT
        moving = (vt[1:-1] > STOP_SPEED) | (vt[2:] > STOP_SPEED) | (vt[:-2] > STOP_SPEED)
        jr = float(np.sqrt(np.mean(j[moving] ** 2))) if np.any(moving) else 0.0
        m["jerk_rms"] = jr
        p["jerk"] = 3.0 * max(0.0, jr - JERK_RMS_LIMIT)

    # ---- clearance to static obstacles (true geometry)
    min_clear = math.inf
    clear_pen = 0.0
    if sc.world.obstacles and len(tr):
        corners = geom.car_corners(x, y, tr[:, 3])
        cx, cy = x + 1.45 * np.cos(tr[:, 3]), y + 1.45 * np.sin(tr[:, 3])
        for ob in sc.world.obstacles:
            near = np.nonzero(np.hypot(cx - ob.x, cy - ob.y) < 6.0)[0]
            if len(near) == 0:
                continue
            oc = ob.corners()
            c = min(geom.quad_distance(corners[i], oc) for i in near)
            min_clear = min(min_clear, c)
            if c < CLEARANCE_TARGET:
                clear_pen += 10.0 * (CLEARANCE_TARGET - c) / CLEARANCE_TARGET
    m["min_clearance"] = float(min_clear) if math.isfinite(min_clear) else None   # no obstacles
    p["clearance"] = clear_pen

    # ---- speeding, centering, discontinuities, invalid plans, planner time
    if ticks:
        s_t = np.array([tk.s for tk in ticks])
        d_t = np.array([tk.d for tk in ticks])
        v_t = np.array([tk.v for tk in ticks])
        lim = np.array([route.speed_limit_at(si) for si in s_t])
        p["speeding"] = 3.0 * float(np.sum(np.maximum(v_t - lim - SPEED_TOL, 0.0)) * DT)
        m["max_over_limit"] = float(np.max(v_t - lim))
        half = 0.5 * route.lane_width
        obs_s = [(o.s - 0.5 * o.length, o.s + 0.5 * o.length) for o in sc.world.obstacles
                 if abs(o.d) - 0.5 * o.width < half]          # only ones that intrude on the lane
        free = np.ones(len(ticks), dtype=bool)
        for a0, a1 in obs_s:
            free &= ~((a1 >= s_t - CENTER_BEHIND) & (a0 <= s_t + CENTER_AHEAD))
        w = v_t * DT * (v_t > 0.5) * free
        frac = float(np.sum(w * (np.abs(d_t) > CENTER_TOL)) / max(np.sum(w), 1e-6))
        m["off_center_fraction"] = frac
        p["centering"] = 20.0 * frac if np.sum(w) > 1.0 else 0.0
        n_disc = sum(tk.disc for tk in ticks)
        n_inv = sum(tk.invalid for tk in ticks)
        m["discontinuities"], m["invalid_plans"] = n_disc, n_inv
        p["discontinuity"] = 0.25 * n_disc
        p["invalid"] = 1.0 * n_inv
        pm = np.array([tk.plan_ms for tk in ticks])
        m["plan_ms_mean"], m["plan_ms_max"] = float(pm.mean()), float(pm.max())
        m["distance"] = float(s_t[-1] - s_t[0])
        p["instability"] = 0.0
        m["instability"] = _instability(res)
        p["instability"] = 150.0 * max(0.0, m["instability"] - INSTABILITY_LIMIT)

        # ---- stop quality
        fb_s = np.array([_fb_s(ref, tk) for tk in ticks])
        runs = _stop_runs(v_t)
        timid, needless = 0.0, 0.0
        stop_lines = route.stop_lines
        for a0, a1 in runs:
            fs = fb_s[a1 - 1]
            near_line = [sl for sl in stop_lines if -2.0 <= sl.s - fs <= 15.0]
            for sl in near_line:
                gap = sl.s - fs
                if gap > TIMID_GAP:
                    timid += 2.0 * (gap - TIMID_GAP)
            if not near_line:
                for p_ in sc.world.pedestrians:
                    if p_.id in sc.curb_pedestrian_ids and 0.0 <= p_.s - fs <= 25.0:
                        needless = 10.0
        if sc.closure_face_s is not None and res.finished:
            gap = sc.closure_face_s - fb_s[-1]
            m["closure_gap"] = float(gap)
            if gap > CLOSURE_TIMID_GAP:
                timid += 2.0 * (gap - CLOSURE_TIMID_GAP)
        p["timid_stop"], p["needless_stop"] = timid, needless

    if sc.par_time and res.finished and fail is None:
        p["time"] = 50.0 * max(0.0, res.t_end / sc.par_time - 1.0)
        m["par_time"] = sc.par_time

    g.penalties = {k: _capped(vv, k) for k, vv in p.items()}
    if fail is not None:
        g.status, g.reason, g.score = "FAIL", fail["reason"], 0.0
        g.fail_xy = (fail["x"], fail["y"])
        g.time = fail["t"]
    else:
        g.score = max(0.0, 100.0 - sum(g.penalties.values()))
    return g


def _fb_s(ref, tk) -> float:
    fx, fy = geom.front_bumper_xy(tk.x, tk.y, tk.psi)
    s, _ = geom.project_points(ref, fx, fy, tk.s + 3.7, 15.0)
    return float(s[0])


def _instability(res: RunResult) -> float:
    """Mean |change in lateral offset| between consecutive plans, 5..30 m ahead."""
    ref = res.scenario.route.reference
    coarse = ref.resampled(1.0)
    vals = []
    prev = None
    for tk in res.ticks:
        if tk.plan is None:
            prev = None
            continue
        pts = tk.plan[::2]
        sp, dp = geom.project_points(coarse, pts[:, 0], pts[:, 1], tk.s + 20.0, 45.0)
        cur = (sp, dp)
        if prev is not None and tk.v > 0.5:
            sig = tk.s + np.arange(5.0, 30.1, 1.0)
            ok = ((sig >= max(prev[0].min(), cur[0].min()))
                  & (sig <= min(prev[0].max(), cur[0].max())))
            if ok.sum() >= 5 and np.all(np.diff(prev[0]) > 0) and np.all(np.diff(cur[0]) > 0):
                d_prev = np.interp(sig[ok], prev[0], prev[1])
                d_cur = np.interp(sig[ok], cur[0], cur[1])
                vals.append(float(np.mean(np.abs(d_cur - d_prev))))
        prev = cur
    return float(np.mean(vals)) if vals else 0.0
