"""The ideal follower: a perfect controller that drives exactly along your plan.

Every 0.1 s the car moves along the trajectory you just returned, starting
from its own projection onto it, with the speed you asked for.  Times are
recomputed from your geometry and speeds (constant acceleration between
consecutive points, i.e. dt = 2*ds / (v_i + v_{i+1})), so
``relative_time_sec`` is ignored.

If the plan is unusable the car performs an emergency stop at -5 m/s^2 along
the last usable path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from common.geometry import Polyline
from common.messages import CarState, ReferenceTrajectory

MAX_SPACING = 1.0        # [m]
DISC_POS = 0.3           # [m]
DISC_YAW = math.radians(10.0)
DISC_SPEED = 1.0         # [m/s]
ESTOP_DECEL = 5.0        # [m/s^2]
STOPPED = 1e-3           # [m/s]


@dataclass
class StepResult:
    samples: List[Tuple[float, float, float, float, float]]  # (t, x, y, yaw, v)
    state: CarState
    invalid: bool = False
    reason: str = ""
    discontinuity: bool = False
    disc_detail: str = ""
    plan_xyv: Optional[np.ndarray] = None   # (N, 3) cleaned plan, for logging


def _clean_plan(plan) -> Tuple[Optional[np.ndarray], str]:
    if plan is None:
        return None, "planner returned None"
    if not isinstance(plan, ReferenceTrajectory):
        return None, f"planner returned {type(plan).__name__}, not ReferenceTrajectory"
    if len(plan.points) < 2:
        return None, "trajectory has fewer than 2 points"
    arr = plan.as_arrays()
    x, y, v = arr["x"], arr["y"], arr["velocity_mps"]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(v))):
        return None, "trajectory contains NaN or inf"
    if np.any(v < -1e-6):
        return None, "trajectory has negative speeds"
    seg = np.hypot(np.diff(x), np.diff(y))
    if np.any(seg > MAX_SPACING + 1e-6):
        return None, f"trajectory points {seg.max():.2f} m apart (max {MAX_SPACING} m)"
    keep = np.concatenate(([True], seg > 1e-6))
    if keep.sum() < 2:
        return None, "trajectory has no length"
    # duplicate points: keep the first, but honor the lowest requested speed
    vv = v.copy()
    idx = np.cumsum(keep) - 1
    vmin = np.full(keep.sum(), np.inf)
    np.minimum.at(vmin, idx, vv)
    return np.column_stack([x[keep], y[keep], np.maximum(vmin, 0.0)]), ""


def _path(xyv: np.ndarray) -> Polyline:
    """The plan as a polyline whose end headings are the curve's tangent, not the
    end chord's (a chord is off by curvature * spacing / 2, ~1 deg in tight turns)."""
    x, y = xyv[:, 0], xyv[:, 1]
    h = np.unwrap(np.arctan2(np.diff(y), np.diff(x)))
    yaw = np.empty(len(x))
    if len(h) == 1:
        yaw[:] = h[0]
    else:
        yaw[1:-1] = 0.5 * (h[:-1] + h[1:])
        yaw[0] = 1.5 * h[0] - 0.5 * h[1]
        yaw[-1] = 1.5 * h[-1] - 0.5 * h[-2]
    return Polyline(x, y, yaw=yaw)


class IdealFollower:
    def __init__(self):
        self.last_path: Optional[Polyline] = None

    def step(self, ego: CarState, plan, t0: float, dt: float = 0.1) -> StepResult:
        xyv, reason = _clean_plan(plan)
        if xyv is None:
            return self._estop(ego, t0, dt, reason, self.last_path)

        path = _path(xyv)
        v = xyv[:, 2]
        s_car, _, i = path.project(ego.x, ego.y)
        # project() extrapolates past the ends; the car can only start on the plan
        s_car = min(max(s_car, 0.0), path.length)
        disc, detail = False, ""
        dpos = math.hypot(xyv[0, 0] - ego.x, xyv[0, 1] - ego.y)
        dyaw = abs((path.yaw[0] - ego.psi + math.pi) % (2 * math.pi) - math.pi)
        dv = abs(v[0] - ego.v)
        if dpos > DISC_POS or dyaw > DISC_YAW or dv > DISC_SPEED:
            disc = True
            detail = f"start off by {dpos:.2f} m, {math.degrees(dyaw):.1f} deg, {dv:.2f} m/s"

        # speed at the projection (linear in s between points)
        u = 0.0 if path.seg_len[i] <= 0 else (s_car - path.s[i]) / path.seg_len[i]
        v_cur = float(v[i] + (v[i + 1] - v[i]) * min(max(u, 0.0), 1.0))
        pos, remaining, t = s_car, dt, t0
        samples = []
        nseg = len(path.seg_len)
        while remaining > 1e-9:
            if i >= nseg:
                if v_cur > 0.05:
                    self.last_path = path
                    ex, ey, eyaw = self._pose(path, pos)
                    return self._estop(CarState(x=ex, y=ey, v=v_cur, psi=eyaw, stamp=t),
                                       t, remaining, "trajectory ended while still moving "
                                       "(plan too short)", path, samples, xyv, disc, detail,
                                       t_end=t0 + dt)
                v_cur = 0.0
                break
            s_b, v_b = path.s[i + 1], v[i + 1]
            L = s_b - pos
            if L <= 1e-9:
                i += 1
                continue
            if v_cur + v_b < STOPPED:
                v_cur = 0.0
                break
            T = 2.0 * L / (v_cur + v_b)
            if T <= remaining:
                remaining -= T
                t += T
                pos, v_cur = s_b, float(v_b)
                x, y, yaw = self._pose(path, pos)
                samples.append((t, x, y, yaw, v_cur))
                i += 1
            else:
                a = (v_b * v_b - v_cur * v_cur) / (2.0 * L)
                ds = min(max(v_cur * remaining + 0.5 * a * remaining ** 2, 0.0), L)
                v_cur = max(0.0, v_cur + a * remaining)
                pos += ds
                remaining = 0.0
        x, y, yaw = self._pose(path, pos)
        samples.append((t0 + dt, x, y, yaw, v_cur))
        self.last_path = path
        return StepResult(samples, CarState(x=x, y=y, v=v_cur, psi=yaw, stamp=t0 + dt),
                          discontinuity=disc, disc_detail=detail, plan_xyv=xyv)

    @staticmethod
    def _pose(path: Polyline, s: float):
        x, y, h = path.interp(s)
        return float(x), float(y), float((h + math.pi) % (2 * math.pi) - math.pi)

    def _estop(self, ego: CarState, t0: float, dt: float, reason: str,
               path: Optional[Polyline], samples=None, xyv=None, disc=False, detail="",
               t_end: Optional[float] = None) -> StepResult:
        """Brake at ESTOP_DECEL along ``path`` (or straight ahead)."""
        samples = list(samples or [])
        v0 = ego.v
        tau = min(dt, v0 / ESTOP_DECEL) if v0 > 0 else 0.0
        dist = v0 * tau - 0.5 * ESTOP_DECEL * tau * tau
        v1 = max(0.0, v0 - ESTOP_DECEL * dt)
        n = max(1, int(math.ceil(dist / 0.5)))
        if path is not None:
            s0, _, _ = path.project(ego.x, ego.y)
            s0 = max(s0, 0.0)   # past the end we keep going straight, see below
        pts = []
        for k in range(1, n + 1):
            ds = dist * k / n
            if path is not None and s0 + ds <= path.length:
                x, y, yaw = self._pose(path, s0 + ds)
            else:
                if path is not None:
                    ex, ey, eyaw = self._pose(path, path.length)
                    extra = s0 + ds - path.length
                    base = (ex, ey, eyaw)
                else:
                    base, extra = (ego.x, ego.y, ego.psi), ds
                x = base[0] + extra * math.cos(base[2])
                y = base[1] + extra * math.sin(base[2])
                yaw = base[2]
            pts.append((x, y, yaw))
        t_final = t_end if t_end is not None else t0 + dt
        for k, (x, y, yaw) in enumerate(pts):
            tk = t0 + (k + 1) / n * (t_final - t0)
            samples.append((tk, x, y, yaw, v1 if k == n - 1 else v0))
        x, y, yaw = pts[-1] if dist > 0 else (ego.x, ego.y, ego.psi)
        return StepResult(samples, CarState(x=x, y=y, v=v1, psi=yaw, stamp=t_final),
                          invalid=True, reason=reason, discontinuity=disc,
                          disc_detail=detail, plan_xyv=xyv)
