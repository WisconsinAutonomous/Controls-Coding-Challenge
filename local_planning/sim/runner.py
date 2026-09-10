"""Closed loop: perception -> your planner -> ideal follower, at 10 Hz.

Hard safety rules are checked online so the run stops at the first one that
is broken (there is no point driving on after hitting a barrel).
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common import vehicle as V
from common.messages import CarState, LightColor

from . import geom
from .follower import IdealFollower
from .perception import PerceptionModel
from .scenarios import Scenario
from .world import Observation

DT = 0.1
LINE_TOL = 0.05              # [m] how far past a line counts as crossing it
STOP_SPEED = 0.05            # [m/s] "stopped"
STOP_SIGN_HOLD = 3.0         # [s]
STOP_SIGN_ZONE = 4.0         # [m] front bumper must be within this of the line
PED_LANE_MARGIN = 1.0        # [m] pedestrian conflict zone beyond the ego lane
MAX_INVALID_FRACTION = 0.05
GOAL_TOL = 3.0               # [m]
CLOSURE_GAP = (0.0, 10.0)    # [m] where to wait behind a closed road
CLOSURE_HOLD = 5.0           # [s]


@dataclass
class TickRecord:
    t: float
    x: float
    y: float
    psi: float
    v: float
    s: float
    d: float
    invalid: bool
    reason: str
    disc: bool
    disc_detail: str
    plan: Optional[np.ndarray]
    plan_ms: float
    dets: Optional[np.ndarray] = None   # (K, 3) world x, y, class of this tick's detections


@dataclass
class RunResult:
    scenario: Scenario
    ticks: List[TickRecord]
    trace: np.ndarray                  # (M, 5): t, x, y, yaw, v
    fail: Optional[Dict] = None        # {'t','reason','x','y'}
    finished: bool = False
    t_end: float = 0.0
    first_exception: str = ""        # short, readable form (where in your code, and the error)
    events: List[str] = field(default_factory=list)
    planner_missing: List[str] = field(default_factory=list)   # parts the planner reports as not done


def _dets_world(ego: CarState, objects) -> np.ndarray:
    if not objects:
        return np.zeros((0, 3), dtype=np.float32)
    c, s = math.cos(ego.psi), math.sin(ego.psi)
    fx, fy = ego.x + V.REAR_AXLE_TO_FRONT_BUMPER * c, ego.y + V.REAR_AXLE_TO_FRONT_BUMPER * s
    return np.array([(fx + c * o.x - s * o.y, fy + s * o.x + c * o.y, o.obj_class)
                     for o in objects], dtype=np.float32)


def short_error(t: Optional[float] = None) -> str:
    """The current exception as 'where in the candidate's code' plus the message."""
    etype, err, tb = sys.exc_info()
    frames = [f for f in traceback.extract_tb(tb) if "/sim/" not in f.filename.replace("\\", "/")
              and "/common/" not in f.filename.replace("\\", "/")]
    where = ""
    if frames and etype.__name__ != "PartError":      # a PartError message says it all
        f = frames[-1]
        where = f"  {os.path.basename(f.filename)}, line {f.lineno}, in {f.name}\n    {f.line}\n"
    when = f"t={t:.1f}s: " if t is not None else ""
    return f"{when}{etype.__name__}: {err}\n{where}".rstrip()


def _state_at(route, s: float, v: float) -> CarState:
    x, y, yaw = route.reference.interp(s)
    return CarState(x=float(x), y=float(y), v=v,
                    psi=float((yaw + math.pi) % (2 * math.pi) - math.pi))


class _Monitor:
    """Online hard-rule checks."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.route = sc.route
        self.half = 0.5 * self.route.lane_width
        self.stop_hold: Dict[int, float] = {}
        self.stop_ok: Dict[int, bool] = {}
        self.closure_hold = 0.0
        self.s_hint = sc.start_s
        self.events: List[str] = []
        self.cw_peds = {}
        for sl in self.route.stop_lines:
            if sl.kind == "crosswalk":
                self.cw_peds[sl.id] = [p for p in sc.world.pedestrians if p.crosses and
                                       sl.crosswalk_s_start - 1 <= p.s <= sl.crosswalk_s_end + 1]

    def fb_s(self, x, y, yaw) -> float:
        fx, fy = geom.front_bumper_xy(x, y, yaw)
        s, _ = geom.project_points(self.route.reference, fx, fy, self.s_hint + 3.7, 20.0)
        return float(s[0])

    def check_samples(self, samples, prev) -> Optional[Dict]:
        """Check the swept samples of one tick.  ``prev`` is the pose before."""
        ref = self.route.reference
        pts = [prev] + list(samples)
        arr = np.array(pts)
        t, x, y, yaw = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        corners = geom.car_corners(x, y, yaw)               # (M, 4, 2)
        M = len(pts)
        cs, cd = geom.project_points(ref, corners[..., 0].ravel(), corners[..., 1].ravel(),
                                     self.s_hint, 25.0)
        cs, cd = cs.reshape(M, 4), cd.reshape(M, 4)
        fx, fy = geom.front_bumper_xy(x, y, yaw)
        fs, _ = geom.project_points(ref, fx, fy, self.s_hint + 3.7, 25.0)

        for i in range(1, M):
            # --- lines
            left_lim = self.route.left_limit_array(cs[i])
            if np.any(cd[i] < -self.half - LINE_TOL):
                return self._fail(t[i], x[i], y[i], "crossed the solid white edge line (left the road)")
            over = cd[i] > left_lim + LINE_TOL
            if np.any(over):
                j = int(np.argmax(cd[i] - left_lim))
                if cd[i][j] > 3 * self.half + LINE_TOL:
                    msg = "left the road on the left"
                else:
                    kind = self.route.line_at("left", cs[i][j]).replace("_", " ")
                    msg = f"crossed a {kind} line where crossing is not allowed"
                return self._fail(t[i], x[i], y[i], msg)
            # --- static obstacles
            cx, cy = x[i] + 1.45 * math.cos(yaw[i]), y[i] + 1.45 * math.sin(yaw[i])
            for ob in self.sc.world.obstacles:
                if math.hypot(ob.x - cx, ob.y - cy) > 6.0:
                    continue
                if geom.quads_overlap(corners[i], ob.corners()):
                    from common.messages import ObjClass
                    return self._fail(t[i], x[i], y[i],
                                      f"hit a {ObjClass.NAMES.get(ob.obj_class, 'obstacle')}")
            # --- pedestrians
            for p in self.sc.world.pedestrians:
                px, py = self.sc.world.pedestrian_xy(p, t[i])
                if math.hypot(px - cx, py - cy) < 4.0:
                    if geom.point_quad_distance(np.array([px, py]), corners[i]) < p.radius:
                        return self._fail(t[i], x[i], y[i], "hit a pedestrian")
            # --- stop lines crossed between samples i-1 and i
            for sl in self.route.stop_lines:
                if fs[i - 1] < sl.s <= fs[i]:
                    tc = t[i - 1] + (t[i] - t[i - 1]) * (sl.s - fs[i - 1]) / max(fs[i] - fs[i - 1], 1e-9)
                    if sl.kind == "traffic_light":
                        L = self.sc.world.light_for_stop_line(sl.id)
                        if L is not None and L.color(tc) == LightColor.RED:
                            return self._fail(tc, x[i], y[i], "ran a red light")
                        self.events.append(f"t={tc:.1f}s crossed light {sl.id} on "
                                           f"{'yellow' if L and L.color(tc) == LightColor.YELLOW else 'green'}")
                    elif sl.kind == "stop_sign" and not self.stop_ok.get(sl.id, False):
                        return self._fail(tc, x[i], y[i],
                                          "rolled through a stop sign (need a full stop for "
                                          f"{STOP_SIGN_HOLD:.0f} s with the bumper within "
                                          f"{STOP_SIGN_ZONE:.0f} m of the line)")
            # --- crosswalks
            for sl in self.route.stop_lines:
                if sl.kind != "crosswalk":
                    continue
                if cs[i].max() >= sl.crosswalk_s_start and cs[i].min() <= sl.crosswalk_s_end:
                    for p in self.cw_peds[sl.id]:
                        if abs(p.d_at(t[i])) <= self.half + PED_LANE_MARGIN:
                            return self._fail(t[i], x[i], y[i],
                                              "entered the crosswalk while a pedestrian was crossing "
                                              "your lane (failed to yield)")
        self.s_hint = float(np.mean(cs[-1]))
        return None

    def after_tick(self, state: CarState, t: float) -> bool:
        """Update stop-sign / closure bookkeeping.  Returns True on success-finish."""
        fs = self.fb_s(state.x, state.y, state.psi)
        for sl in self.route.stop_lines:
            if sl.kind != "stop_sign" or self.stop_ok.get(sl.id):
                continue
            gap = sl.s - fs
            if state.v < STOP_SPEED and -0.05 <= gap <= STOP_SIGN_ZONE:
                self.stop_hold[sl.id] = self.stop_hold.get(sl.id, 0.0) + DT
                if self.stop_hold[sl.id] >= STOP_SIGN_HOLD - 1e-6:
                    self.stop_ok[sl.id] = True
                    self.events.append(f"t={t:.1f}s stop sign {sl.id} satisfied "
                                       f"(bumper {gap:.1f} m from line)")
            else:
                self.stop_hold[sl.id] = 0.0
        if self.sc.closure_face_s is not None:
            gap = self.sc.closure_face_s - fs
            if state.v < STOP_SPEED and CLOSURE_GAP[0] <= gap <= CLOSURE_GAP[1]:
                self.closure_hold += DT
                if self.closure_hold >= CLOSURE_HOLD - 1e-6:
                    self.events.append(f"t={t:.1f}s waiting behind the closure "
                                       f"({gap:.1f} m gap): success")
                    return True
            else:
                self.closure_hold = 0.0
        return False

    def _fail(self, t, x, y, reason):
        return {"t": float(t), "x": float(x), "y": float(y), "reason": reason}


def run_scenario(sc: Scenario, planner_cls, verbose_errors: bool = True,
                 raise_errors: bool = False) -> RunResult:
    route = sc.route
    world = sc.world
    perception = PerceptionModel(world, sc.seed)
    follower = IdealFollower()
    mon = _Monitor(sc)
    ego = _state_at(route, sc.start_s, sc.start_v)
    ticks: List[TickRecord] = []
    trace = [(0.0, ego.x, ego.y, ego.psi, ego.v)]
    result = RunResult(sc, ticks, np.zeros((0, 5)))

    try:
        planner = planner_cls(copy.deepcopy(route))
    except Exception:
        if raise_errors:
            raise
        result.first_exception = short_error()
        result.fail = {"t": 0.0, "x": ego.x, "y": ego.y,
                       "reason": "Planner.__init__ raised an exception"}
        result.trace = np.array(trace)
        return result

    n_invalid = 0
    t = 0.0
    n_steps = int(round(sc.time_limit / DT))
    for k in range(n_steps):
        t = k * DT
        s_rear, d_rear = geom.project_points(route.reference, ego.x, ego.y, mon.s_hint, 25.0)
        fb = mon.fb_s(ego.x, ego.y, ego.psi)
        world.update_triggers(t, fb, ego.v)
        obs = Observation(t=t, ego=copy.copy(ego), objects=perception.observe(t, ego))
        obs.ego.stamp = t
        dets = _dets_world(ego, obs.objects.objects)
        plan, err = None, ""
        t0 = time.perf_counter()
        try:
            plan = planner.plan(obs)
        except Exception as e:
            if raise_errors:
                raise
            msg = f"{type(e).__name__}: {e}"
            err = "Planner.plan raised " + (msg if len(msg) <= 90 else msg[:87] + "...")
            if not result.first_exception:
                result.first_exception = short_error(t)
        plan_ms = 1000.0 * (time.perf_counter() - t0)
        step = follower.step(ego, plan, t, DT)
        invalid = step.invalid or bool(err)
        n_invalid += int(invalid)
        ticks.append(TickRecord(t, ego.x, ego.y, ego.psi, ego.v, float(s_rear[0]),
                                float(d_rear[0]), invalid, err or step.reason, step.discontinuity,
                                step.disc_detail,
                                None if step.plan_xyv is None else step.plan_xyv.astype(np.float32),
                                plan_ms, dets))
        fail = mon.check_samples(step.samples, trace[-1])
        trace.extend(step.samples)
        ego = step.state
        if fail is not None:
            if step.discontinuity:
                fail["reason"] += f" (this tick your plan did not start at the car: {step.disc_detail})"
            elif invalid:
                fail["reason"] += f" (during an emergency stop: {err or step.reason})"
            result.fail = fail
            break
        if n_invalid >= 20 and n_invalid > MAX_INVALID_FRACTION * (k + 1):
            result.fail = {"t": t, "x": ego.x, "y": ego.y,
                           "reason": f"too many unusable plans ({n_invalid} of {k + 1} ticks); "
                                     f"last problem: {err or step.reason}"}
            break
        if mon.after_tick(ego, t + DT):
            result.finished = True
            break
        s_now, _ = geom.project_points(route.reference, ego.x, ego.y, mon.s_hint, 25.0)
        if sc.closure_face_s is None and (
                (s_now[0] >= route.goal_s - GOAL_TOL and ego.v < STOP_SPEED)
                or s_now[0] >= route.goal_s + GOAL_TOL):
            result.finished = True
            break
    else:
        result.fail = {"t": sc.time_limit, "x": ego.x, "y": ego.y,
                       "reason": f"timeout: did not reach the goal within {sc.time_limit:.0f} s"}
    result.t_end = t + DT
    result.trace = np.array(trace)
    result.events = mon.events
    result.planner_missing = sorted(getattr(planner, "missing", []) or [])
    if (result.fail is None and len(ticks) > 0
            and n_invalid > MAX_INVALID_FRACTION * len(ticks)):
        result.fail = {"t": result.t_end, "x": ego.x, "y": ego.y,
                       "reason": f"too many unusable plans ({n_invalid} of {len(ticks)} ticks)"}
    return result
