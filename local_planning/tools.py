"""A small toolbox for your planner: the plumbing, so you can focus on the decisions.

* :class:`ObstacleMemory` remembers what perception has seen, in world and Frenet coordinates.
* :func:`lateral_transition` and :func:`transition_length` make smooth sideways shifts.
* :func:`stations` and :func:`path_curvature` set up where you plan and how curvy it is.
* :func:`build_trajectory` turns (s, d, v) into the ReferenceTrajectory message.
* :class:`Debouncer` filters a flickering yes/no signal (like a traffic-light color).

``planner.py`` already uses these.  You do not need to edit this file.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common import vehicle as V
from common.messages import CarState, ObjClass, ReferenceTrajectory

__all__ = ["TrackedObject", "ObstacleMemory", "lateral_transition", "transition_length",
           "stations", "path_curvature", "build_trajectory", "Debouncer", "detection_to_world"]


def detection_to_world(ego: CarState, det):
    """A detection's (x forward, y left) from the front bumper, in world x, y."""
    c, s = math.cos(ego.psi), math.sin(ego.psi)
    bx = ego.x + V.REAR_AXLE_TO_FRONT_BUMPER * c
    by = ego.y + V.REAR_AXLE_TO_FRONT_BUMPER * s
    return bx + c * det.x - s * det.y, by + s * det.x + c * det.y


# ---------------------------------------------------------------------- #
# Obstacle memory
# ---------------------------------------------------------------------- #
@dataclass
class TrackedObject:
    """One remembered object.  World position is smoothed; Frenet is along the route.

    ``length`` / ``width`` are the object's extent along / across the road
    (taken from the detection, which reports them along / across the car).
    """

    object_id: int
    obj_class: int
    x: float
    y: float
    length: float
    width: float
    s: float = 0.0
    d: float = 0.0
    s_min: float = 0.0
    s_max: float = 0.0
    d_min: float = 0.0
    d_max: float = 0.0
    vx: float = 0.0          # world velocity [m/s] (pedestrians only)
    vy: float = 0.0
    vs: float = 0.0          # velocity along / across the route [m/s] (pedestrians only)
    vd: float = 0.0
    seen: bool = True        # detected this tick (False = remembered)
    hits: int = 0
    last_seen: float = 0.0
    _hist: deque = field(default_factory=lambda: deque(maxlen=12), repr=False)

    @property
    def name(self) -> str:
        return ObjClass.NAMES.get(self.obj_class, "unknown")

    @property
    def is_pedestrian(self) -> bool:
        return self.obj_class == ObjClass.PEDESTRIAN


class ObstacleMemory:
    """Remembers what perception has seen, in world and Frenet coordinates.

    * Converts detections from the front-bumper vehicle frame to the world.
    * Averages each object's position and size over time (per ``object_id``).
    * KEEPS static objects after they leave the camera's field of view, until
      the car is ``forget_behind`` meters past them.  (You lose sight of a
      barrel while you are next to it.  This remembers it for you.)
    * Estimates pedestrian velocity from their last ~1 s of positions.

    Traffic lights are not stored here: read their color straight from the
    detections (see :class:`Debouncer`).
    """

    def __init__(self, alpha: float = 0.05, pedestrian_alpha: float = 0.5,
                 min_hits: int = 2, forget_behind: float = 30.0,
                 pedestrian_timeout: float = 1.5):
        self.alpha, self.ped_alpha = alpha, pedestrian_alpha
        self.min_hits, self.forget_behind = min_hits, forget_behind
        self.ped_timeout = pedestrian_timeout
        self.tracks: Dict[int, TrackedObject] = {}
        self._s_hint: Optional[float] = None

    def update(self, obs, route_map) -> List[TrackedObject]:
        """Fold in this tick's detections.  Returns remembered objects, sorted by ``s_min``."""
        ego, t = obs.ego, obs.t
        ref = route_map.reference
        s_car, _, _ = ref.project(ego.x, ego.y, s_hint=self._s_hint)
        self._s_hint = s_car
        for tr in self.tracks.values():
            tr.seen = False
        for det in obs.objects.objects:
            if det.obj_class == ObjClass.TRAFFIC_LIGHT:
                continue
            wx, wy = detection_to_world(ego, det)
            tr = self.tracks.get(det.object_id)
            if tr is None:
                tr = TrackedObject(det.object_id, det.obj_class, wx, wy, det.length, det.width)
                self.tracks[det.object_id] = tr
            else:
                # static objects: running mean of every sighting (weight never below
                # ``alpha``); pedestrians: a short moving average so they can move
                a = self.ped_alpha if tr.is_pedestrian else max(1.0 / (tr.hits + 1), self.alpha)
                tr.x += a * (wx - tr.x)
                tr.y += a * (wy - tr.y)
                tr.length += a * (det.length - tr.length)
                tr.width += a * (det.width - tr.width)
            tr.seen, tr.hits, tr.last_seen = True, tr.hits + 1, t
            if tr.is_pedestrian:
                tr._hist.append((t, wx, wy))
                self._velocity(tr, t)
            tr.s, tr.d, _ = ref.project(tr.x, tr.y, s_hint=s_car + det.x + V.REAR_AXLE_TO_FRONT_BUMPER)
            if tr.is_pedestrian:
                _, _, h = ref.interp(tr.s)
                tr.vs = tr.vx * math.cos(h) + tr.vy * math.sin(h)
                tr.vd = -tr.vx * math.sin(h) + tr.vy * math.cos(h)
            tr.s_min, tr.s_max = tr.s - 0.5 * tr.length, tr.s + 0.5 * tr.length
            tr.d_min, tr.d_max = tr.d - 0.5 * tr.width, tr.d + 0.5 * tr.width
        for k in list(self.tracks):
            tr = self.tracks[k]
            if tr.is_pedestrian and t - tr.last_seen > self.ped_timeout:
                del self.tracks[k]
            elif not tr.is_pedestrian and tr.s_max < s_car - self.forget_behind:
                del self.tracks[k]
        out = [tr for tr in self.tracks.values() if tr.hits >= self.min_hits]
        return sorted(out, key=lambda tr: tr.s_min)

    @staticmethod
    def _velocity(tr: TrackedObject, t: float) -> None:
        h = np.array([p for p in tr._hist if p[0] >= t - 1.0])
        if len(h) < 3 or h[-1, 0] - h[0, 0] < 0.2:
            return
        tt = h[:, 0] - h[:, 0].mean()
        den = float(np.sum(tt * tt))
        tr.vx = float(np.sum(tt * (h[:, 1] - h[:, 1].mean())) / den)
        tr.vy = float(np.sum(tt * (h[:, 2] - h[:, 2].mean())) / den)


# ---------------------------------------------------------------------- #
# Lateral offsets
# ---------------------------------------------------------------------- #
def lateral_transition(s, s_begin: float, s_end: float, d_from: float, d_to: float,
                       shape: str = "quintic") -> np.ndarray:
    """A smooth shift of the lateral offset from ``d_from`` to ``d_to``.

    Constant ``d_from`` before ``s_begin``, constant ``d_to`` after ``s_end``,
    and a smooth S-curve in between.  Add two of them to make a nudge::

        d = (lateral_transition(s, a - L, a, 0.0, 0.5)     # move 0.5 m left before a
             + lateral_transition(s, b, b + L, 0.0, -0.5))  # and back after b

    ``shape='quintic'`` (default) starts and ends with zero curvature, so the
    steering is smooth.  ``'cosine'`` is fine too but has a small curvature
    jump at both ends.  Anchor ``s_begin``/``s_end`` to the obstacle, not to
    the car, or your plan will slide forward every tick.

    How long does it need to be?  The peak extra curvature of a quintic shift
    is about 5.77 * |d_to - d_from| / L**2, so the lateral acceleration at
    speed v is v**2 times that.  :func:`transition_length` does the math.
    """
    s = np.asarray(s, dtype=float)
    L = max(s_end - s_begin, 1e-6)
    u = np.clip((s - s_begin) / L, 0.0, 1.0)
    if shape == "cosine":
        h = 0.5 - 0.5 * np.cos(math.pi * u)
    else:
        h = u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)
    return d_from + (d_to - d_from) * h


def transition_length(delta_d: float, speed: float, max_lat_accel: float = 1.5,
                      max_extra_curvature: float = 0.1) -> float:
    """Shortest quintic transition [m] for a lateral shift of ``delta_d`` at ``speed``
    that keeps the extra lateral acceleration under ``max_lat_accel`` and the
    extra curvature under ``max_extra_curvature`` (the car's limit is 0.214 1/m)."""
    k_peak = 5.77 * abs(delta_d)
    return max(5.0, math.sqrt(k_peak / max_extra_curvature),
               speed * math.sqrt(k_peak / max_lat_accel))


# ---------------------------------------------------------------------- #
# Stations
# ---------------------------------------------------------------------- #
def stations(s0: float, length: float = 50.0, stop_s: Optional[float] = None,
             spacing: float = 0.5) -> np.ndarray:
    """Arc-length stations from ``s0`` to ``s0 + length``, ``spacing`` apart.

    If ``stop_s`` is within reach, the LAST station is exactly ``stop_s`` (so a
    stop lands on the right spot; check ``s[-1] >= stop_s``).  Very close to (or past) ``stop_s`` you get
    three stations: where you are, the stop point, and one just beyond it.
    """
    if stop_s is not None and stop_s - s0 < spacing:
        end = max(stop_s, s0 + 0.01)       # never a zero-length step
        return np.array([s0, end, end + spacing])
    end = s0 + length if stop_s is None else min(s0 + length, stop_s)
    s = np.arange(s0, end - 1e-6, spacing)
    if end - s[-1] < 0.1 * spacing and len(s) > 1:
        s[-1] = end                  # snap: the last station is exactly the end (or stop)
    else:
        s = np.append(s, end)
    return s


# ---------------------------------------------------------------------- #
# Geometry and the output message
# ---------------------------------------------------------------------- #
def _xy_curvature(x, y):
    ds = np.maximum(np.hypot(np.diff(x), np.diff(y)), 1e-6)
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    k = np.gradient(yaw) / np.append(ds, ds[-1])
    return yaw, k


def path_curvature(route_map, s, d) -> np.ndarray:
    """Curvature [1/m] of the path given by offsets ``d`` at stations ``s``.

    Use it for curve speed limits: ``v_max = sqrt(max_lat_accel / |curvature|)``.
    """
    s, d = np.asarray(s, dtype=float), np.asarray(d, dtype=float)
    if len(s) < 3:
        return route_map.reference.curvature_at(s) * np.ones(len(s))
    pad = 3                          # extend both ends straight along the path's slope
    h0, h1 = s[1] - s[0], s[-1] - s[-2]
    k0, k1 = (d[1] - d[0]) / h0, (d[-1] - d[-2]) / h1
    back, fwd = h0 * np.arange(pad, 0, -1), h1 * np.arange(1, pad + 1)
    se = np.concatenate((s[0] - back, s, s[-1] + fwd))
    de = np.concatenate((d[0] - k0 * back, d, d[-1] + k1 * fwd))
    x, y = route_map.reference.frenet_to_xy(se, de)
    _, k = _xy_curvature(x, y)
    return k[pad:-pad]


def build_trajectory(route_map, s, d, v, ego: CarState,
                     blend: bool = True) -> ReferenceTrajectory:
    """Turn stations ``s``, lateral offsets ``d`` and speeds ``v`` into a ReferenceTrajectory.

    * The first point is the car itself, at its current speed (``v[0]`` and
      ``d[0]`` are replaced by what the car is actually doing).
    * With ``blend=True`` the first 5 to 10 m ease from the car's actual
      offset and heading into your ``d``, so a small change in your plan does
      not turn into a kink.  If the car is already on your offsets (the usual
      case) the blend changes nothing; the car drives your ``d``.
    * yaw, curvature, acceleration and relative time are filled in
      consistently with the geometry.
    """
    s = np.asarray(s, dtype=float)
    d = np.asarray(d, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    if not (len(s) == len(d) == len(v)) or len(s) < 2:
        raise ValueError("s, d and v must have the same length (>= 2)")
    ref = route_map.reference
    s0, d0, _ = ref.project(ego.x, ego.y, s_hint=float(s[0]))
    _, _, yaw_ref = ref.interp(s0)
    dpsi = (ego.psi - yaw_ref + math.pi) % (2.0 * math.pi) - math.pi
    d0p = math.tan(dpsi) * (1.0 - float(ref.curvature_at(s0)) * d0)
    if blend and len(s) > 2:
        Lb = float(np.clip(1.0 * ego.v + 5.0, 5.0, 10.0))
        sb = min(s[0] + Lb, s[-1])
        if sb - s[0] > 1.0:
            gp = np.gradient(d, s, edge_order=2)
            gpp = np.gradient(gp, s, edge_order=2)
            db, dbp, dbpp = (float(np.interp(sb, s, q)) for q in (d, gp, gpp))
            a0 = float(gpp[0])        # start bending the way your plan bends here
            T = sb - s[0]
            A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                          [6 * T, 12 * T ** 2, 20 * T ** 3]])
            b = np.array([db - d0 - d0p * T - 0.5 * a0 * T * T, dbp - d0p - a0 * T, dbpp - a0])
            c3, c4, c5 = np.linalg.solve(A, b)
            u = s - s[0]
            inside = u < T
            q = d0 + d0p * u + 0.5 * a0 * u ** 2 + c3 * u ** 3 + c4 * u ** 4 + c5 * u ** 5
            d[inside] = q[inside]
    x, y = ref.frenet_to_xy(s, d)
    x[0], y[0] = ego.x, ego.y
    v[0] = ego.v
    back = np.array([1.5, 1.0, 0.5])
    xb = ego.x - back * math.cos(ego.psi)
    yb = ego.y - back * math.sin(ego.psi)
    yaw, k = _xy_curvature(np.r_[xb, x], np.r_[yb, y])
    yaw, k = yaw[3:], k[3:]
    ds = np.maximum(np.hypot(np.diff(x), np.diff(y)), 1e-6)
    acc = np.append((v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds), 0.0)
    t = np.concatenate(([0.0], np.cumsum(2.0 * ds / np.maximum(v[1:] + v[:-1], 1e-3))))
    return ReferenceTrajectory.from_arrays(x, y, yaw, v, acc, k, t)


# ---------------------------------------------------------------------- #
# Flicker filter
# ---------------------------------------------------------------------- #
class Debouncer:
    """A yes/no signal that ignores flicker.

    Turns on after ``frames_on`` consecutive ``True`` updates and off after
    ``frames_off`` consecutive ``False`` updates.  ``update(None)`` (nothing
    detected this frame) leaves it alone.  Example for a traffic light::

        self.red = Debouncer(frames_on=2, frames_off=3)
        ...
        color = ...  # custom_classification of the light, or None if not seen
        is_red = self.red.update(None if color is None else color == LightColor.RED)
    """

    def __init__(self, frames_on: int = 2, frames_off: int = 3, initial: bool = False):
        self.frames_on, self.frames_off = frames_on, frames_off
        self.state = initial
        self._count = 0

    def update(self, value: Optional[bool]) -> bool:
        if value is None:
            return self.state
        if bool(value) == self.state:
            self._count = 0
        else:
            self._count += 1
            if self._count >= (self.frames_on if value else self.frames_off):
                self.state, self._count = bool(value), 0
        return self.state
