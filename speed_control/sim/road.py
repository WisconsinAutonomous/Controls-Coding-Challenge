"""The road and the planner: what speed the car should be doing, and where to stop.

The planner turns speed limits and stops into a smooth speed plan along the
road (accelerate at most 1.5 m/s^2, slow down at most 2.0 m/s^2, speed 0 at
every stop).  Every tick it tells your controller:

* the speed the plan wants at the car's current position,
* how fast that target is changing as the car drives (its time derivative),
* how far away the next stop point is.

Like our real planner, the plan never asks for less than 1 m/s where the car
has to pull away (start of the road, after a stop), so it does not ask a
standing car to stay standing.  While a stop is active, the plan is zero at
and beyond the stop point.

A stop is released (the plan continues past it) once the car has been
stopped within 1.5 m of the stop point for the hold time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from common import vehicle as V

SPACING = 0.5          # [m] plan resolution
ACCEL = 1.5            # [m/s^2] planned acceleration
DECEL = 2.0            # [m/s^2] planned deceleration
STOPPED = 0.05         # [m/s] "stopped"
LAUNCH = 1.0           # [m/s] minimum planned speed where the car pulls away
STOP_WINDOW = 1.5      # [m] stop within this distance of the point
STOP_VISIBLE = 80.0    # [m] stop_distance is reported within this range
LEAD_VISIBLE = 80.0    # [m] lead car is reported within this range


@dataclass
class SpeedTarget:
    """What the planner asks for this tick (all measured from the rear axle)."""

    v: float                                 # [m/s] target speed
    a: float                                 # [m/s^2] how fast the target speed is changing
    stop_distance: Optional[float] = None    # [m] to the next stop point, None if none within 80 m
    lead_distance: Optional[float] = None    # [m] bumper-to-bumper gap to the car ahead (Speed Control stretch)
    lead_speed: Optional[float] = None       # [m/s] its speed


@dataclass
class Stop:
    x: float
    hold: float = 3.0
    label: str = "stop"
    released: bool = False
    overrun: bool = False
    still: float = 0.0
    error: Optional[float] = None     # car position - stop point, when released


@dataclass
class LeadCar:
    """A car ahead that follows its own speed schedule [(t, v), ...] (linear in between)."""

    x0: float                                  # [m] its rear bumper at t = 0
    schedule: List[Tuple[float, float]]
    leaves_at: float = math.inf                # [m] it turns off the road here
    x: float = 0.0
    v: float = 0.0

    def __post_init__(self):
        self.x = self.x0
        self.v = self.schedule[0][1]

    def step(self, t: float, dt: float) -> None:
        ts = [p[0] for p in self.schedule]
        vs = [p[1] for p in self.schedule]
        self.v = float(np.interp(t, ts, vs))
        self.x += self.v * dt

    @property
    def gone(self) -> bool:
        return self.x > self.leaves_at


def speed_plan(x: np.ndarray, vmax: np.ndarray, v_start: float, anchors) -> np.ndarray:
    """Forward/backward pass: v <= vmax, accel/decel limits, v <= v_a at each anchor (x_a, v_a)."""
    vmax = vmax.copy()
    for xa, va in anchors:
        i = int(np.argmin(np.abs(x - xa)))
        vmax[i] = min(vmax[i], va)
    v = vmax.copy()
    v[0] = min(v[0], v_start)
    for i in range(len(x) - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2 * ACCEL * (x[i + 1] - x[i])))
    for i in range(len(x) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * DECEL * (x[i + 1] - x[i])))
    return v


class Road:
    def __init__(self, length: float, speed_limit: Callable[[float], float],
                 stops: List[Stop], v_start: float = 0.0, lead: Optional[LeadCar] = None,
                 seed: int = 0):
        n = int(math.floor(length / SPACING))
        self.x = np.arange(n + 1) * SPACING
        if length - self.x[-1] > 1e-6:
            self.x = np.append(self.x, length)
        self.length = length
        self.limit = np.array([speed_limit(xi) for xi in self.x])
        vmax = self.limit.copy()
        vmax[-1] = 0.0
        self.vmax = vmax
        self.v_start = v_start
        self.stops = sorted(stops, key=lambda s: s.x)
        self.lead = lead
        self.rng = np.random.default_rng(seed + 777)
        self._restarts: List[Tuple[float, float]] = []
        self._rebuild()

    def _rebuild(self) -> None:
        vmax = self.vmax.copy()
        stop = self.next_stop()
        if stop is not None:
            vmax[self.x >= stop.x - 1e-9] = 0.0     # nothing is planned past an active stop
        self.v = speed_plan(self.x, vmax, max(self.v_start, LAUNCH), self._restarts)

    def limit_at(self, x: float) -> float:
        return float(np.interp(x, self.x, self.limit))

    def plan_at(self, x: float) -> float:
        return float(np.interp(x, self.x, self.v))

    def next_stop(self) -> Optional[Stop]:
        for s in self.stops:
            if not s.released:
                return s
        return None

    def nominal_time(self) -> float:
        v = speed_plan(self.x, self.vmax, max(self.v_start, LAUNCH), [(s.x, 0.0) for s in self.stops])
        vv = v[1:] + v[:-1]
        dx = np.diff(self.x)
        ok = vv > 1e-6
        return float(np.sum(2 * dx[ok] / vv[ok]) + sum(s.hold for s in self.stops))

    # ------------------------------------------------------------------ #
    def target(self, x_car: float, v_meas: float) -> SpeedTarget:
        stop = self.next_stop()
        x_stop = stop.x if stop is not None else self.length
        v = self.plan_at(x_car)
        # target.a is how fast the target changes as YOU drive: the slope of the
        # plan times your speed.  Standing still, the target is not changing.
        slope = (self.plan_at(x_car + SPACING) - v) / SPACING
        a = slope * max(v_meas, 0.0)
        dist = x_stop - x_car
        tgt = SpeedTarget(v=v, a=a, stop_distance=dist if dist <= STOP_VISIBLE else None)
        if self.lead is not None and not self.lead.gone:
            gap = self.lead.x - (x_car + V.REAR_AXLE_TO_FRONT_BUMPER)
            if gap <= LEAD_VISIBLE:
                tgt.lead_distance = gap + self.rng.normal(0.0, 0.10)
                tgt.lead_speed = max(0.0, self.lead.v + self.rng.normal(0.0, 0.10))
        return tgt

    def update(self, x_car: float, v_car: float, dt: float) -> None:
        """Stop bookkeeping with the TRUE car state."""
        changed = False
        for s in self.stops:
            if s.released:
                continue
            if x_car > s.x + STOP_WINDOW and s.still < s.hold:
                s.overrun, s.released, s.error = True, True, x_car - s.x
                self._restarts.append((self._grid(x_car), max(v_car, LAUNCH)))
                changed = True
            elif v_car < STOPPED and abs(x_car - s.x) <= STOP_WINDOW:
                s.still += dt
                if s.still >= s.hold:
                    s.released, s.error = True, x_car - s.x
                    self._restarts.append((self._grid(x_car), LAUNCH))
                    changed = True
            else:
                s.still = 0.0
            break    # only the next stop matters
        if changed:
            self._rebuild()

    def _grid(self, xq: float) -> float:
        return float(self.x[max(0, int(np.searchsorted(self.x, xq)) - 1)])
