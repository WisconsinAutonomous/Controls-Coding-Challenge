"""Stand-in for the mid planner: publishes the ReferenceTrajectory you track.

Every 0.1 s it publishes a window of the route starting at (or just behind)
the car and reaching 50 m ahead, with a speed profile that respects the speed
limit, a lateral-acceleration limit in curves, comfortable accel/decel, and
stops.  While a stop is active the trajectory ENDS at the stop point with zero
speed, exactly like the real planner does for a stop sign or red light.  Once
the car has been stopped there long enough, the stop is released and the
trajectory continues.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from common.geometry import Polyline
from common.messages import CarState, ReferenceTrajectory

SPACING = 0.5          # [m] between published points
HORIZON = 50.0         # [m] published length
LAT_ACCEL_MAX = 2.0    # [m/s^2] comfort limit used for curve speeds
ACCEL_MAX = 1.5        # [m/s^2] planned acceleration
DECEL_MAX = 2.0        # [m/s^2] planned deceleration
STOPPED_SPEED = 0.05   # [m/s] "stopped" threshold for releasing a stop
STOP_WINDOW = 3.0      # [m] must stop within this distance of the stop point


@dataclass
class Stop:
    s: float                        # stop point along the route [m]
    hold: float = 3.0               # [s] required standstill before release
    reveal_distance: float = math.inf  # planner only knows about it within this range
    label: str = "stop"
    # runtime bookkeeping, filled in during the run
    revealed: bool = False
    released: bool = False
    overrun: bool = False
    still_time: float = 0.0
    stop_error: Optional[float] = None   # s_car - s at the moment of release


def speed_profile(s: np.ndarray, vmax: np.ndarray, v_start: float,
                  anchors: List[tuple]) -> np.ndarray:
    """Forward/backward pass: v <= vmax, |a| <= limits, v <= v_anchor at each anchor."""
    vmax = vmax.copy()
    for s_a, v_a in anchors:
        i = int(np.argmin(np.abs(s - s_a)))
        vmax[i] = min(vmax[i], v_a)
    v = vmax.copy()
    v[0] = min(v[0], v_start)
    ds = np.diff(s)
    for i in range(len(s) - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * ACCEL_MAX * ds[i]))
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * DECEL_MAX * ds[i]))
    return v


class ReferenceProvider:
    def __init__(self, route: Polyline, speed_limit, stops: List[Stop], v_start: float):
        self.route = route
        n = int(math.floor(route.length / SPACING))
        s = np.arange(n + 1) * SPACING
        if route.length - s[-1] > 1e-6:
            s = np.append(s, route.length)
        self.s = s
        self.x, self.y, self.yaw = route.interp(s)
        self.kappa = route.curvature_at(s)
        limit = np.array([speed_limit(si) for si in s])
        k = np.maximum(np.abs(self.kappa), 1e-4)
        self.vmax = np.minimum(limit, np.sqrt(LAT_ACCEL_MAX / k))
        self.vmax[-1] = 0.0
        self.v_start = v_start
        self.stops = sorted(stops, key=lambda st: st.s)
        self._restart: List[tuple] = []   # (s, v) where the car carried on after a stop
        self._last_s = 0.0
        self._rebuild()

    # ------------------------------------------------------------------ #
    def _anchors(self) -> List[tuple]:
        a = [(st.s, 0.0) for st in self.stops if st.revealed and not st.released]
        return a + list(self._restart)

    def _rebuild(self) -> None:
        self.v = speed_profile(self.s, self.vmax, self.v_start, self._anchors())

    def _grid_at(self, s_query: float) -> float:
        """The published grid point at or just behind ``s_query``."""
        return float(self.s[max(0, int(np.searchsorted(self.s, s_query)) - 1)])

    def v_ref_at(self, s_query: float) -> float:
        return float(np.interp(s_query, self.s, self.v))

    def active_stop(self) -> Optional[Stop]:
        for st in self.stops:
            if st.revealed and not st.released:
                return st
        return None

    def project(self, x: float, y: float) -> tuple:
        s, d, _ = self.route.project(x, y, s_hint=self._last_s, window=15.0)
        self._last_s = s
        return s, d

    # ------------------------------------------------------------------ #
    def update(self, s_true: float, v_true: float, dt: float) -> None:
        """Advance stop logic using the TRUE car state (called every tick)."""
        changed = False
        for st in self.stops:
            if st.released:
                continue
            if not st.revealed and s_true >= st.s - st.reveal_distance:
                st.revealed = True
                changed = True
            if not st.revealed:
                continue
            if s_true > st.s + STOP_WINDOW and st.still_time < st.hold:
                # Ran through the stop: record it and re-plan from the car's
                # current speed so the reference does not jump to cruise.
                st.overrun, st.released = True, True
                st.stop_error = s_true - st.s
                self._restart.append((self._grid_at(s_true), v_true))
                changed = True
                continue
            if v_true < STOPPED_SPEED and abs(s_true - st.s) <= STOP_WINDOW:
                st.still_time += dt
                if st.still_time >= st.hold:
                    st.released = True
                    st.stop_error = s_true - st.s
                    self._restart.append((self._grid_at(s_true), 0.0))
                    changed = True
            else:
                st.still_time = 0.0
        if changed:
            self._rebuild()

    def publish(self, state: CarState, stamp: float) -> ReferenceTrajectory:
        """The trajectory the planner would publish for this (measured) state."""
        s_car, _ = self.project(state.x, state.y)
        i0 = max(0, int(math.floor(s_car / SPACING)))
        i1 = min(len(self.s) - 1, i0 + int(HORIZON / SPACING))
        st = self.active_stop()
        if st is not None:
            # While a stop is active the trajectory always ENDS at the stop
            # point, even if the car is already on top of it.
            i_stop = int(np.argmin(np.abs(self.s - st.s)))
            i1 = min(i1, i_stop)
        i1 = max(i1, 1)
        i0 = min(i0, i1 - 1)
        sl = slice(i0, i1 + 1)
        v = self.v[sl]
        ds = np.diff(self.s[sl])
        a = np.zeros_like(v)
        a[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds)
        a[-1] = a[-2] if len(a) > 1 else 0.0
        dt = 2.0 * ds / np.maximum(v[1:] + v[:-1], 1e-3)
        t_rel = np.concatenate(([0.0], np.cumsum(np.minimum(dt, 1e3))))
        return ReferenceTrajectory.from_arrays(
            self.x[sl], self.y[sl], self.yaw[sl], v, a, self.kappa[sl], t_rel,
            stamp=stamp)

    def nominal_time(self) -> float:
        """Time to drive the full static profile (all stops honored) plus holds."""
        anchors = [(st.s, 0.0) for st in self.stops]
        v = speed_profile(self.s, self.vmax, self.v_start, anchors)
        ds = np.diff(self.s)
        vv = v[1:] + v[:-1]
        t = np.sum(2.0 * ds[vv > 1e-6] / vv[vv > 1e-6])
        return float(t + sum(st.hold for st in self.stops))
