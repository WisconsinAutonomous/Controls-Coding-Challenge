"""World description: the HD-map route the planner is given, and the true world.

Everything a planner may use is in :class:`RouteMap` (known in advance, like
our HD map) and in the per-tick :class:`Observation`.  The true obstacle
geometry, light schedules and pedestrian motion live in :class:`World` and are
only used by the simulator and the grader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from common.geometry import Polyline
from common.messages import CarState, LightColor, ObjectArray

# Lane-line kinds.  Dashed lines may be crossed, everything else may not.
SOLID_WHITE = "solid_white"
DASHED_WHITE = "dashed_white"
SOLID_YELLOW = "solid_yellow"
DOUBLE_YELLOW = "double_yellow"
DASHED_YELLOW = "dashed_yellow"
CROSSABLE_KINDS = {DASHED_WHITE, DASHED_YELLOW}


@dataclass
class LaneLineSpan:
    """One stretch of a lane line.  ``side`` is 'left' or 'right' of the ego lane."""

    side: str
    s_start: float
    s_end: float
    kind: str


@dataclass
class SpeedLimitZone:
    s_start: float
    s_end: float
    max_speed_mps: float


@dataclass
class StopLine:
    """A stop line across the ego lane at arc length ``s``.

    ``kind`` is 'stop_sign', 'traffic_light' or 'crosswalk'.  For crosswalks,
    the painted crosswalk spans ``crosswalk_s_start`` to ``crosswalk_s_end``
    (the stop line itself is a few meters before it).
    """

    id: int
    s: float
    kind: str
    crosswalk_s_start: Optional[float] = None
    crosswalk_s_end: Optional[float] = None


@dataclass
class RouteMap:
    """What the HD map tells you about the road ahead.  Given once, never changes.

    * ``reference``: centerline of the EGO lane (the lane you should normally
      drive in), as a :class:`common.geometry.Polyline` in world coordinates.
      Frenet ``d`` is measured from it, positive to the left.
    * The ego lane spans ``d`` in [-lane_width/2, +lane_width/2].
    * ``left_lane``: None, 'same_direction' or 'opposite_direction'.  If it
      exists it spans ``d`` in [+lane_width/2, +3*lane_width/2].
    * The right edge line (d = -lane_width/2) and the outer edge of the left
      lane are never crossable.  The line between the two lanes is described
      by ``lane_lines`` spans on side 'left'.
    """

    reference: Polyline
    lane_width: float
    lane_lines: List[LaneLineSpan]
    left_lane: Optional[str]
    speed_limits: List[SpeedLimitZone]
    stop_lines: List[StopLine]
    goal_s: float
    default_speed_limit: float = 10.0

    def line_at(self, side: str, s: float) -> str:
        """Kind of the ``side`` ('left'/'right') ego-lane line at arc length ``s``."""
        for span in self.lane_lines:
            if span.side == side and span.s_start <= s < span.s_end:
                return span.kind
        return SOLID_WHITE

    def crossable(self, side: str, s: float) -> bool:
        """True if you may drive across the ``side`` ego-lane line at ``s``."""
        if side == "left" and self.left_lane is None:
            return False
        return self.line_at(side, s) in CROSSABLE_KINDS

    def speed_limit_at(self, s: float) -> float:
        """Speed limit [m/s] at arc length ``s`` (the lowest zone that applies)."""
        limit = self.default_speed_limit
        for z in self.speed_limits:
            if z.s_start <= s < z.s_end:
                limit = min(limit, z.max_speed_mps)
        return limit

    def crossable_array(self, side: str, s: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`crossable`."""
        s = np.asarray(s, dtype=float)
        out = np.zeros(s.shape, dtype=bool)
        if side == "left" and self.left_lane is None:
            return out
        for span in self.lane_lines:
            if span.side == side and span.kind in CROSSABLE_KINDS:
                out |= (s >= span.s_start) & (s < span.s_end)
        return out

    def left_limit_array(self, s: np.ndarray) -> np.ndarray:
        """Largest d any part of the car may reach at arc length(s) ``s``."""
        half = 0.5 * self.lane_width
        cross = self.crossable_array("left", s)
        return np.where(cross, 3.0 * half, half)


@dataclass
class Observation:
    """What your planner sees every tick (10 Hz)."""

    t: float
    ego: CarState
    objects: ObjectArray


# ---------------------------------------------------------------------- #
# True world (simulator / grader only)
# ---------------------------------------------------------------------- #
@dataclass
class Obstacle:
    """A static obstacle: an oriented box in world coordinates."""

    id: int
    obj_class: int
    x: float
    y: float
    yaw: float
    length: float   # along yaw
    width: float    # across yaw
    height: float = 1.0
    s: float = 0.0  # where it sits along the route (for plotting and grading)
    d: float = 0.0

    def corners(self) -> np.ndarray:
        """(4, 2) world corners, counter-clockwise."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hl, hw = 0.5 * self.length, 0.5 * self.width
        local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([self.x, self.y])


@dataclass
class TrafficLight:
    """A signal controlling the stop line ``stop_line_id``.

    The schedule is a list of (time, color) switch points.  ``trigger`` lets a
    scenario start the schedule relative to the car's approach:

    * ``('time_to_line', T, D)``: switch to yellow when the car's front bumper
      is less than T seconds (at its current speed) or D meters from the line.
    * ``('distance', D)``: start the schedule when the front bumper is within
      D meters of the line.
    Before the trigger fires the light shows ``initial``.
    """

    id: int
    stop_line_id: int
    x: float
    y: float
    z: float
    initial: int
    trigger: Optional[tuple] = None
    # phases after the trigger, as (duration, color); the last one lasts forever
    phases: List[Tuple[float, int]] = field(default_factory=list)
    t_trigger: Optional[float] = None

    def color(self, t: float) -> int:
        if self.t_trigger is None or t < self.t_trigger:
            return self.initial
        elapsed = t - self.t_trigger
        for dur, col in self.phases:
            if elapsed < dur:
                return col
            elapsed -= dur
        return self.phases[-1][1] if self.phases else self.initial

    def switch_times(self) -> List[Tuple[float, int]]:
        """Absolute (time, new color) switch points once triggered."""
        if self.t_trigger is None:
            return []
        out, t = [], self.t_trigger
        for dur, col in self.phases:
            out.append((t, col))
            t += dur
        return out


@dataclass
class Pedestrian:
    """A pedestrian that (optionally) walks straight across the road.

    Walks from (s, d_start) to (s, d_end) at ``speed`` once the car's front
    bumper is less than ``trigger_time`` seconds (at its current speed) or
    ``trigger_distance`` meters from ``s``.  A pedestrian with neither
    trigger never moves.
    """

    id: int
    s: float
    d_start: float
    d_end: float
    speed: float = 1.2
    trigger_time: Optional[float] = None
    trigger_distance: Optional[float] = None
    radius: float = 0.3
    t_start: Optional[float] = None
    crosses: bool = True

    def d_at(self, t: float) -> float:
        if self.t_start is None or t <= self.t_start:
            return self.d_start
        step = self.speed * (t - self.t_start)
        span = self.d_end - self.d_start
        return self.d_start + math.copysign(min(step, abs(span)), span)


@dataclass
class World:
    """Everything that is really out there."""

    route: RouteMap
    obstacles: List[Obstacle] = field(default_factory=list)
    lights: List[TrafficLight] = field(default_factory=list)
    pedestrians: List[Pedestrian] = field(default_factory=list)

    def update_triggers(self, t: float, fb_s: float, v: float) -> None:
        """Fire light / pedestrian triggers given the front-bumper arc length."""
        stop_s = {sl.id: sl.s for sl in self.route.stop_lines}
        for L in self.lights:
            if L.t_trigger is not None or L.trigger is None:
                continue
            dist = stop_s[L.stop_line_id] - fb_s
            kind = L.trigger[0]
            if kind == "time_to_line":
                T, D = L.trigger[1], L.trigger[2]
                if 0.0 <= dist and (dist / max(v, 0.1) <= T or dist <= D):
                    L.t_trigger = t
            elif kind == "distance":
                if dist <= L.trigger[1]:
                    L.t_trigger = t
        for p in self.pedestrians:
            if p.t_start is not None or not p.crosses:
                continue
            dist = p.s - fb_s
            if ((p.trigger_time is not None and 0.0 <= dist <= p.trigger_time * max(v, 0.1))
                    or (p.trigger_distance is not None and dist <= p.trigger_distance)):
                p.t_start = t

    def pedestrian_xy(self, p: Pedestrian, t: float) -> Tuple[float, float]:
        x, y = self.route.reference.frenet_to_xy(p.s, p.d_at(t))
        return float(x), float(y)

    def light_for_stop_line(self, stop_line_id: int) -> Optional[TrafficLight]:
        for L in self.lights:
            if L.stop_line_id == stop_line_id:
                return L
        return None


def light_color_name(c: int) -> str:
    return {LightColor.RED: "red", LightColor.YELLOW: "yellow",
            LightColor.GREEN: "green"}.get(c, "unknown")
