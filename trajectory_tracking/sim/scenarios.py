"""Test scenarios.  Every route is synthetic (built from straights, clothoids
and arcs) and every one ends with a stop at the end of the route."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from common.geometry import Polyline, TrackBuilder
from .reference import Stop


@dataclass
class Scenario:
    name: str
    description: str
    route: Polyline
    speed_limit: Callable[[float], float]
    stops: List[Stop] = field(default_factory=list)
    v0: float = 0.0                 # initial speed [m/s]
    d0: float = 0.0                 # initial lateral offset from the route [m], + = left
    dpsi0: float = 0.0              # initial heading error [rad], + = CCW of the route
    grade: Optional[Callable[[float], float]] = None   # grade (rise/run) vs distance driven
    core: bool = True
    lat_grace: float = 0.0          # [s] ignore lateral error for this long (recovery)
    chrono_ok: bool = True          # can run on the optional Chrono backend


def _const(v: float) -> Callable[[float], float]:
    return lambda s: v


def stop_sign() -> Scenario:
    route = TrackBuilder(yaw0=math.radians(35)).straight(150).build()
    return Scenario(
        "stop_sign", "Start from rest, stop at a stop sign, stop at the end. Longitudinal control.",
        route, _const(8.0), [Stop(90.0, hold=3.0, label="stop sign")], v0=0.0)


def city_blocks() -> Scenario:
    route = (TrackBuilder(yaw0=math.radians(150))
             .straight(40).turn(12, -90, 5).straight(50).turn(15, 90, 6)
             .straight(20).turn(20, -45, 5).turn(20, 45, 5).straight(25)
             .turn(40, 120, 10).straight(40).build())
    return Scenario(
        "city_blocks", "Right and left turns, an S-bend, and a long sweeper. Heading crosses +/-180 deg.",
        route, _const(9.0), v0=0.0)


def lane_change() -> Scenario:
    route = (TrackBuilder(yaw0=math.radians(-20))
             .straight(50).lane_change(3.5, 35).straight(25).lane_change(-3.5, 35)
             .straight(60).build())
    return Scenario(
        "lane_change", "Double lane change at 10 m/s. Transient response with a slow steering actuator.",
        route, _const(10.0), v0=10.0)


def hairpin() -> Scenario:
    route = (TrackBuilder(yaw0=math.radians(80))
             .straight(25).turn(7.0, 180, 5).straight(20).turn(6.5, -180, 5)
             .straight(25).build())
    return Scenario(
        "hairpin", "Two tight U-turns. Needs most of the steering range at the steering rate limit.",
        route, _const(5.0), v0=0.0)


def _hill_grade(odo: float) -> float:
    pts = [(0, 0.0), (40, 0.0), (50, 0.08), (130, 0.08), (150, -0.08),
           (210, -0.08), (220, 0.0), (1e9, 0.0)]
    for (s0, g0), (s1, g1) in zip(pts[:-1], pts[1:]):
        if s0 <= odo < s1:
            return g0 + (g1 - g0) * (odo - s0) / (s1 - s0)
    return 0.0


def hill() -> Scenario:
    route = TrackBuilder(yaw0=math.radians(-110)).straight(260).build()
    return Scenario(
        "hill", "8% climb with a stop halfway up, then an 8% descent. Uses CarState.pitch.",
        route, _const(7.0), [Stop(100.0, hold=3.0, label="stop on hill")], v0=0.0,
        grade=_hill_grade, core=False, chrono_ok=False)


def late_stop() -> Scenario:
    route = (TrackBuilder(yaw0=math.radians(10))
             .straight(60).turn(150, 30, 20).straight(170).build())
    return Scenario(
        "late_stop", "12 m/s cruise; two stops that the planner only sees 40 m and 28 m ahead.",
        route, _const(12.0),
        [Stop(150.0, hold=5.0, reveal_distance=40.0, label="red light"),
         Stop(250.0, hold=2.0, reveal_distance=28.0, label="pedestrian")],
        v0=12.0, core=False)


def recovery() -> Scenario:
    route = (TrackBuilder(yaw0=0.0)
             .straight(30).turn(25, 60, 8).turn(25, -60, 8).straight(60).build())
    return Scenario(
        "recovery", "Start 1 m right of the path, 12 deg off heading, at 6 m/s, then an S-bend.",
        route, _const(7.0), v0=6.0, d0=-1.0, dpsi0=math.radians(12), core=False, lat_grace=3.0)


SCENARIOS: Dict[str, Callable[[], Scenario]] = {
    "stop_sign": stop_sign,
    "city_blocks": city_blocks,
    "lane_change": lane_change,
    "hairpin": hairpin,
    "hill": hill,
    "late_stop": late_stop,
    "recovery": recovery,
}
