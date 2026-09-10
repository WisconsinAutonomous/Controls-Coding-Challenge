"""Scenario definitions.

Seed 0 is the nominal layout used for the public scoreboard.  Other seeds
move obstacles, change light timings and pedestrian behavior, and use a
different noise stream.  We grade on seeds you have not seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from common.geometry import Polyline, TrackBuilder
from common.messages import LightColor, ObjClass

from .world import (DASHED_WHITE, DASHED_YELLOW, DOUBLE_YELLOW, SOLID_WHITE,
                    LaneLineSpan, Obstacle, Pedestrian, RouteMap, SpeedLimitZone,
                    StopLine, TrafficLight, World)

LANE_WIDTH = 3.6


@dataclass
class Scenario:
    name: str
    tier: str                 # 'core' or 'stretch'
    description: str
    seed: int
    world: World
    start_s: float
    start_v: float
    time_limit: float
    par_time: Optional[float] = None
    # road_closed: success = stopped behind the closure (see grading)
    closure_face_s: Optional[float] = None
    curb_pedestrian_ids: List[int] = field(default_factory=list)

    @property
    def route(self) -> RouteMap:
        return self.world.route


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _jit(rng, seed: int, lo: float, hi: float) -> float:
    """Uniform jitter, but exactly 0 on the nominal seed."""
    return 0.0 if seed == 0 else float(rng.uniform(lo, hi))


def _box(route: RouteMap, oid: int, cls: int, s: float, d: float,
         length: float, width: float, height: float = 1.0) -> Obstacle:
    x, y = route.reference.frenet_to_xy(s, d)
    _, _, yaw = route.reference.interp(s)
    return Obstacle(oid, cls, float(x), float(y), float(yaw), length, width, height, s, d)


def _barrel(route, oid, s, d):
    return _box(route, oid, ObjClass.BARREL, s, d, 0.6, 0.6, 1.0)


def _cone(route, oid, s, d):
    return _box(route, oid, ObjClass.CONE, s, d, 0.4, 0.4, 0.7)


def _barricade(route, oid, s, d, width):
    return _box(route, oid, ObjClass.BARRICADE, s, d, 0.4, width, 1.1)


def _two_way_lines(length: float, passing: List[tuple] = ()) -> List[LaneLineSpan]:
    """Double yellow on the left except dashed-yellow passing zones."""
    spans, s = [], 0.0
    for a, b in sorted(passing):
        if a > s:
            spans.append(LaneLineSpan("left", s, a, DOUBLE_YELLOW))
        spans.append(LaneLineSpan("left", a, b, DASHED_YELLOW))
        s = b
    spans.append(LaneLineSpan("left", s, length + 50.0, DOUBLE_YELLOW))
    spans.append(LaneLineSpan("right", 0.0, length + 50.0, SOLID_WHITE))
    return spans


def _light(route: RouteMap, lid: int, stop_line: StopLine, **kw) -> TrafficLight:
    x, y = route.reference.frenet_to_xy(stop_line.s + 5.0, 0.0)
    return TrafficLight(id=lid, stop_line_id=stop_line.id, x=float(x), y=float(y),
                        z=5.5, **kw)


def _route(track: Polyline, lines, left_lane, limits, stops, goal_s, default=10.0):
    return RouteMap(reference=track, lane_width=LANE_WIDTH, lane_lines=lines,
                    left_lane=left_lane, speed_limits=limits, stop_lines=stops,
                    goal_s=goal_s, default_speed_limit=default)


# ---------------------------------------------------------------------- #
# scenarios
# ---------------------------------------------------------------------- #
def curvy_road(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(70)).straight(40).turn(30, 35, 10)
             .turn(30, -45, 10).straight(25).turn(15, 90, 8).straight(70)
             .turn(40, -30, 10).straight(45).build())
    zs = 150.0 + _jit(rng, seed, -10, 10)
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [SpeedLimitZone(zs, zs + 60.0, 6.0)], [], track.length - 10.0)
    return Scenario("curvy_road", "core",
                    "S-bends, a tight 90 deg left turn and a 6 m/s school zone.",
                    seed, World(route), 3.0, 0.0, time_limit=130.0, par_time=56.5)


def stop_sign(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(-165)).straight(80).turn(10, -90, 4)
             .straight(65).build())
    sl = StopLine(1, 70.0 + _jit(rng, seed, -5, 3), "stop_sign")
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [], [sl], track.length - 10.0, default=8.0)
    return Scenario("stop_sign", "core",
                    "Full stop at a stop sign, then a right turn through the intersection.",
                    seed, World(route), 3.0, 0.0, time_limit=100.0, par_time=45.0)


def barrel_nudge(seed: int = 0) -> Scenario:
    """In-lane nudges only: every obstacle leaves room to pass inside the lane,
    and the double yellow means a lane change is never an option."""
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(-35)).straight(40).turn(120, 20, 15)
             .straight(70).turn(150, -25, 15).straight(80).turn(200, 10, 10)
             .straight(60).build())
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [], [], track.length - 10.0, default=9.0)
    half = 0.5 * LANE_WIDTH
    j = lambda lo, hi: _jit(rng, seed, lo, hi)

    def inner(intrusion, side, width):
        """Center d of an object whose inner edge reaches ``intrusion`` m into the lane."""
        edge = side * (half - intrusion)
        return edge + side * 0.5 * width

    obs = [
        _barrel(route, 1, 70.0 + j(-6, 6), inner(0.95 + j(-0.04, 0.0), -1, 0.6)),
        _cone(route, 2, 150.0 + j(-6, 6), inner(0.8 + j(-0.1, 0.1), +1, 0.4)),
        _barricade(route, 3, 230.0 + j(-6, 6), inner(0.85 + j(-0.1, 0.1), -1, 1.4), 1.4),
        _barrel(route, 4, 310.0 + j(-6, 6), inner(0.95 + j(-0.04, 0.0), +1, 0.6)),
        _cone(route, 5, 370.0 + j(-5, 5), -2.5),                       # on the shoulder: ignore it
    ]
    return Scenario("barrel_nudge", "core",
                    "Barrels and cones stick partway into your lane: nudge around each one "
                    "without leaving the lane, then come back to the center.",
                    seed, World(route, obstacles=obs), 3.0, 0.0, time_limit=150.0, par_time=65.5)


def barrels(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(130)).straight(40).turn(150, 25, 20)
             .straight(60).turn(120, -30, 20).straight(90).build())
    lines = [LaneLineSpan("left", 0.0, track.length + 50, DASHED_WHITE),
             LaneLineSpan("right", 0.0, track.length + 50, SOLID_WHITE)]
    route = _route(track, lines, "same_direction", [], [], track.length - 10.0, default=9.0)
    j = lambda lo, hi: _jit(rng, seed, lo, hi)
    s1, s2, s3 = 60.0 + j(-4, 4), 120.0 + j(-4, 4), 190.0 + j(-4, 4)
    obs = [
        _barrel(route, 1, s1, -1.3 + j(-0.15, 0.15)),                 # right side, nudge left
        _cone(route, 2, s2 - 2.0, -0.7 + j(-0.15, 0.15)),              # cluster blocks the lane
        _cone(route, 3, s2, 0.0 + j(-0.15, 0.15)),
        _cone(route, 4, s2 + 2.0, 0.7 + j(-0.1, 0.1)),
        _barrel(route, 5, s3, 1.3 + j(-0.15, 0.15)),                  # left side, nudge right
        _cone(route, 6, 250.0 + j(-5, 5), -2.4),                      # on the shoulder: ignore it
    ]
    return Scenario("barrels", "stretch",
                    "Two same-direction lanes: nudge around barrels, change lanes around a cone cluster.",
                    seed, World(route, obstacles=obs), 3.0, 0.0, time_limit=130.0, par_time=55.5)


def _work_zone_track():
    return (TrackBuilder(yaw0=math.radians(20)).straight(60).turn(200, 10, 15)
            .straight(80).turn(200, -10, 15).straight(60).build())


def road_work(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = _work_zone_track()
    b = 100.0 + _jit(rng, seed, -5, 5)
    zone = (60.0, 150.0 + _jit(rng, seed, -5, 5))
    route = _route(track, _two_way_lines(track.length, [zone]), "opposite_direction",
                   [], [], track.length - 10.0)
    obs = [_cone(route, 1, b - 6, -1.2), _cone(route, 2, b - 4, -0.4), _cone(route, 3, b - 2, 0.4),
           _barricade(route, 4, b, -0.4, 2.8),
           _cone(route, 5, b + 2, 1.0), _cone(route, 6, b + 4, 1.0), _cone(route, 7, b + 6, 1.0),
           _cone(route, 8, b + 8, 0.6)]
    return Scenario("road_work", "stretch",
                    "Work zone blocks your lane on a two-way road: pass in the passing zone, "
                    "be back before the double yellow.",
                    seed, World(route, obstacles=obs), 3.0, 0.0, time_limit=100.0, par_time=44.0)


def road_closed(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = _work_zone_track()
    b = 100.0 + _jit(rng, seed, -8, 8)
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [], [], track.length - 10.0)
    obs = [_cone(route, 1, b - 3, -1.2), _cone(route, 2, b - 3, 1.2),
           _barricade(route, 3, b, 0.0, 3.6),
           _cone(route, 4, b + 3, -0.6), _cone(route, 5, b + 3, 0.6)]
    face = min(o.s - 0.5 * o.length for o in obs)
    return Scenario("road_closed", "stretch",
                    "Your lane is closed and you may not cross the double yellow: stop behind it.",
                    seed, World(route, obstacles=obs), 3.0, 0.0, time_limit=70.0,
                    par_time=27.5, closure_face_s=face)


def traffic_light(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(-60)).straight(100).turn(250, 12, 20)
             .straight(90).turn(250, -12, 20).straight(150).build())
    sl1, sl2 = StopLine(1, 150.0, "traffic_light"), StopLine(2, 320.0, "traffic_light")
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [], [sl1, sl2], track.length - 10.0)
    T = 2.6 if seed == 0 else float(rng.uniform(1.4, 4.2))
    G = 9.0 if seed == 0 else float(rng.uniform(7.0, 12.0))
    lights = [
        _light(route, 1, sl1, initial=LightColor.GREEN, trigger=("time_to_line", T, 8.0),
               phases=[(3.0, LightColor.YELLOW), (12.0, LightColor.RED), (1e9, LightColor.GREEN)]),
        _light(route, 2, sl2, initial=LightColor.RED, trigger=("distance", 40.0),
               phases=[(G, LightColor.RED), (1e9, LightColor.GREEN)]),
    ]
    return Scenario("traffic_light", "stretch",
                    "A light turns yellow as you approach (stop or go?), then a red that turns green.",
                    seed, World(route, lights=lights), 5.0, 0.0, time_limit=220.0, par_time=100.0)


def crosswalk(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(200)).straight(90).turn(300, 8, 10)
             .straight(80).turn(300, -8, 10).straight(90).build())
    sl = StopLine(1, 117.0, "crosswalk", crosswalk_s_start=120.0, crosswalk_s_end=124.0)
    route = _route(track, _two_way_lines(track.length), "opposite_direction",
                   [], [sl], track.length - 10.0, default=9.0)
    trig = 3.5 if seed == 0 else float(rng.uniform(2.6, 4.5))
    spd = 1.2 if seed == 0 else float(rng.uniform(1.0, 1.5))
    peds = [Pedestrian(1, 122.0, -3.2, 6.8, speed=spd, trigger_time=trig, trigger_distance=10.0),
            Pedestrian(2, 205.0 + _jit(rng, seed, -10, 10), -3.6, -3.6, speed=0.0,
                       crosses=False)]
    return Scenario("crosswalk", "stretch",
                    "Yield to a pedestrian in a crosswalk; do not stop for one who is just "
                    "standing on the curb.",
                    seed, World(route, pedestrians=peds), 3.0, 0.0, time_limit=150.0, par_time=64.5,
                    curb_pedestrian_ids=[2])


def gauntlet(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    track = (TrackBuilder(yaw0=math.radians(100)).straight(70).turn(60, -20, 10)
             .straight(50).turn(15, 90, 8).straight(130).turn(80, -25, 15)
             .straight(110).turn(80, 15, 10).straight(110).build())
    j = lambda lo, hi: _jit(rng, seed, lo, hi)
    cone_s = 243.0 + j(-6, 6)
    stops = [StopLine(1, 145.0, "stop_sign"),
             StopLine(2, 440.0, "traffic_light"),
             StopLine(3, 540.0, "crosswalk", crosswalk_s_start=543.0, crosswalk_s_end=547.0)]
    route = _route(track, _two_way_lines(track.length, [(200.0, 295.0)]),
                   "opposite_direction", [SpeedLimitZone(20.0, 95.0, 6.0)], stops,
                   track.length - 10.0)
    obs = [_barrel(route, 1, 60.0 + j(-5, 5), -1.3),
           _cone(route, 2, cone_s - 3, -0.9), _cone(route, 3, cone_s, 0.0),
           _cone(route, 4, cone_s + 3, 0.8), _cone(route, 5, cone_s + 6, 0.2)]
    G = 8.0 if seed == 0 else float(rng.uniform(6.0, 11.0))
    lights = [_light(route, 1, stops[1], initial=LightColor.RED, trigger=("distance", 40.0),
                     phases=[(G, LightColor.RED), (1e9, LightColor.GREEN)])]
    trig = 3.5 if seed == 0 else float(rng.uniform(2.6, 4.5))
    peds = [Pedestrian(1, 545.0, -3.2, 6.8, speed=1.2, trigger_time=trig, trigger_distance=10.0),
            Pedestrian(2, 590.0, -3.6, -3.6, speed=0.0, crosses=False)]
    return Scenario("gauntlet", "stretch",
                    "Everything at once: school zone, barrel, stop sign, work zone pass, "
                    "red light and a crosswalk.",
                    seed, World(route, obstacles=obs, lights=lights, pedestrians=peds),
                    3.0, 0.0, time_limit=280.0, par_time=126.5, curb_pedestrian_ids=[2])


SCENARIOS: Dict[str, Callable[[int], Scenario]] = {
    "curvy_road": curvy_road,
    "stop_sign": stop_sign,
    "barrel_nudge": barrel_nudge,
    "barrels": barrels,
    "road_work": road_work,
    "road_closed": road_closed,
    "traffic_light": traffic_light,
    "crosswalk": crosswalk,
    "gauntlet": gauntlet,
}
