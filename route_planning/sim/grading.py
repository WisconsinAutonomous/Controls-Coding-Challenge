"""Route validation and scoring."""

from __future__ import annotations

import math
from typing import FrozenSet, Optional, Tuple

import numpy as np

from .oracle import edge_cost, heading_ok
from .town import Edge, Town

# Points: up to 90 per trip for how close the route is to the best possible
# one, averaged over the trips, plus up to 10 per scenario for planning time.
OPT_GOOD, OPT_BAD = 1.00, 1.25        # route cost / optimal cost


def lin(value: float, good: float, bad: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def validate(town: Town, route, start_edge: Edge, goal: int, cost: str,
             closed: FrozenSet[int] = frozenset(),
             heading: Optional[float] = None) -> Tuple[bool, str, float]:
    """Check a route and return (ok, message, cost)."""
    if route is None:
        return False, "no route returned (None), but the destination is reachable", math.inf
    try:
        route = [int(n) for n in route]
    except (TypeError, ValueError):
        return False, "the route must be a list of node ids", math.inf
    if not route:
        return False, "the route is empty", math.inf
    start = start_edge.end
    if route[0] != start:
        return False, (f"the route starts at node {route[0]}, but the car is driving toward node "
                       f"{start}: start from start_node_from_pose()"), math.inf
    total = 0.0
    prev = start_edge.start
    for u, v in zip(route[:-1], route[1:]):
        e = town.edge_between(u, v)
        if e is None:
            if town.edge_between(v, u) is not None:
                back = town.edge_between(v, u)
                return False, (f"the route goes {u} -> {v} on {back.street}, which is one-way "
                               f"({v} -> {u} only)"), math.inf
            return False, f"the route jumps from node {u} to node {v}, but no street connects them", math.inf
        if e.id in closed:
            return False, f"the route uses {e.street} from {u} to {v}, which is closed", math.inf
        if not town.turn_allowed(prev, u, v):
            kind = town.turn_type(prev, u, v).replace("_", "-")
            return False, f"the route makes a forbidden {kind} at node {u} ({prev} -> {u} -> {v})", math.inf
        total += edge_cost(e, cost) + town.turn_cost(prev, u, v)
        prev = u
    if route[-1] != goal:
        return False, f"the route ends at node {route[-1]}, but the destination is node {goal}", math.inf
    last = start_edge if len(route) == 1 else town.edge_between(route[-2], route[-1])
    if heading is not None and not heading_ok(town, last, heading):
        return False, (f"the car arrives at the destination driving the wrong way on {last.street} "
                       f"(it must arrive heading {math.degrees(heading):.0f} deg ENU)"), math.inf
    return True, "ok", total


def trip_points(ok: bool, ratio: float) -> float:
    return 90.0 * lin(ratio, OPT_GOOD, OPT_BAD) if ok else 0.0


def time_points(total_time: float, good: float, bad: float, valid_fraction: float) -> float:
    return 10.0 * lin(total_time, good, bad) * valid_fraction
