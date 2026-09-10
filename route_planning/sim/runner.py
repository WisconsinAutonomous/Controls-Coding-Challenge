"""Runs your plan_route() on every trip of a scenario and grades it."""

from __future__ import annotations

import copy
import math
import numbers
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

from .explore import compare_searches
from .grading import time_points, trip_points, validate
from .oracle import Oracle
from .scenarios import Scenario, Trip, pose_on
from .town import closure_of


class BadReturn(Exception):
    """plan_route returned something that is not a route."""


@dataclass
class TripResult:
    index: int
    destination: str
    ok: bool
    message: str
    ratio: float = math.nan          # your cost / optimal cost
    points: float = 0.0
    routes: List[List[int]] = field(default_factory=list)   # 1 route, or 2 with a replan
    closures: List[FrozenSet[int]] = field(default_factory=list)
    plan_time: float = 0.0
    explored: Optional[dict] = None  # information only: your dijkstra()/astar() on this trip


@dataclass
class ScenarioResult:
    scenario: Scenario
    trips: List[TripResult]
    total_time: float
    score: float
    status: str                      # PASS, FAIL, TIMEOUT
    missing_parts: List[str] = field(default_factory=list)


def _as_route(out):
    """What plan_route returned, as a list of node ids (or None).  Raises BadReturn."""
    if (isinstance(out, (tuple, list)) and len(out) in (2, 3)
            and (out[0] is None or isinstance(out[0], (list, tuple)))):
        out = out[0]          # (path, expanded) or (path, cost, expanded): use the path
    if out is None:
        return None
    if isinstance(out, (set, frozenset)):
        raise BadReturn("plan_route returned a set, but a route must be an ORDERED list of node ids")
    if isinstance(out, (str, bytes, dict)) or not hasattr(out, "__iter__"):
        raise BadReturn(f"plan_route should return a list of node ids (or None), got a {type(out).__name__}")
    route = list(out)
    if route and isinstance(route[0], (list, tuple)):
        raise BadReturn("plan_route returned a list of lists: return just the path, "
                        "a flat list of node ids like [12, 7, 3]")
    for n in route:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            if isinstance(n, float) and n.is_integer():
                continue
            raise BadReturn(f"a route must contain node ids (whole numbers), but it contains {n!r}")
    return [int(n) for n in route]


def _explain(exc: BaseException, planner_file: str) -> str:
    """One friendly line about an exception raised by the candidate's code."""
    if isinstance(exc, NotImplementedError):
        return str(exc) or "a part is not implemented yet"
    if isinstance(exc, BadReturn):
        return str(exc)
    tb = traceback.extract_tb(exc.__traceback__)
    mine = os.path.abspath(planner_file) if planner_file else ""
    frames = [f for f in tb if os.path.abspath(f.filename) == mine]
    where = f" at {os.path.basename(mine)} line {frames[-1].lineno}" if frames else ""
    text = str(exc)
    changes_town = "mappingproxy" in text or (
        "'tuple' object" in text and any(w in text for w in
                                         ("item assignment", "item deletion", "'append'", "'remove'",
                                          "'pop'", "'insert'", "'extend'", "'clear'", "'sort'")))
    hint = " (the town is read-only: build your own structures instead of changing it)" if changes_town else ""
    return f"your code raised {type(exc).__name__}: {exc}{where}{hint}"


def _plan(planner, scn: Scenario, pose, letter: str, closed):
    """Call plan_route on a private copy of the town; returns (route, seconds)."""
    town = copy.copy(scn.town)          # a planner that reassigns town.* cannot affect later trips
    t0 = time.perf_counter()
    out = planner.plan_route(town, pose, letter, scn.cost, frozenset(closed))
    return _as_route(out), time.perf_counter() - t0


def _point_along(scn: Scenario, route: List[int], frac: float):
    """Edge and pose of the car after driving ``frac`` of the route's length."""
    town = scn.town
    legs = [town.edge_between(u, v) for u, v in zip(route[:-1], route[1:])]
    total = sum(e.length for e in legs)
    goal, acc = frac * total, 0.0
    for k, e in enumerate(legs):
        if acc + e.length >= goal or k == len(legs) - 1:
            return k, e, pose_on(town, e, 0.4)
        acc += e.length
    return None


def _run_trip(planner, scn: Scenario, k: int, trip: Trip) -> TripResult:
    town = scn.town
    pfile = getattr(planner, "__file__", "")
    goal = town.destinations[trip.destination].node
    heading = town.destinations[trip.destination].heading if scn.check_heading else None
    res = TripResult(k, trip.destination, False, "")
    closed = frozenset(trip.closed)
    res.closures.append(closed)
    try:
        route, dt = _plan(planner, scn, trip.pose, trip.destination, closed)
        res.plan_time += dt
    except Exception as exc:  # noqa: BLE001  (report anything the candidate raises)
        res.message = _explain(exc, pfile)
        return res
    ok, msg, cost = validate(town, route, trip.start_edge, goal, scn.cost, closed, heading)
    if route is not None:
        res.routes.append(route)
    if not ok:
        res.message = msg
        return res
    best = Oracle(town, scn.cost, closed).best_cost(trip.start_edge, goal, heading)
    ratio = cost / best if best > 0 else 1.0

    if trip.replan_at is not None and len(route) >= 4:
        # Drive part of the way, then a street ahead of you closes.
        k_leg, edge_now, pose_now = _point_along(scn, route, trip.replan_at)
        ahead = [town.edge_between(u, v) for u, v in zip(route[k_leg + 1:-1], route[k_leg + 2:])]
        new_closed = None
        for e in ahead[1:] + ahead[:1]:
            cl = closed | closure_of(town, e)
            if edge_now.id in cl:
                continue
            if math.isfinite(Oracle(town, scn.cost, cl).best_cost(edge_now, goal, heading)):
                new_closed = cl
                break
        if new_closed is not None:
            res.closures.append(new_closed)
            try:
                route2, dt = _plan(planner, scn, pose_now, trip.destination, new_closed)
                res.plan_time += dt
            except Exception as exc:  # noqa: BLE001
                res.message = "after the new closure: " + _explain(exc, pfile)
                return res
            if route2 is not None:
                res.routes.append(route2)
            ok2, msg2, cost2 = validate(town, route2, edge_now, goal, scn.cost, new_closed, heading)
            if not ok2:
                res.message = "after the new closure: " + msg2
                return res
            best2 = Oracle(town, scn.cost, new_closed).best_cost(edge_now, goal, heading)
            ratio = 0.5 * (ratio + (cost2 / best2 if best2 > 0 else 1.0))

    res.ok, res.message, res.ratio = True, "ok", ratio
    return res


def missing_parts(planner) -> List[str]:
    """Which TODO functions still raise NotImplementedError (probed on a tiny town)."""
    from .town import Destination, Edge as E, Node, Town
    tiny = Town({0: Node(0, 0, 0), 1: Node(1, 100, 0)},
                [E(0, 0, 1, 100.0, 10.0, "Test St"), E(1, 1, 0, 100.0, 10.0, "Test St")],
                {"A": Destination("A", 1, None, "Test St")})
    graph = {0: [(1, 100.0)], 1: [(0, 100.0)]}
    probes = [("Part 1 (build_graph)", lambda: planner.build_graph(tiny)),
              ("Part 2 (dijkstra)", lambda: planner.dijkstra(graph, 0, 1)),
              ("Part 3 (straight_line_heuristic)", lambda: planner.straight_line_heuristic(tiny, 1)),
              ("Part 3 (astar)", lambda: planner.astar(graph, 0, 1, lambda n: 0.0))]
    missing = []
    for name, fn in probes:
        try:
            fn()
        except NotImplementedError:
            missing.append(name)
        except Exception:  # noqa: BLE001  (broken is not missing; check.py explains)
            pass
    return missing


def run_scenario(scn: Scenario, planner, measure: bool = True) -> ScenarioResult:
    """Grade every trip.  ``measure`` also records, for information only, how many
    nodes the candidate's own dijkstra() and astar() explore (core scenarios)."""
    results: List[TripResult] = []
    total_time = 0.0
    for k, trip in enumerate(scn.trips):
        r = _run_trip(planner, scn, k, trip)
        r.points = trip_points(r.ok, r.ratio)
        results.append(r)
        total_time += r.plan_time
        if total_time > scn.time_limit:
            for j in range(k + 1, len(scn.trips)):
                results.append(TripResult(j, scn.trips[j].destination, False, "not run: time limit"))
            break
    if measure and scn.core:
        for r, trip in zip(results, scn.trips):
            goal = scn.town.destinations[trip.destination].node
            r.explored = compare_searches(planner, copy.copy(scn.town), scn.cost, trip.closed,
                                          trip.start_edge.end, goal)
    n_ok = sum(r.ok for r in results)
    frac = n_ok / len(results) if results else 0.0
    if total_time > scn.time_limit:
        status, score = "TIMEOUT", 0.0
    else:
        status = "PASS" if n_ok == len(results) else "FAIL"
        score = (sum(r.points for r in results) / max(1, len(results))
                 + time_points(total_time, scn.time_good, scn.time_bad, frac))
    return ScenarioResult(scn, results, total_time, score, status, missing_parts(planner))
