"""Scenarios: a town, a cost type, and a list of trips.  ``seed`` changes the
town and the trips; seed 0 is the public one."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, FrozenSet, List, Optional

import numpy as np

from common.messages import CarState
from .oracle import Oracle, edge_cost, heading_ok
from .town import MAIN, Edge, Town, closure_of, generate_town


@dataclass
class Trip:
    start_edge: Edge                         # the car starts 30% along this edge
    destination: str                         # letter
    pose: CarState                           # where the car is when you plan
    closed: FrozenSet[int] = frozenset()     # closures known before you plan
    replan_at: Optional[float] = None        # fraction of the route after which a new closure appears


def pose_on(town: Town, e: Edge, frac: float) -> CarState:
    a, b = town.nodes[e.start], town.nodes[e.end]
    return CarState(x=a.x + frac * (b.x - a.x), y=a.y + frac * (b.y - a.y), v=8.0,
                    psi=math.atan2(b.y - a.y, b.x - a.x))


@dataclass
class Scenario:
    name: str
    description: str
    core: bool
    town: Town
    cost: str                       # "distance" or "time"
    trips: List[Trip]
    check_heading: bool = False     # arrive_facing: must arrive driving the right way
    time_good: float = 0.5          # [s] total planning time for full time points
    time_bad: float = 10.0          # [s] total planning time for zero time points
    time_limit: float = 120.0       # [s] total planning time before the scenario fails
    seed: int = 0

    def __post_init__(self):
        self.town.freeze()          # planners may read the town, never change it


def _no_headings(town: Town) -> Town:
    """Outside arrive_facing the arrival direction does not matter: heading = None."""
    town.destinations = {k: replace(d, heading=None) for k, d in town.destinations.items()}
    return town


def _edges_ok_for_start(town: Town) -> List[Edge]:
    return [e for e in town.edges if e.length > 40.0]


def _random_trips(town: Town, rng, n: int, cost: str, min_dist: float,
                  accept: Optional[Callable[[Edge, str], bool]] = None, tries: int = 4000) -> List[Trip]:
    starts = _edges_ok_for_start(town)
    letters = sorted(town.destinations)
    trips: List[Trip] = []
    used = set()
    for _ in range(tries):
        if len(trips) >= n:
            break
        e = starts[int(rng.integers(len(starts)))]
        letter = letters[int(rng.integers(len(letters)))]
        dest = town.destinations[letter]
        if (e.id, letter) in used or e.end == dest.node or e.start == dest.node:
            continue
        if town.distance(e.end, dest.node) < min_dist:
            continue
        if accept is not None and not accept(e, letter):
            continue
        used.add((e.id, letter))
        trips.append(Trip(e, letter, pose_on(town, e, 0.3)))
    return trips


# ---------------------------------------------------------------------- #
def first_route(seed: int = 0) -> Scenario:
    town = _no_headings(generate_town(seed, 6, 5, block=110.0, bridges=2, one_way_pairs=1,
                                      n_destinations=8, removed_segments=3, name="Riverside"))
    rng = np.random.default_rng(1000 + seed)
    trips = _random_trips(town, rng, 8, "distance", min_dist=250.0)
    return Scenario("first_route", "A small town. Shortest distance, no surprises.", True,
                    town, "distance", trips, seed=seed)


def rush_hour(seed: int = 0) -> Scenario:
    town = _no_headings(generate_town(seed + 101, 8, 7, block=120.0, bridges=2, ring_road=True,
                                      diagonal=True, one_way_pairs=1, n_destinations=10,
                                      removed_segments=8, name="Lakeview"))
    orc = Oracle(town, "time")
    orc_d = Oracle(town, "distance")
    rng = np.random.default_rng(2000 + seed)

    def differs(e: Edge, letter: str) -> bool:
        goal = town.destinations[letter].node
        p = orc_d.path(e.end, goal)
        if p is None:
            return False
        t_short = sum(edge_cost(town.edge_between(u, v), "time") for u, v in zip(p[:-1], p[1:]))
        return t_short > 1.05 * float(orc.from_node(e.end)[goal])

    trips = _random_trips(town, rng, 8, "time", min_dist=350.0, accept=differs)
    trips += _random_trips(town, rng, 2, "time", min_dist=350.0)
    return Scenario("rush_hour", "Travel time, not distance: the fastest route is not the shortest.",
                    True, town, "time", trips, seed=seed)


def road_closed(seed: int = 0) -> Scenario:
    town = _no_headings(generate_town(seed + 202, 8, 7, block=120.0, bridges=3, ring_road=True,
                                      diagonal=False, one_way_pairs=1, n_destinations=10,
                                      removed_segments=8, name="Millbrook"))
    orc = Oracle(town, "time")
    rng = np.random.default_rng(3000 + seed)
    trips = _random_trips(town, rng, 6, "time", min_dist=450.0)
    for t in trips:
        goal = town.destinations[t.destination].node
        p = orc.path(t.start_edge.end, goal)
        # close a street in the middle of the best route, before you plan
        for k in range(len(p) // 2, len(p) - 1):
            cl = closure_of(town, town.edge_between(p[k], p[k + 1]))
            if t.start_edge.id not in cl and math.isfinite(
                    Oracle(town, "time", cl).best_cost(t.start_edge, goal)):
                t.closed = cl
                break
        t.replan_at = 0.35
    return Scenario("road_closed", "Known closures, plus a new one that appears while you drive.",
                    True, town, "time", trips, seed=seed)


def no_left_turn(seed: int = 0) -> Scenario:
    town = _no_headings(generate_town(seed + 303, 8, 7, block=120.0, bridges=2, ring_road=True,
                                      diagonal=True, one_way_pairs=1, n_destinations=10,
                                      removed_segments=7, name="Hillcrest"))
    rng = np.random.default_rng(4000 + seed)
    town.left_turn_penalty, town.right_turn_penalty, town.u_turns_allowed = 10.0, 3.0, False
    leaving: Dict[int, List[Edge]] = {}
    for e in town.edges:
        leaving.setdefault(e.start, []).append(e)
    for e1 in town.edges:
        v = e1.end
        if len(leaving.get(v, [])) < 3:
            continue
        busy = any(e.speed_limit >= MAIN for e in leaving[v])
        for e2 in leaving[v]:
            if busy and town.turn_type(e1.start, v, e2.end) == "left" and rng.random() < 0.6:
                town.no_left_turns.add((e1.start, v, e2.end))
    plain = Oracle(town, "time")
    turned = Oracle(town, "time")

    def reachable_and_tricky(e: Edge, letter: str) -> bool:
        goal = town.destinations[letter].node
        if not math.isfinite(turned.best_cost(e, goal)):
            return False
        p = plain.path(e.end, goal)
        if p is None:
            return False
        full = [e.start] + p
        return any(not town.turn_allowed(a, b, c) for a, b, c in zip(full, full[1:], full[2:]))

    trips = _random_trips(town, rng, 8, "time", min_dist=350.0, accept=reachable_and_tricky)
    return Scenario("no_left_turn", "Turn penalties, no-left-turn signs, and no U-turns.", False,
                    town, "time", trips, seed=seed)


def arrive_facing(seed: int = 0) -> Scenario:
    town = generate_town(seed + 404, 7, 6, block=120.0, bridges=2, ring_road=False, diagonal=True,
                         one_way_pairs=1, n_destinations=12, removed_segments=6, name="Fairview")
    plain = Oracle(town, "time")
    rng = np.random.default_rng(5000 + seed)

    def wrong_way_if_naive(e: Edge, letter: str) -> bool:
        dest = town.destinations[letter]
        p = plain.path(e.end, dest.node)
        if p is None or len(p) < 2:
            return False
        return not heading_ok(town, town.edge_between(p[-2], p[-1]), dest.heading)

    trips = _random_trips(town, rng, 6, "time", min_dist=300.0, accept=wrong_way_if_naive)
    trips += _random_trips(town, rng, 2, "time", min_dist=300.0)
    return Scenario("arrive_facing", "Arrive driving the right way, so the car ends up on the right side.",
                    False, town, "time", trips, check_heading=True, seed=seed)


def big_city(seed: int = 0) -> Scenario:
    town = _no_headings(generate_town(seed + 505, 60, 60, block=100.0, bridges=6, ring_road=True,
                                      diagonal=True, one_way_pairs=6, n_destinations=26,
                                      removed_segments=500, name="Metropolis"))
    rng = np.random.default_rng(6000 + seed)
    trips = _random_trips(town, rng, 100, "time", min_dist=2500.0)
    return Scenario("big_city", "About 3,600 nodes and 100 trips. Is your search fast enough?", False,
                    town, "time", trips, time_good=3.0, time_bad=15.0, time_limit=20.0, seed=seed)


SCENARIOS: Dict[str, Callable[[int], Scenario]] = {
    "first_route": first_route,
    "rush_hour": rush_hour,
    "road_closed": road_closed,
    "no_left_turn": no_left_turn,
    "arrive_facing": arrive_facing,
    "big_city": big_city,
}
