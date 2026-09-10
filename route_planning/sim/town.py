"""The town: intersections (nodes), one-direction lane segments (edges),
destinations, and the helpers you are allowed to use.

Everything is synthetic.  A town is generated from a seed: a street grid with
uneven blocks and gently curving streets, a river you can only cross on a few
bridges, some one-way streets and dead ends, fast arterials, and destinations
that sit in the middle of a block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

MPH = 0.44704
RESIDENTIAL = 25 * MPH      # 11.2 m/s
MAIN = 35 * MPH             # 15.6 m/s
ARTERIAL = 45 * MPH         # 20.1 m/s

TREES = ["Maple", "Oak", "Elm", "Pine", "Cedar", "Birch", "Walnut", "Spruce", "Willow",
         "Aspen", "Cherry", "Hickory", "Juniper", "Linden", "Poplar", "Sycamore"]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass(frozen=True)
class Node:
    id: int
    x: float        # [m] East
    y: float        # [m] North


@dataclass(frozen=True)
class Edge:
    """One lane segment you can drive in ONE direction: from ``start`` to ``end``.

    A two-way street is two Edges (one each way).  A one-way street is one Edge.
    """
    id: int
    start: int          # node id you leave from
    end: int            # node id you arrive at
    length: float       # [m]
    speed_limit: float  # [m/s]
    street: str


@dataclass(frozen=True)
class Destination:
    """A place the operator can pick by letter, like our waypoints file."""
    letter: str
    node: int                 # the node you must reach
    heading: Optional[float]  # [rad] ENU direction you must be driving when you arrive,
                              # or None when it does not matter (every scenario but arrive_facing)
    street: str


@dataclass
class Town:
    """Read-only for your planner: build your own structures (like the graph
    in Part 1) instead of changing the town.  Each trip gets its own copy."""

    nodes: Dict[int, Node]
    edges: List[Edge]
    destinations: Dict[str, Destination]
    name: str = "town"
    river: List[Tuple[float, float]] = field(default_factory=list)   # for plotting
    bridges: List[Tuple[int, int]] = field(default_factory=list)     # node pairs, for plotting
    # Turn rules (only used by the no_left_turn scenario; otherwise turns are free).
    left_turn_penalty: float = 0.0      # [s]
    right_turn_penalty: float = 0.0     # [s]
    u_turns_allowed: bool = True
    no_left_turns: Set[Tuple[int, int, int]] = field(default_factory=set)   # (from, via, to)

    def __post_init__(self):
        self._by_pair = {(e.start, e.end): e for e in self.edges}
        self.edge_by_id = {e.id: e for e in self.edges}
        self.max_speed_limit = max(e.speed_limit for e in self.edges)

    def freeze(self) -> "Town":
        """Make every container read-only, so a planner cannot change the town by accident."""
        self.nodes = MappingProxyType(dict(self.nodes))
        self.edges = tuple(self.edges)
        self.destinations = MappingProxyType(dict(self.destinations))
        self.river = tuple(self.river)
        self.bridges = tuple(self.bridges)
        self.no_left_turns = frozenset(self.no_left_turns)
        self._by_pair = MappingProxyType(dict(self._by_pair))
        self.edge_by_id = MappingProxyType(dict(self.edge_by_id))
        return self

    # ------------------------------------------------------------------ #
    def edge_between(self, u: int, v: int) -> Optional[Edge]:
        """The edge from u to v, or None if you cannot drive directly from u to v."""
        return self._by_pair.get((u, v))

    def heading(self, u: int, v: int) -> float:
        """ENU heading [rad] of the straight line from node u to node v."""
        a, b = self.nodes[u], self.nodes[v]
        return math.atan2(b.y - a.y, b.x - a.x)

    def distance(self, u: int, v: int) -> float:
        """Straight-line distance [m] between two nodes."""
        a, b = self.nodes[u], self.nodes[v]
        return math.hypot(b.x - a.x, b.y - a.y)

    def turn_type(self, u: int, v: int, w: int) -> str:
        """'straight', 'left', 'right' or 'u_turn' for driving u -> v -> w."""
        d = (self.heading(v, w) - self.heading(u, v) + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > math.radians(150):
            return "u_turn"
        if d > math.radians(30):
            return "left"
        if d < -math.radians(30):
            return "right"
        return "straight"

    def turn_allowed(self, u: int, v: int, w: int) -> bool:
        """False if the town forbids turning u -> v -> w (no-left-turn signs, no U-turns)."""
        t = self.turn_type(u, v, w)
        if t == "u_turn" and not self.u_turns_allowed:
            return False
        return not (t == "left" and (u, v, w) in self.no_left_turns)

    def turn_cost(self, u: int, v: int, w: int) -> float:
        """Extra seconds for the turn u -> v -> w (0 unless the scenario has turn penalties)."""
        t = self.turn_type(u, v, w)
        if t == "left":
            return self.left_turn_penalty
        if t == "right":
            return self.right_turn_penalty
        return 0.0

    @property
    def has_turn_rules(self) -> bool:
        return bool(self.left_turn_penalty or self.right_turn_penalty
                    or not self.u_turns_allowed or self.no_left_turns)


# ---------------------------------------------------------------------- #
# Helpers for the car's position (provided, so they are not a trap)
# ---------------------------------------------------------------------- #
def current_edge(town: Town, pose) -> Edge:
    """The edge the car is driving on: the nearest one whose direction matches the car's heading.

    ``pose`` is anything with ``x``, ``y`` and ``psi`` (a ``CarState``).
    """
    best, best_d = None, math.inf
    for e in town.edges:
        a, b = town.nodes[e.start], town.nodes[e.end]
        h = math.atan2(b.y - a.y, b.x - a.x)
        if abs((pose.psi - h + math.pi) % (2 * math.pi) - math.pi) > math.radians(60):
            continue
        dx, dy = b.x - a.x, b.y - a.y
        t = ((pose.x - a.x) * dx + (pose.y - a.y) * dy) / max(dx * dx + dy * dy, 1e-9)
        t = min(1.0, max(0.0, t))
        d = math.hypot(pose.x - (a.x + t * dx), pose.y - (a.y + t * dy))
        if d < best_d:
            best, best_d = e, d
    if best is None:
        raise ValueError("no edge matches the car's pose")
    return best


def start_node_from_pose(town: Town, pose) -> int:
    """The node your route must start from: the end of the edge the car is driving on.

    The car cannot turn around in the middle of a block, so the first place it
    can make a decision is the next node ahead of it.
    """
    return current_edge(town, pose).end


# ---------------------------------------------------------------------- #
# Town generator
# ---------------------------------------------------------------------- #
def generate_town(seed: int, nx: int, ny: int, block: float = 120.0, bridges: int = 2,
                  ring_road: bool = False, diagonal: bool = False, one_way_pairs: int = 1,
                  n_destinations: int = 8, removed_segments: int = 0, name: str = "town") -> Town:
    """A deterministic synthetic town.  Retries internally until it is strongly connected."""
    for attempt in range(50):
        town = _try_generate(seed * 7919 + attempt, nx, ny, block, bridges, ring_road, diagonal,
                             one_way_pairs, n_destinations, removed_segments, name)
        if _strongly_connected(town):
            return town
    raise RuntimeError("could not generate a connected town")


def _try_generate(seed, nx, ny, block, bridges, ring_road, diagonal, one_way_pairs,
                  n_destinations, removed_segments, name) -> Town:
    rng = np.random.default_rng(seed)
    # Uneven block lengths, then a smooth warp so streets curve gently instead
    # of looking like graph paper.  The warp is a sum of sine waves in random
    # directions, 4 to 14 blocks long.  Their slopes add up to at most 0.7, so
    # the warp never folds a street over another one.
    colx = np.concatenate(([0.0], np.cumsum(rng.uniform(0.75, 1.3, nx - 1)))) * block
    rowy = np.concatenate(([0.0], np.cumsum(rng.uniform(0.75, 1.3, ny - 1)))) * block
    slopes = rng.uniform(0.6, 1.4, 6)
    slopes *= 0.7 / slopes.sum()
    waves = []
    for sl in slopes:
        lam = rng.uniform(4.0, 14.0) * block
        th, ph = rng.uniform(0, 2 * math.pi, 2)
        dirn = rng.uniform(0, 2 * math.pi)
        waves.append((sl * lam / (2 * math.pi), lam, math.cos(th), math.sin(th), ph,
                      math.cos(dirn), math.sin(dirn)))

    def warp(x: float, y: float) -> Tuple[float, float]:
        dx = dy = 0.0
        for amp, lam, ux, uy, ph, vx, vy in waves:
            w = amp * math.sin(2 * math.pi * (ux * x + uy * y) / lam + ph)
            dx += w * vx
            dy += w * vy
        return x + dx, y + dy

    nid = {}
    nodes: Dict[int, Node] = {}
    for i in range(nx):
        for j in range(ny):
            k = len(nodes)
            nid[(i, j)] = k
            x, y = warp(colx[i] + rng.uniform(-0.10, 0.10) * block,
                        rowy[j] + rng.uniform(-0.10, 0.10) * block)
            nodes[k] = Node(k, float(x), float(y))

    river_row = ny // 2 - 1 if ny >= 4 else None
    bridge_cols = set()
    if river_row is not None:
        interior = list(range(nx))
        bridge_cols = set(int(c) for c in rng.choice(interior, size=min(bridges, nx), replace=False))

    # one-way couplets: pairs of adjacent north-south streets, one each way
    one_way: Dict[int, int] = {}          # column -> +1 (northbound only) or -1 (southbound only)
    cand = [i for i in range(1, nx - 2) if i not in bridge_cols and i + 1 not in bridge_cols]
    rng.shuffle(cand)
    for i in cand:
        if len(one_way) >= 2 * one_way_pairs:
            break
        if i in one_way or i + 1 in one_way or i - 1 in one_way or i + 2 in one_way:
            continue
        one_way[i], one_way[i + 1] = 1, -1

    segments = []   # (a, b, speed, street, directions) directions: 0 both, 1 a->b only, -1 b->a only
    for i in range(nx):
        for j in range(ny - 1):
            if river_row is not None and j == river_row and i not in bridge_cols:
                continue
            ring = ring_road and (i == 0 or i == nx - 1)
            speed = ARTERIAL if ring else (MAIN if i % 3 == 1 else RESIDENTIAL)
            street = f"{TREES[i % len(TREES)]} St"
            if river_row is not None and j == river_row:
                street = f"{TREES[i % len(TREES)]} St Bridge"
            segments.append((nid[(i, j)], nid[(i, j + 1)], speed, street, one_way.get(i, 0)))
    for j in range(ny):
        for i in range(nx - 1):
            ring = ring_road and (j == 0 or j == ny - 1)
            speed = ARTERIAL if ring else (MAIN if j % 3 == 1 else RESIDENTIAL)
            segments.append((nid[(i, j)], nid[(i + 1, j)], speed, f"{_ordinal(j + 1)} Ave", 0))
    if diagonal:
        for i in range(min(nx, ny) - 1):
            j0 = i
            if river_row is not None and j0 == river_row:
                continue   # the diagonal has no bridge of its own
            segments.append((nid[(i, j0)], nid[(i + 1, j0 + 1)], ARTERIAL, "Diagonal Blvd", 0))

    # a few pedestrian-only blocks make the grid irregular
    if removed_segments:
        removable = [k for k, sg in enumerate(segments)
                     if sg[4] == 0 and "Bridge" not in sg[3] and "Diagonal" not in sg[3]]
        drop = set(int(k) for k in rng.choice(removable, size=min(removed_segments, len(removable)),
                                              replace=False))
        segments = [sg for k, sg in enumerate(segments) if k not in drop]

    # mid-block destinations on two-way segments (not on bridges)
    two_way = [k for k, sg in enumerate(segments)
               if sg[4] == 0 and "Diagonal" not in sg[3] and "Bridge" not in sg[3]]
    picks = [int(k) for k in rng.choice(two_way, size=min(n_destinations, len(two_way)), replace=False)]
    dest_info = []
    new_segments = []
    split = set(picks)
    for k, (a, b, speed, street, dirn) in enumerate(segments):
        if k not in split:
            new_segments.append((a, b, speed, street, dirn))
            continue
        m = len(nodes)
        na, nb = nodes[a], nodes[b]
        nodes[m] = Node(m, 0.5 * (na.x + nb.x), 0.5 * (na.y + nb.y))
        new_segments.append((a, m, speed, street, 0))
        new_segments.append((m, b, speed, street, 0))
        forward = bool(rng.integers(0, 2))
        h = math.atan2(nb.y - na.y, nb.x - na.x)
        dest_info.append((m, h if forward else h + math.pi, street))
    segments = new_segments

    edges: List[Edge] = []
    for a, b, speed, street, dirn in segments:
        na, nb = nodes[a], nodes[b]
        L = math.hypot(nb.x - na.x, nb.y - na.y)
        if dirn >= 0:
            edges.append(Edge(len(edges), a, b, L, speed, street))
        if dirn <= 0:
            edges.append(Edge(len(edges), b, a, L, speed, street))

    order = rng.permutation(len(dest_info))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    destinations = {}
    for idx, k in enumerate(order[:len(letters)]):
        m, h, street = dest_info[int(k)]
        h = (h + math.pi) % (2 * math.pi) - math.pi
        destinations[letters[idx]] = Destination(letters[idx], m, h, street)

    river, bridges = [], []
    if river_row is not None:
        # The river runs between the two rows of streets it separates.
        mids = [(0.5 * (nodes[nid[(i, river_row)]].x + nodes[nid[(i, river_row + 1)]].x),
                 0.5 * (nodes[nid[(i, river_row)]].y + nodes[nid[(i, river_row + 1)]].y)) for i in range(nx)]
        ext = 0.45 * block
        river = [(mids[0][0] - ext, mids[0][1])] + [(float(x), float(y)) for x, y in mids] + \
                [(mids[-1][0] + ext, mids[-1][1])]
        bridges = [(nid[(i, river_row)], nid[(i, river_row + 1)]) for i in sorted(bridge_cols)]
    return Town(nodes=nodes, edges=edges, destinations=destinations, name=name, river=river,
                bridges=bridges)


def _strongly_connected(town: Town) -> bool:
    out: Dict[int, List[int]] = {n: [] for n in town.nodes}
    inc: Dict[int, List[int]] = {n: [] for n in town.nodes}
    for e in town.edges:
        out[e.start].append(e.end)
        inc[e.end].append(e.start)

    def reach(adj) -> int:
        start = next(iter(town.nodes))
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen)

    return reach(out) == len(town.nodes) and reach(inc) == len(town.nodes)


def closure_of(town: Town, edge: Edge) -> FrozenSet[int]:
    """Closing a street segment closes both directions of it."""
    ids = {edge.id}
    back = town.edge_between(edge.end, edge.start)
    if back is not None:
        ids.add(back.id)
    return frozenset(ids)
