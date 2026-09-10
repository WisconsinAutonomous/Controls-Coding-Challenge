"""The grader's source of truth: optimal route costs.

It uses a vectorized Bellman-Ford (relax every edge at once, repeat until
nothing changes), which is a different algorithm from the ones you write.
"""

from __future__ import annotations

import math
from typing import FrozenSet, List, Optional

import numpy as np

from .town import Edge, Town

HEADING_TOL = math.radians(45)


def edge_cost(e: Edge, cost: str) -> float:
    return e.length if cost == "distance" else e.length / e.speed_limit


def heading_ok(town: Town, e: Edge, heading: Optional[float]) -> bool:
    if heading is None:
        return True
    h = town.heading(e.start, e.end)
    return abs((h - heading + math.pi) % (2 * math.pi) - math.pi) <= HEADING_TOL


class _Relaxer:
    """min-plus relaxation of dist[dst] <- dist[src] + w, vectorized."""

    def __init__(self, n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray):
        order = np.argsort(dst, kind="stable")
        self.n = n
        self.src, self.dst, self.w = src[order], dst[order], w[order]
        if len(self.dst):
            self.targets, self.starts = np.unique(self.dst, return_index=True)
        else:
            self.targets, self.starts = np.zeros(0, int), np.zeros(0, int)

    def run(self, init: np.ndarray) -> np.ndarray:
        d = init.copy()
        if not len(self.src):
            return d
        for _ in range(self.n + 1):
            cand = d[self.src] + self.w
            best = np.minimum.reduceat(cand, self.starts)
            new = d.copy()
            new[self.targets] = np.minimum(d[self.targets], best)
            if np.array_equal(new, d):
                break
            d = new
        return d


class Oracle:
    def __init__(self, town: Town, cost: str, closed: FrozenSet[int] = frozenset()):
        self.town, self.cost = town, cost
        self.edges: List[Edge] = [e for e in town.edges if e.id not in closed]
        self.n = max(town.nodes) + 1
        src = np.array([e.start for e in self.edges], dtype=int)
        dst = np.array([e.end for e in self.edges], dtype=int)
        w = np.array([edge_cost(e, cost) for e in self.edges], dtype=float)
        self._nodes = _Relaxer(self.n, src, dst, w)
        self._rev = _Relaxer(self.n, dst, src, w)
        self._states = None

    # -------------------------------------------------------------- #
    def from_node(self, start: int) -> np.ndarray:
        """Cheapest cost from ``start`` to every node (no turn rules)."""
        d = np.full(self.n, math.inf)
        d[start] = 0.0
        return self._nodes.run(d)

    def to_node(self, goal: int) -> np.ndarray:
        """Cheapest cost from every node to ``goal`` (no turn rules)."""
        d = np.full(self.n, math.inf)
        d[goal] = 0.0
        return self._rev.run(d)

    def dijkstra_expanded(self, start: int, goal: int) -> int:
        """How many nodes Dijkstra settles before (and including) the goal."""
        d = self.from_node(start)
        if not math.isfinite(d[goal]):
            return int(np.sum(np.isfinite(d)))
        return int(np.sum(d <= d[goal] + 1e-9))

    def path(self, start: int, goal: int) -> Optional[List[int]]:
        """One optimal node path (no turn rules), for scenario design."""
        d = self.from_node(start)
        if not math.isfinite(d[goal]):
            return None
        into = {}
        for e in self.edges:
            into.setdefault(e.end, []).append(e)
        path = [goal]
        while path[-1] != start:
            v = path[-1]
            for e in into.get(v, []):
                if abs(d[e.start] + edge_cost(e, self.cost) - d[v]) < 1e-7 and e.start not in path:
                    path.append(e.start)
                    break
            else:
                return None
        return path[::-1]

    # -------------------------------------------------------------- #
    def _build_states(self):
        town = self.town
        idx = {e.id: k for k, e in enumerate(self.edges)}
        leaving = {}
        for k, e in enumerate(self.edges):
            leaving.setdefault(e.start, []).append(k)
        src, dst, w = [], [], []
        for k1, e1 in enumerate(self.edges):
            for k2 in leaving.get(e1.end, []):
                e2 = self.edges[k2]
                if not town.turn_allowed(e1.start, e1.end, e2.end):
                    continue
                src.append(k1)
                dst.append(k2)
                w.append(edge_cost(e2, self.cost) + town.turn_cost(e1.start, e1.end, e2.end))
        self._state_idx = idx
        self._states = _Relaxer(len(self.edges), np.array(src, int), np.array(dst, int),
                                np.array(w, float))

    def best_cost(self, start_edge: Edge, goal: int, heading: Optional[float] = None) -> float:
        """Cheapest cost from the end of ``start_edge`` to ``goal``, with turn rules
        and (optionally) a required arrival heading."""
        if not self.town.has_turn_rules and heading is None:
            return float(self.from_node(start_edge.end)[goal])
        if self._states is None:
            self._build_states()
        d = np.full(len(self.edges), math.inf)
        if start_edge.id not in self._state_idx:
            return math.inf
        d[self._state_idx[start_edge.id]] = 0.0
        d = self._states.run(d)
        best = math.inf
        for k, e in enumerate(self.edges):
            if e.end == goal and heading_ok(self.town, e, heading):
                best = min(best, float(d[k]))
        return best
