"""Measure how many nodes a search explores, without trusting its own count.

We hand your dijkstra()/astar() a graph that remembers which nodes' neighbor
lists were read.  Reading graph[node] is what a search does when it expands a
node, so this counts the nodes your search actually worked on.
"""

from __future__ import annotations

from typing import List, Optional


class RecordingGraph(dict):
    """A dict that remembers the order in which nodes' neighbor lists were read."""

    def __init__(self, base):
        super().__init__(base)
        self.order: List = []
        self._seen = set()

    def _note(self, key):
        if key not in self._seen:
            self._seen.add(key)
            self.order.append(key)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)


def explored(search, graph, start, goal, *extra):
    """Run ``search(graph, start, goal, *extra)`` on a recording graph.

    Returns (order, count, result).  ``count`` is the number of nodes the
    search settled: the nodes it expanded, plus the goal when it was reached
    without being expanded (searches stop as soon as they pop the goal).
    """
    rec = RecordingGraph(graph)
    result = search(rec, start, goal, *extra)
    order = list(rec.order)
    path = result[0] if isinstance(result, tuple) and result else None
    reached = isinstance(path, (list, tuple)) and len(path) > 0 and path[-1] == goal
    if reached and goal not in rec._seen:
        order.append(goal)
    return order, len(order), result


def compare_searches(planner, town, cost, closed, start, goal) -> Optional[dict]:
    """Nodes explored by the candidate's own dijkstra() and astar() on one trip.

    Information only: it is never part of the score.  None if the parts it
    needs are not written yet or raise.
    """
    try:
        graph = planner.build_graph(town, cost, frozenset(closed))
        d_order, d_count, _ = explored(planner.dijkstra, graph, start, goal)
    except Exception:  # noqa: BLE001  (Parts 1/2 not done or broken: nothing to report)
        return None
    out = {"dijkstra": d_count, "dijkstra_order": d_order, "astar": None, "astar_order": None}
    try:
        h = planner.straight_line_heuristic(town, goal, cost)
        a_order, a_count, _ = explored(planner.astar, graph, start, goal, h)
        out["astar"], out["astar_order"] = a_count, a_order
    except Exception:  # noqa: BLE001  (Part 3 not done or broken)
        pass
    return out
