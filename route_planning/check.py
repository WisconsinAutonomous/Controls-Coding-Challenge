#!/usr/bin/env python3
"""Checkpoints: small tests for each part, with hints when something is off.

    python check.py            # every part
    python check.py part2      # just one part
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import numbers
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `common`
sys.path.insert(0, HERE)

from sim.explore import explored  # noqa: E402
from sim.oracle import Oracle  # noqa: E402
from sim.town import Destination, Edge, Node, Town  # noqa: E402


class Part:
    def __init__(self, title: str):
        self.title, self.passed, self.failed = title, 0, 0
        print(f"\n{title}")

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  [ok] {msg}")

    def bad(self, msg: str, hint: str = "") -> None:
        self.failed += 1
        print(f"  [X]  {msg}")
        if hint:
            print(f"       hint: {hint}")

    def crashed(self, what: str) -> None:
        self.failed += 1
        tb = traceback.format_exc().strip().splitlines()
        print(f"  [X]  {what} raised an error:")
        for line in tb[-3:]:
            print(f"         {line}")

    def done(self) -> bool:
        total = self.passed + self.failed
        print(f"  -> {self.passed}/{total} checks passed")
        return self.failed == 0


def not_done(fn, *args) -> bool:
    try:
        fn(*args)
    except NotImplementedError:
        return True
    except Exception:  # noqa: BLE001  (broken is not "not done")
        return False
    return False


# ---------------------------------------------------------------------- #
# Small test worlds (answers below are worked out by hand)
# ---------------------------------------------------------------------- #
def tiny_town() -> Town:
    """  3 ---- 4          Edges (one per direction you can drive):
         |      ^            0<->1 123.4 m at 11.2 m/s, 1<->2 87.5 m at 15.6 m/s,
         v      |            3<->4 123.4 m at 11.2 m/s, 1 -> 4 one-way 96.3 m at 11.2 m/s,
         0 ---- 1 ---- 2 -> 5    3 -> 0 one-way 96.3 m at 13.4 m/s, 2 -> 5 one-way 89.4 m at 20.1 m/s
    """
    nodes = {0: Node(0, 0.0, 0.0), 1: Node(1, 123.4, 0.0), 2: Node(2, 210.9, 0.0),
             3: Node(3, 0.0, 96.3), 4: Node(4, 123.4, 96.3), 5: Node(5, 300.3, 0.0)}
    E = Edge
    edges = [E(0, 0, 1, 123.4, 11.2, "1st Ave"), E(1, 1, 0, 123.4, 11.2, "1st Ave"),
             E(2, 1, 2, 87.5, 15.6, "1st Ave"), E(3, 2, 1, 87.5, 15.6, "1st Ave"),
             E(4, 1, 4, 96.3, 11.2, "Oak St"), E(5, 3, 0, 96.3, 13.4, "Elm St"),
             E(6, 3, 4, 123.4, 11.2, "2nd Ave"), E(7, 4, 3, 123.4, 11.2, "2nd Ave"),
             E(8, 2, 5, 89.4, 20.1, "Dead End Rd")]
    return Town(nodes, edges, {"A": Destination("A", 5, None, "Dead End Rd")}, name="Tinytown")


# Worked out by hand from the edge list above (time = length / speed limit).
TINY_DISTANCE = {0: [(1, 123.4)], 1: [(0, 123.4), (2, 87.5), (4, 96.3)], 2: [(1, 87.5), (5, 89.4)],
                 3: [(0, 96.3), (4, 123.4)], 4: [(3, 123.4)], 5: []}
TINY_TIME = {0: [(1, 123.4 / 11.2)], 1: [(0, 123.4 / 11.2), (2, 87.5 / 15.6), (4, 96.3 / 11.2)],
             2: [(1, 87.5 / 15.6), (5, 89.4 / 20.1)], 3: [(0, 96.3 / 13.4), (4, 123.4 / 11.2)],
             4: [(3, 123.4 / 11.2)], 5: []}


def grid_graph(seed: int, n: int = 12):
    """A random n x n road grid with positions, some one-way streets, and costs
    that are never shorter than the straight-line distance."""
    rng = random.Random(seed)
    pos = {i * n + j: (100.0 * i + rng.uniform(-20, 20), 100.0 * j + rng.uniform(-20, 20))
           for i in range(n) for j in range(n)}
    graph = {k: [] for k in pos}
    for i in range(n):
        for j in range(n):
            a = i * n + j
            for b in ([(i + 1) * n + j] if i + 1 < n else []) + ([i * n + j + 1] if j + 1 < n else []):
                d = math.dist(pos[a], pos[b]) * rng.uniform(1.0, 1.8)
                r = rng.random()
                if r > 0.15:
                    graph[a].append((b, d))
                if r < 0.85:
                    graph[b].append((a, d))
    return graph, pos


def true_costs(graph, start):
    """Bellman-Ford: slow and simple, but certainly right."""
    d = {k: math.inf for k in graph}
    d[start] = 0.0
    for _ in range(len(graph)):
        changed = False
        for u, nbrs in graph.items():
            if d[u] == math.inf:
                continue
            for v, c in nbrs:
                if d[u] + c < d[v] - 1e-12:
                    d[v] = d[u] + c
                    changed = True
        if not changed:
            break
    return d


def is_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def search_result(p: Part, out, what: str):
    """Check the shape of a (path, cost, expanded) result.  Returns it cleaned up, or None."""
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        p.bad(f"{what} should return a tuple (path, cost, expanded), got {out!r}"[:200])
        return None
    path, cost, expanded = out
    if path is not None and (isinstance(path, (str, set, frozenset, dict)) or not hasattr(path, "__iter__")):
        p.bad(f"{what}: the path should be a list of nodes (or None), got a {type(path).__name__}")
        return None
    if not is_number(cost):
        p.bad(f"{what}: the cost should be a number (math.inf when there is no path), got {cost!r}")
        return None
    if not (is_number(expanded) and float(expanded).is_integer()):
        p.bad(f"{what}: `expanded` should be a whole number of nodes, got {expanded!r}")
        return None
    return (list(path) if path is not None else None), float(cost), int(expanded)


def path_problems(graph, path, start, goal, cost):
    """A message describing what is wrong with (path, cost), or '' if it is fine."""
    if not path:
        return f"the path should be a list of nodes, got {path!r}"
    if path[0] != start:
        return f"the path should start at {start}, but starts at {path[0]}"
    if path[-1] != goal:
        return f"the path should end at {goal}, but ends at {path[-1]}"
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        step = [c for (w, c) in graph.get(u, []) if w == v]
        if not step:
            back = any(w == u for (w, _) in graph.get(v, []))
            extra = f" ({v} -> {u} exists, but it is one-way)" if back else ""
            return f"your path goes {u} -> {v}, but there is no edge from {u} to {v}{extra}"
        total += min(step)
    if abs(total - cost) > 1e-6 * max(1.0, abs(total)):
        return f"you returned cost {cost:.3f}, but the edges along your path add up to {total:.3f}"
    return ""


def adjacency(graph, n):
    """Sorted [(neighbor, cost)] for node n, or None if it is not in that shape."""
    try:
        return sorted((int(v), float(c)) for v, c in graph.get(n, []))
    except (TypeError, ValueError, AttributeError):
        return None


def same_lists(a, b) -> bool:
    return a is not None and len(a) == len(b) and all(
        x[0] == y[0] and abs(x[1] - y[1]) < 1e-6 for x, y in zip(a, sorted(b)))


def show(lst):
    return "[" + ", ".join(f"({v}, {c:.3f})" for v, c in sorted(lst)) + "]"


# ---------------------------------------------------------------------- #
def check_part1(rp) -> bool:
    p = Part("Part 1: build_graph()")
    if not_done(rp.build_graph, tiny_town()):
        print("  not implemented yet: this is where to start!")
        return False
    town = tiny_town()
    try:
        g = rp.build_graph(town)
    except Exception:  # noqa: BLE001
        p.crashed("build_graph(tiny_town)")
        return p.done()
    if not isinstance(g, dict):
        p.bad(f"build_graph should return a dict, got {type(g).__name__}")
        return p.done()
    missing = [n for n in town.nodes if n not in g]
    if missing:
        p.bad(f"these nodes are not keys in your graph: {missing}",
              "every node needs a key, even one you cannot leave (node 5 is a dead end): give it []")
    else:
        p.ok("every node is a key, including the dead end")
    if not all(adjacency(g, n) is not None for n in town.nodes):
        p.bad("each value should be a list of (neighbor, cost) pairs, e.g. [(1, 123.4), (4, 96.3)]")
        return p.done()
    if any(v == 1 for v, _ in adjacency(g, 4)):
        p.bad("node 4 lists node 1, but Oak St is one-way: you can only drive 1 -> 4",
              "an Edge only lets you drive from edge.start to edge.end; add it to edge.start's list only")
    elif any(v == 3 for v, _ in adjacency(g, 0)):
        p.bad("node 0 lists node 3, but Elm St is one-way: you can only drive 3 -> 0")
    else:
        p.ok("one-way streets go only one way")
    wrong = [n for n in town.nodes if not same_lists(adjacency(g, n), TINY_DISTANCE[n])]
    if wrong:
        n = wrong[0]
        p.bad(f"with cost='distance', node {n} should have {show(TINY_DISTANCE[n])}, got {show(adjacency(g, n))}",
              "the distance cost of an edge is edge.length")
    else:
        p.ok("distance costs are right")
    try:
        gt = rp.build_graph(town, cost="time")
        wrong = [n for n in town.nodes if not same_lists(adjacency(gt, n), TINY_TIME[n])]
        if wrong:
            n = wrong[0]
            p.bad(f"with cost='time', node {n} should have {show(TINY_TIME[n])}, got {show(adjacency(gt, n) or [])}",
                  "time = edge.length / edge.speed_limit, as a decimal (not // and not rounded)")
        else:
            p.ok("time costs are right (e.g. 87.5 m at 15.6 m/s = 5.609 s)")
    except Exception:  # noqa: BLE001
        p.crashed("build_graph(tiny_town, cost='time')")
    for cost, table in (("distance", TINY_DISTANCE), ("time", TINY_TIME)):
        try:
            gc = rp.build_graph(town, cost=cost, closed=frozenset({2, 3}))
            if any(v == 2 for v, _ in adjacency(gc, 1) or []) or any(v == 1 for v, _ in adjacency(gc, 2) or []):
                p.bad(f"cost='{cost}': edges 2 and 3 (1st Ave between 1 and 2) are closed, "
                      "but your graph still uses them", "skip every edge whose edge.id is in `closed`")
            elif not same_lists(adjacency(gc, 2), [(5, table[2][1][1])]):
                p.bad(f"cost='{cost}': closing edges 2 and 3 should leave node 2 with "
                      f"{show([(5, table[2][1][1])])}, got {show(adjacency(gc, 2) or [])}")
            else:
                p.ok(f"closed edges are skipped (cost='{cost}')")
        except Exception:  # noqa: BLE001
            p.crashed(f"build_graph(tiny_town, cost='{cost}', closed={{2, 3}})")
    return p.done()


def check_part2(rp) -> bool:
    p = Part("Part 2: dijkstra()")
    if not_done(rp.dijkstra, {0: []}, 0, 0):
        print("  not implemented yet")
        return False

    def run(graph, s, t, what):
        try:
            order, measured, out = explored(rp.dijkstra, graph, s, t)
        except Exception:  # noqa: BLE001
            p.crashed(what)
            return None, None
        return search_result(p, out, what), measured

    # The classic trap: 0 -> 2 looks cheap at first (4), but 0 -> 1 -> 2 is cheaper (2).
    trap = {0: [(1, 1.0), (2, 4.0)], 1: [(2, 1.0), (3, 5.0)], 2: [(3, 1.0)], 3: []}
    out, _ = run(trap, 0, 3, "dijkstra(trap, 0, 3)")
    if out:
        path, cost, _ = out
        if path == [0, 1, 2, 3] and abs(cost - 3.0) < 1e-9:
            p.ok("finds the cheapest path, not the first one it sees")
        elif abs(cost - 5.0) < 1e-9 or abs(cost - 6.0) < 1e-9:
            p.bad(f"returned cost {cost} via {path}, but 0 -> 1 -> 2 -> 3 costs 3",
                  "a node's cost is only final when you POP it. Two usual causes: marking a node done "
                  "when you first SEE (push) it, or stopping when the goal is pushed instead of popped.")
        else:
            p.bad(f"expected path [0, 1, 2, 3] with cost 3, got {path} with cost {cost}")
    ring = {0: [(1, 1.0)], 1: [(2, 1.0)], 2: [(0, 1.0)]}
    out, _ = run(ring, 2, 1, "dijkstra(one_way_ring, 2, 1)")
    if out:
        msg = path_problems(ring, out[0], 2, 1, out[1]) if out[0] else "you returned no path"
        if msg or abs(out[1] - 2.0) > 1e-9:
            p.bad(msg or f"expected cost 2 (2 -> 0 -> 1), got {out[1]}")
        else:
            p.ok("respects one-way edges (2 -> 0 -> 1, not 2 -> 1)")
    island = {0: [(1, 1.0)], 1: [], 2: [(0, 1.0)]}
    out, _ = run(island, 0, 2, "dijkstra(island, 0, 2)")
    if out:
        if out[0] is None and out[1] == math.inf:
            p.ok("returns (None, math.inf, ...) when the goal cannot be reached")
        else:
            p.bad(f"node 2 cannot be reached from 0; expected (None, inf, ...), got {out[:2]}",
                  "when the frontier runs empty without popping the goal, there is no path")
    out, _ = run(trap, 1, 1, "dijkstra(trap, 1, 1)")
    if out:
        if out[0] == [1] and out[1] == 0:
            p.ok("start == goal gives ([start], 0)")
        else:
            p.bad(f"start == goal should give ([1], 0.0, ...), got {out[:2]}")
    worst, count_msg = "", ""
    for seed in range(8):
        graph, _ = grid_graph(seed)
        rng = random.Random(100 + seed)
        for _ in range(6):
            s, t = rng.choice(list(graph)), rng.choice(list(graph))
            best = true_costs(graph, s)[t]
            out, measured = run(graph, s, t, f"dijkstra(grid{seed}, {s}, {t})")
            if out is None:
                return p.done()
            path, cost, expanded = out
            if best == math.inf:
                if path is not None:
                    worst = worst or f"from {s} to {t} there is no route, but you returned {path}"
                continue
            msg = path_problems(graph, path, s, t, cost) if path is not None else "you returned no path"
            if not msg and cost > best + 1e-6:
                msg = f"from {s} to {t} you found cost {cost:.1f}, but {best:.1f} is possible"
            worst = worst or msg
            if measured > 1 and abs(expanded - measured) > 1 and not count_msg:
                count_msg = (f"from {s} to {t} you report expanded = {expanded}, but your search settled "
                             f"{measured} nodes (we watched which neighbor lists it read)")
    if worst:
        p.bad(f"on random grids: {worst}")
    else:
        p.ok("matches the true cheapest cost on 48 random trips, with valid paths")
    if count_msg:
        p.bad(count_msg, "count a node once, when you settle it (not every time it comes off the heap)")
    else:
        p.ok("`expanded` counts the nodes you settle")
    return p.done()


def check_part3(rp) -> bool:
    p = Part("Part 3: straight_line_heuristic() and astar()")
    tiny = tiny_town()
    h_missing = not_done(rp.straight_line_heuristic, tiny, 5)
    a_missing = not_done(rp.astar, {0: []}, 0, 0, lambda n: 0.0)
    if h_missing and a_missing:
        print("  not implemented yet")
        return False
    from sim.scenarios import rush_hour
    town = rush_hour(0).town
    goal = town.destinations["A"].node
    if h_missing:
        p.bad("straight_line_heuristic() is not implemented yet")
    else:
        for cost in ("distance", "time"):
            try:
                h = rp.straight_line_heuristic(copy.copy(town), goal, cost)
                vals = {n: h(n) for n in town.nodes}
                if not all(is_number(v) and math.isfinite(v) for v in vals.values()):
                    bad_n = next(n for n, v in vals.items() if not (is_number(v) and math.isfinite(v)))
                    p.bad(f"cost='{cost}': h({bad_n}) returned {vals[bad_n]!r}; h must return a finite number")
                    continue
                truth = Oracle(town, cost).to_node(goal)
                over = [(n, vals[n], truth[n]) for n in town.nodes if vals[n] > truth[n] + 1e-6]
                if over:
                    n, hv, tv = max(over, key=lambda x: x[1] - x[2])
                    hint = ("divide the distance by the FASTEST speed anywhere (town.max_speed_limit)"
                            if cost == "time" else "use the straight-line distance to the goal")
                    p.bad(f"cost='{cost}': h({n}) = {hv:.1f}, but the real best route from {n} costs only "
                          f"{tv:.1f}. A* can then return a wrong path.", hint)
                elif all(vals[n] <= 1e-9 for n in town.nodes if n != goal):
                    p.bad(f"cost='{cost}': your heuristic is 0 everywhere, so A* is just Dijkstra",
                          "estimate the remaining cost from the straight-line distance")
                else:
                    p.ok(f"cost='{cost}': never overestimates (admissible) and is not zero")
            except Exception:  # noqa: BLE001
                p.crashed(f"straight_line_heuristic(town, goal, '{cost}')")
    if a_missing:
        p.bad("astar() is not implemented yet")
        return p.done()
    worst, n_a, n_d = "", 0, 0
    for seed in range(6):
        graph, pos = grid_graph(50 + seed)
        rng = random.Random(200 + seed)
        for _ in range(6):
            s, t = rng.choice(list(graph)), rng.choice(list(graph))
            best = true_costs(graph, s)[t]
            h = (lambda tt: (lambda n: math.dist(pos[n], pos[tt])))(t)
            try:
                _, a_count, out = explored(rp.astar, graph, s, t, h)
            except Exception:  # noqa: BLE001
                p.crashed(f"astar(grid, {s}, {t}, h)")
                return p.done()
            res = search_result(p, out, f"astar(grid, {s}, {t}, h)")
            if res is None:
                return p.done()
            path, cost, _ = res
            if best == math.inf:
                continue
            msg = path_problems(graph, path, s, t, cost) if path is not None else "you returned no path"
            if msg and "add up to" in msg and path is not None:
                msg += " (did you return the priority cost + heuristic instead of the real cost?)"
            if not msg and cost > best + 1e-6:
                msg = f"from {s} to {t} A* found cost {cost:.1f}, but {best:.1f} is possible"
            worst = worst or msg
            try:
                _, d_count, _ = explored(rp.dijkstra, graph, s, t)
                n_a, n_d = n_a + a_count, n_d + d_count
            except Exception:  # noqa: BLE001
                pass
    if worst:
        p.bad(f"on random grids: {worst}")
    else:
        p.ok("A* finds the true cheapest cost on 36 random trips")
    if n_d and n_a < 0.8 * n_d:
        p.ok(f"we watched both searches: A* settled {n_a} nodes in total, Dijkstra {n_d} "
             f"({100 * (1 - n_a / n_d):.0f}% less work)")
    elif n_d:
        p.bad(f"we watched both searches: A* settled {n_a} nodes, Dijkstra {n_d}. A* should need clearly fewer",
              "order the frontier by cost_so_far + heuristic(node), not by cost_so_far alone")
    try:
        graph, _ = grid_graph(99)
        r0 = search_result(p, rp.astar(graph, 0, 143, lambda n: 0.0), "astar(graph, 0, 143, lambda n: 0.0)")
        r1 = search_result(p, rp.dijkstra(graph, 0, 143), "dijkstra(graph, 0, 143)")
        if r0 and r1:
            if abs(r0[1] - r1[1]) < 1e-6:
                p.ok("with a zero heuristic, A* gives the same answer as Dijkstra")
            else:
                p.bad(f"with heuristic 0, A* returned {r0[1]:.1f} but Dijkstra {r1[1]:.1f}; they should agree")
    except Exception:  # noqa: BLE001
        p.crashed("astar(graph, 0, 143, lambda n: 0.0)")
    return p.done()


class Spy:
    """Watches which of your functions plan_route calls, and with what."""

    def __init__(self, rp):
        self.rp, self.inside, self.saved = rp, False, {}
        self.calls = {"build_graph": [], "dijkstra": [], "astar": []}

    def __enter__(self):
        for name in ("plan_route", "build_graph", "dijkstra", "astar"):
            self.saved[name] = getattr(self.rp, name)
        orig_plan = self.saved["plan_route"]

        def plan(*a, **k):
            self.inside = True
            try:
                return orig_plan(*a, **k)
            finally:
                self.inside = False
        self.rp.plan_route = plan
        for name in ("build_graph", "dijkstra", "astar"):
            setattr(self.rp, name, self._watch(name, self.saved[name]))
        return self

    def _watch(self, name, fn):
        def wrapper(*a, **k):
            if self.inside:
                self.calls[name].append((a, k))
            return fn(*a, **k)
        return wrapper

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(self.rp, name, fn)

    def graph_costs(self):
        out = set()
        for a, k in self.calls["build_graph"]:
            out.add(k.get("cost", a[1] if len(a) > 1 else "distance"))
        return out

    def passed_closures(self) -> bool:
        return any(("closed" in k) or len(a) > 2 for a, k in self.calls["build_graph"])

    def heuristics(self):
        for a, k in self.calls["astar"][:3]:
            goal = k.get("goal", a[2] if len(a) > 2 else None)
            h = k.get("heuristic", a[3] if len(a) > 3 else None)
            if goal is not None and callable(h):
                yield goal, h


def check_part4(rp) -> bool:
    p = Part("Part 4: plan_route() on the core scenarios")
    if not_done(rp.build_graph, tiny_town()) or not_done(rp.dijkstra, {0: []}, 0, 0):
        print("  not ready yet: plan_route() needs build_graph() and dijkstra() (Parts 1 and 2)")
        return False
    from sim.runner import BadReturn, _as_route, run_scenario
    from sim.scenarios import SCENARIOS
    scn0 = SCENARIOS["first_route"](0)
    trip = scn0.trips[0]
    try:
        out = rp.plan_route(copy.copy(scn0.town), trip.pose, trip.destination, scn0.cost, frozenset())
        _as_route(out)
        p.ok("plan_route returns a route (a list of node ids)")
    except BadReturn as exc:
        p.bad(str(exc), "plan_route should end with `return path`")
        return p.done()
    except Exception:  # noqa: BLE001
        p.crashed("plan_route(...) on first_route")
        return p.done()
    with Spy(rp) as spy:
        results = {name: (SCENARIOS[name](0), None) for name in ("first_route", "rush_hour", "road_closed")}
        for name, (scn, _) in results.items():
            results[name] = (scn, run_scenario(scn, rp, measure=False))
    for name, (scn, res) in results.items():
        bad = [t for t in res.trips if not t.ok]
        if bad:
            msg, hint = bad[0].message, ""
            if "closed" in msg:
                hint = ("plan_route never passes `closed` to build_graph()" if not spy.passed_closures()
                        else "a closed road is still in your graph: check how build_graph() skips closed edges")
            elif "not implemented" in msg:
                hint = "finish the earlier parts first"
            p.bad(f"{name}: {len(bad)} of {len(res.trips)} trips failed. First problem: {msg}", hint)
            continue
        ratio = sum(t.ratio for t in res.trips) / len(res.trips)
        if ratio > 1.001:
            hint = "some routes are not the cheapest: run python check.py part2 and part3"
            if scn.cost == "time" and "time" not in spy.graph_costs():
                hint = "this scenario scores travel TIME, but plan_route builds the graph by distance: pass cost=cost"
            else:
                for goal, h in spy.heuristics():
                    try:
                        truth = Oracle(scn.town, scn.cost).to_node(goal)
                        if any(h(n) > truth[n] + 1e-6 for n in scn.town.nodes):
                            hint = (f"the heuristic plan_route gives astar() overestimates for cost='{scn.cost}': "
                                    "use straight_line_heuristic(town, goal, cost)")
                            break
                    except Exception:  # noqa: BLE001
                        pass
            p.bad(f"{name}: every route is valid, but on average {100 * (ratio - 1):.1f}% slower than the best", hint)
            continue
        p.ok(f"{name}: all {len(res.trips)} trips valid and as cheap as possible, score {res.score:.1f}")
    if spy.calls["astar"]:
        p.ok("plan_route uses your astar()")
    elif spy.calls["dijkstra"]:
        p.bad("plan_route still calls dijkstra()", "switch to astar() with straight_line_heuristic (step 3)")
    return p.done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", nargs="?", choices=["part1", "part2", "part3", "part4"])
    ap.add_argument("--planner", default=os.path.join(HERE, "route_planner.py"))
    args = ap.parse_args()
    # Your own helper modules next to this file can be imported (harness modules still win).
    _own_dir = os.path.dirname(os.path.abspath(args.planner))
    if _own_dir not in sys.path:
        sys.path.append(_own_dir)
    spec = importlib.util.spec_from_file_location("candidate_route_planner", args.planner)
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    parts = {"part1": check_part1, "part2": check_part2, "part3": check_part3, "part4": check_part4}
    todo = [args.part] if args.part else list(parts)
    results = {name: parts[name](rp) for name in todo}
    print()
    if all(results.values()):
        print("All checks passed." + ("" if args.part else " Now try `python run.py`, then the stretch scenarios."))
        return 0
    first = next(k for k, v in results.items() if not v)
    print(f"Next: work on {first.replace('part', 'Part ')} (python check.py {first}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
