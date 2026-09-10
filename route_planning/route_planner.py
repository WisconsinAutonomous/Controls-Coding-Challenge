"""Your route planner.  This is the file you edit.

Work through the parts in order.  After each one, check it:

    python check.py part1      (then part2, part3, part4)
    python run.py              (the full scenarios, once Parts 1 and 2 work)

Every TODO below raises NotImplementedError until you replace it.  Read the
docstrings: they explain what each function must do, step by step.

The town (see sim/town.py):
  * town.nodes: dict {node_id: Node(id, x, y)}, positions in meters (x East, y North)
  * town.edges: list of Edge(id, start, end, length, speed_limit, street).  An
    edge is one lane you can drive in ONE direction, from `start` to `end`.
    A two-way street is two edges.  A one-way street is one edge.
  * town.destinations: dict {"A": Destination(letter, node, heading, street), ...}
  * town.distance(u, v): straight-line distance between two nodes [m]
  * town.max_speed_limit: the fastest speed limit anywhere in town [m/s]
The town is read-only: build your own structures (like the graph in Part 1)
instead of changing it.
"""

from __future__ import annotations

import heapq  # noqa: F401  (you will want this in Part 2)
import math  # noqa: F401  (math.inf is handy)

from sim.town import current_edge, start_node_from_pose  # noqa: F401  (current_edge: stretch)


# ---------------------------------------------------------------------------
# Part 1: turn the town into a graph
# ---------------------------------------------------------------------------
def build_graph(town, cost="distance", closed=frozenset()):
    """Build an adjacency list: for every node, the nodes you can drive to next.

    Returns a dict  {node_id: [(neighbor_id, edge_cost), ...], ...}

    * EVERY node in town.nodes must be a key, even one you cannot leave
      (give it an empty list).  Your search will look nodes up by id.
    * Each Edge(start, end, ...) lets you drive from `start` to `end` only.
      Add (end, cost) to the list of `start`, and nothing to `end`'s list.
    * Skip any edge whose `id` is in `closed` (a set of edge ids).
    * cost == "distance": the edge cost is its length [m].
      cost == "time":     the edge cost is the time to drive it at the speed limit [s].

    Hint: start with {node_id: [] for node_id in town.nodes}, then loop over
    town.edges once.
    """
    raise NotImplementedError("Part 1: build_graph() is not implemented yet. Run: python check.py part1")


# ---------------------------------------------------------------------------
# Part 2: Dijkstra's algorithm
# ---------------------------------------------------------------------------
def dijkstra(graph, start, goal):
    """The cheapest path from `start` to `goal` in `graph` (from build_graph).

    Returns (path, cost, expanded):
      * path:     list of node ids [start, ..., goal], or None if the goal cannot be reached
      * cost:     total cost of that path (math.inf if unreachable)
      * expanded: how many nodes you "settled" (popped for the first time); we
                  use it to compare against A* in Part 3

    Dijkstra in plain words:
      1. Keep a FRONTIER of nodes you have reached but not finished, each with
         the cheapest cost found so far.  A priority queue (heapq) hands you
         the cheapest one quickly.  Start with just (0, start).
      2. Pop the cheapest node.  The first time a node is POPPED its cost is
         final ("settled").  Count it.  If it was already settled, skip it.
      3. If it is the goal, you are done: rebuild the path.
      4. Otherwise, for each (neighbor, edge_cost) of the node: if reaching the
         neighbor through this node is cheaper than anything you knew, remember
         the new cost and where you came from, and push it on the frontier.

    Hints:
      * heapq.heappush(heap, (cost, node)) and heapq.heappop(heap) -> (cost, node)
      * The same node can sit in the heap several times with different costs.
        That is fine: skip it when you pop it a second time.
      * Keep came_from[node] = the node you reached it from.  Rebuild the path
        by walking came_from backward from the goal, then reverse it.
      * start == goal is a valid question: the answer is ([start], 0.0, 1).
    """
    raise NotImplementedError("Part 2: dijkstra() is not implemented yet. Run: python check.py part2")


# ---------------------------------------------------------------------------
# Part 3: A*
# ---------------------------------------------------------------------------
def straight_line_heuristic(town, goal, cost="distance"):
    """Return a function h(node) that estimates the cost from `node` to `goal`.

    A* only stays correct if h NEVER overestimates the true remaining cost
    (it is "admissible").
      * cost == "distance": the straight-line distance to the goal is a
        perfect choice.  No road can be shorter than a straight line.
      * cost == "time": what is the fastest you could possibly get there?
        Hint: town.max_speed_limit.

    Example use:  h = straight_line_heuristic(town, goal, "distance")
                  h(node) -> how far node is from the goal, in the same units as the edge costs
    """
    raise NotImplementedError("Part 3: straight_line_heuristic() is not implemented yet. "
                              "Run: python check.py part3")


def astar(graph, start, goal, heuristic):
    """Like dijkstra(), but the frontier is ordered by  cost_so_far + heuristic(node).

    Returns (path, cost, expanded) exactly like dijkstra().

    That one change makes the search reach toward the goal instead of growing
    in every direction.  Everything else (settle when popped, came_from,
    rebuilding the path) stays the same.  Careful: the cost you return is the
    real path cost, not the priority you sorted by.

    Hint: if heuristic is lambda n: 0.0, astar must give the same answer as dijkstra.
    """
    raise NotImplementedError("Part 3: astar() is not implemented yet. Run: python check.py part3")


# ---------------------------------------------------------------------------
# Part 4: plan a real trip
# ---------------------------------------------------------------------------
def plan_route(town, start_pose, destination, cost, closed):
    """What run.py calls for every trip.  Returns the route: a list of node ids
    from the start node to the destination node (or None if you find none).

    * start_pose:  a CarState (x, y, psi) of the car, which is in the middle of a block
    * destination: a letter, e.g. "C"; the node is town.destinations["C"].node
    * cost:        "distance" or "time" (the scenario tells you which matters)
    * closed:      set of edge ids you must not use (road closures)

    This version already works once Parts 1 and 2 are done, but it has three
    problems.  Part 4 is to fix them:
      1. It always plans by distance, even when the scenario asks for time.
      2. It ignores road closures.
      3. It uses Dijkstra even though you wrote a faster A* in Part 3.
    """
    start = start_node_from_pose(town, start_pose)      # provided: the next node ahead of the car
    goal = town.destinations[destination].node
    graph = build_graph(town, cost="distance")           # TODO Part 4: use `cost` and `closed`
    path, total, expanded = dijkstra(graph, start, goal)  # TODO Part 4: use astar + your heuristic
    return path
