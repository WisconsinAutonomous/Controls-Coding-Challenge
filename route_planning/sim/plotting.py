"""Pictures: the town, your route, and which nodes your search looked at."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .explore import compare_searches  # noqa: E402
from .town import ARTERIAL, MAIN, Town, start_node_from_pose  # noqa: E402


def draw_town(ax, town: Town, closed=frozenset(), light=False):
    big = len(town.nodes) > 500
    segs, widths, colors = [], [], []
    oneway = []
    done = set()
    for e in town.edges:
        pair = (min(e.start, e.end), max(e.start, e.end))
        a, b = town.nodes[e.start], town.nodes[e.end]
        two_way = town.edge_between(e.end, e.start) is not None
        if two_way and pair in done:
            continue
        done.add(pair)
        segs.append([(a.x, a.y), (b.x, b.y)])
        w = 2.4 if e.speed_limit >= ARTERIAL - 0.1 else (1.6 if e.speed_limit >= MAIN - 0.1 else 1.0)
        widths.append(w * (0.5 if big else 1.0))
        colors.append("0.80" if light else "0.60")
        if not two_way:
            oneway.append(e)
    if town.river:
        xs, ys = zip(*town.river)
        ax.plot(xs, ys, color="#9cc9ee", lw=10 if not big else 5, solid_capstyle="round", zorder=0,
                label="river (only bridges cross it)")
        for k, (na, nb) in enumerate(town.bridges):
            a, b = town.nodes[na], town.nodes[nb]
            ax.plot([a.x, b.x], [a.y, b.y], color="#7a5230", lw=8 if not big else 6,
                    solid_capstyle="butt", zorder=2, label="bridge" if k == 0 else None)
    ax.add_collection(LineCollection(segs, linewidths=widths, colors=colors, zorder=1))
    if not big:
        for e in oneway:
            a, b = town.nodes[e.start], town.nodes[e.end]
            mx, my = 0.5 * (a.x + b.x), 0.5 * (a.y + b.y)
            ax.annotate("", xy=(mx + 0.08 * (b.x - a.x), my + 0.08 * (b.y - a.y)), xytext=(mx, my),
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0), zorder=2)
    for eid in closed:
        e = town.edge_by_id[eid]
        a, b = town.nodes[e.start], town.nodes[e.end]
        ax.plot([a.x, b.x], [a.y, b.y], color="red", lw=5, alpha=0.5, zorder=3)
        ax.plot(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), "x", color="red", ms=10, mew=3, zorder=7)
    if not big:
        for d in town.destinations.values():
            n = town.nodes[d.node]
            ax.text(n.x, n.y, d.letter, fontsize=7, ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", lw=0.5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_route(ax, town: Town, route, color="tab:orange", lw=3.0, ls="-", label=None, alpha=1.0):
    if not route or len(route) < 2:
        return
    xs = [town.nodes[n].x for n in route]
    ys = [town.nodes[n].y for n in route]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=label, alpha=alpha, zorder=6,
            solid_capstyle="round")


def draw_expanded(ax, town: Town, order, cmap="viridis"):
    if not order:
        return
    pts = [(town.nodes[n].x, town.nodes[n].y, k) for k, n in enumerate(order) if n in town.nodes]
    xs, ys, ks = zip(*pts)
    ax.scatter(xs, ys, c=ks, cmap=cmap, s=14 if len(town.nodes) < 500 else 3, zorder=4, alpha=0.85)


def draw_rules(ax, town: Town, dest=None, check_heading=False):
    """No-left-turn intersections and the required arrival direction (stretch scenarios)."""
    if town.no_left_turns and len(town.nodes) <= 500:
        vias = sorted({v for (_, v, _) in town.no_left_turns})
        ax.plot([town.nodes[v].x for v in vias], [town.nodes[v].y for v in vias], "o", ms=9, mfc="none",
                mec="crimson", mew=1.6, zorder=5, ls="none", label="no left turn (from some directions)")
    if check_heading and dest is not None and dest.heading is not None:
        n = town.nodes[dest.node]
        L = 55.0
        ax.annotate("", xy=(n.x + L * math.cos(dest.heading), n.y + L * math.sin(dest.heading)),
                    xytext=(n.x - L * math.cos(dest.heading), n.y - L * math.sin(dest.heading)),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, mutation_scale=18), zorder=9)
        ax.plot([], [], color="black", lw=2.2, label="arrive driving this way")


def plot_scenario(result, planner, path: str, trip_index: int = 0) -> None:
    """Plot trip number ``trip_index`` (0-based) of a scenario."""
    scn = result.scenario
    town = scn.town
    k = max(0, min(trip_index, len(scn.trips) - 1))
    trip = scn.trips[k]
    tr = result.trips[k] if k < len(result.trips) else None
    dest = town.destinations[trip.destination]
    goal = dest.node
    start = start_node_from_pose(town, trip.pose)
    closed = tr.closures[-1] if tr and tr.closures else frozenset(trip.closed)

    node_only = town.has_turn_rules or scn.check_heading
    note = " (node search: ignores the turn rules)" if town.has_turn_rules else (
        " (node search: ignores the arrival direction)" if scn.check_heading else "")
    panels = []   # (title, explored order, cmap)
    info = compare_searches(planner, town, scn.cost, trip.closed, start, goal)
    if info is not None:
        panels.append((f"your dijkstra() explored {info['dijkstra']} nodes{note}",
                       info["dijkstra_order"], "Blues_r"))
        if info["astar"] is not None:
            panels.append((f"your astar() explored {info['astar']} nodes{note}", info["astar_order"], "Greens_r"))
    if not panels:
        panels.append(("your route (search not shown until Parts 1 and 2 work)", [], None))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.6), squeeze=False)
    for ax, (title, order, cmap) in zip(axes[0], panels):
        draw_town(ax, town, closed)
        draw_rules(ax, town, dest, scn.check_heading)
        if cmap:
            draw_expanded(ax, town, order, cmap)
        if tr and tr.routes:
            if len(tr.routes) > 1:
                draw_route(ax, town, tr.routes[0], color="tab:orange", lw=2.0, ls="--", alpha=0.6,
                           label="first route")
                draw_route(ax, town, tr.routes[-1], color="tab:purple", label="after the new closure")
            else:
                draw_route(ax, town, tr.routes[0],
                           label="plan_route's route" if node_only else "your route")
        ax.plot(trip.pose.x, trip.pose.y, marker=(3, 0, math.degrees(trip.pose.psi) - 90), ms=13,
                color="tab:green", zorder=8, ls="none", label="car")
        g = town.nodes[goal]
        ax.plot(g.x, g.y, "s", color="black", ms=9, zorder=8, ls="none", label=f"destination {trip.destination}")
        ax.set_title(title, fontsize=10)
    status = {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT"}[result.status]
    fail = "" if (tr is None or tr.ok) else f"\ntrip {k + 1}: {tr.message}"
    fig.suptitle(f"{scn.name} ({town.name}): {status}, score {result.score:.1f}/100   "
                 f"[trip {k + 1} of {len(scn.trips)} shown, to {trip.destination}]{fail}", fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=100)
    plt.close(fig)
