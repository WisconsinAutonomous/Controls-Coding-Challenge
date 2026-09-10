"""Plots and animations of a run."""

from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.collections import LineCollection    # noqa: E402
from matplotlib.patches import Polygon, Rectangle    # noqa: E402

from common.messages import LightColor, ObjClass     # noqa: E402

from . import geom                                   # noqa: E402
from .world import (DASHED_WHITE, DASHED_YELLOW, DOUBLE_YELLOW, SOLID_WHITE,  # noqa: E402
                    SOLID_YELLOW)

GOLD, GRAY, ROAD = "#d4a017", "#606060", "#e9e9e9"
OBS_COLOR = {ObjClass.BARREL: "#f28e2b", ObjClass.CONE: "#e15759",
             ObjClass.BARRICADE: "#b07aa1"}
LIGHT_RGB = {LightColor.RED: "#d62728", LightColor.YELLOW: "#f2c500",
             LightColor.GREEN: "#2ca02c"}
STOP_COLOR = {"stop_sign": "#d62728", "traffic_light": "#222222", "crosswalk": "#1f77b4"}


def _line_style(kind):
    return {SOLID_WHITE: dict(color=GRAY, ls="-"), DASHED_WHITE: dict(color=GRAY, ls=(0, (4, 4))),
            SOLID_YELLOW: dict(color=GOLD, ls="-"), DASHED_YELLOW: dict(color=GOLD, ls=(0, (4, 4))),
            DOUBLE_YELLOW: dict(color=GOLD, ls="-")}.get(kind, dict(color=GRAY, ls="-"))


def _draw_road(ax, route, s0=0.0, s1=None):
    ref = route.reference
    s1 = ref.length if s1 is None else s1
    s = np.linspace(max(s0, 0.0), min(s1, ref.length), max(2, int((s1 - s0) / 0.5)))
    half = 0.5 * route.lane_width
    left_edge = 3 * half if route.left_lane else half
    xr, yr = ref.frenet_to_xy(s, -half)
    xl, yl = ref.frenet_to_xy(s, left_edge)
    ax.add_patch(Polygon(np.column_stack([np.r_[xr, xl[::-1]], np.r_[yr, yl[::-1]]]),
                         closed=True, fc=ROAD, ec="none", zorder=0))
    ax.plot(xr, yr, color=GRAY, lw=1.2, zorder=1)
    if route.left_lane:
        ax.plot(xl, yl, color=GRAY, lw=1.2, zorder=1)
    for span in route.lane_lines:
        if span.side != "left":
            continue
        a, b = max(span.s_start, s[0]), min(span.s_end, s[-1])
        if b <= a:
            continue
        ss = np.linspace(a, b, max(2, int((b - a) / 0.5)))
        st = _line_style(span.kind)
        offs = (half - 0.08, half + 0.08) if span.kind == DOUBLE_YELLOW else (half,)
        for o in offs:
            x, y = ref.frenet_to_xy(ss, o)
            ax.plot(x, y, lw=1.0, zorder=1, **st)
    for sl in route.stop_lines:
        if not (s[0] <= sl.s <= s[-1]):
            continue
        if sl.kind == "crosswalk":
            for d0 in np.arange(-half, left_edge, 0.6):
                a0, a1 = sl.crosswalk_s_start, sl.crosswalk_s_end
                xs, ys = ref.frenet_to_xy(np.array([a0, a1, a1, a0]), np.array([d0, d0, d0 + 0.3, d0 + 0.3]))
                ax.add_patch(Polygon(np.column_stack([xs, ys]), fc="white", ec="#9ecae1", lw=0.4, zorder=1))
        x, y = ref.frenet_to_xy(np.array([sl.s, sl.s]), np.array([-half, half]))
        ax.plot(x, y, color=STOP_COLOR[sl.kind], lw=2.5, zorder=2)


def _draw_obstacles(ax, world):
    for ob in world.obstacles:
        hatch = "////" if ob.obj_class == ObjClass.BARRICADE else None
        ax.add_patch(Polygon(ob.corners(), closed=True, zorder=3, hatch=hatch,
                             fc=OBS_COLOR.get(ob.obj_class, "#999999"), ec="black", lw=0.6))


def plot_run(res, g, path: str) -> None:
    sc, route, ref = res.scenario, res.scenario.route, res.scenario.route.reference
    tr, ticks = res.trace, res.ticks
    half = 0.5 * route.lane_width
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[5.2, 2.6, 2.2], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    head = f"{sc.name} (seed {sc.seed}): {g.status}  {g.score:.1f}/100"
    sub = g.reason if g.status == "FAIL" else (g.top_penalties(4) or "no penalties")
    fig.suptitle(head + "\n" + sub, fontsize=12, y=0.995)

    # ---- top-down
    _draw_road(ax, route)
    _draw_obstacles(ax, sc.world)
    for p in sc.world.pedestrians:
        xa, ya = ref.frenet_to_xy(p.s, p.d_start)
        xb, yb = ref.frenet_to_xy(p.s, p.d_end)
        if p.crosses:
            ax.annotate("", (xb, yb), (xa, ya), arrowprops=dict(arrowstyle="->", color="#9467bd", ls=":"))
        ax.plot(xa, ya, "o", color="#9467bd", ms=5, zorder=4)
    for k, L in enumerate(sc.world.lights):
        ax.plot(L.x, L.y, "D", color="#f2c500", mec="#222222", ms=7, zorder=4,
                label="traffic light" if k == 0 else None)
    for tk in ticks[::20]:
        if tk.plan is not None:
            ax.plot(tk.plan[:, 0], tk.plan[:, 1], color="#1f77b4", lw=0.7, alpha=0.35, zorder=2)
    for tk in ticks[::20]:
        c = geom.car_corners(tk.x, tk.y, tk.psi)
        ax.add_patch(Polygon(c, closed=True, fc="none", ec="#1f77b4", lw=0.6, alpha=0.5, zorder=4))
    vmax = max(route.default_speed_limit, float(tr[:, 4].max()) if len(tr) else 1.0)
    if len(tr) > 1:
        pts = tr[:, 1:3].reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(0, vmax), lw=2.4, zorder=5)
        lc.set_array(tr[1:, 4])
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("speed [m/s]")
    x0, y0, _ = ref.interp(sc.start_s)
    xg, yg, _ = ref.interp(route.goal_s)
    ax.plot(x0, y0, "^", color="green", ms=9, zorder=6, label="start")
    ax.plot(xg, yg, "s", color="black", ms=8, zorder=6, label="goal")
    if g.fail_xy:
        ax.plot(*g.fail_xy, "x", color="red", ms=16, mew=3, zorder=7)
        ax.annotate(g.reason[:60], g.fail_xy, textcoords="offset points", xytext=(10, 10),
                    color="red", fontsize=9, zorder=7,
                    bbox=dict(fc="white", ec="red", alpha=0.85))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # ---- Frenet view
    ax2 = fig.add_subplot(gs[1])
    s_t = np.array([tk.s for tk in ticks]) if ticks else np.zeros(0)
    if len(s_t):
        env = np.array([_corner_d(ref, tk) for tk in ticks])
        ax2.fill_between(s_t, env[:, 0], env[:, 1], color="#1f77b4", alpha=0.2, lw=0,
                         label="car footprint")
        ax2.plot(s_t, [tk.d for tk in ticks], color="#1f77b4", lw=1.3, label="rear axle")
    smax = max(route.goal_s + 10, s_t.max() + 5 if len(s_t) else 0)
    ax2.axhline(-half, color=GRAY, lw=1.2)
    if route.left_lane:
        ax2.axhline(3 * half, color=GRAY, lw=1.2)
        ax2.axhline(2 * half, color=GRAY, lw=0.6, ls=":")
    ax2.axhline(0.0, color=GRAY, lw=0.6, ls=":")
    for span in route.lane_lines:
        if span.side == "left":
            st = _line_style(span.kind)
            ax2.plot([span.s_start, min(span.s_end, smax)], [half, half], lw=2.0 if
                     span.kind == DOUBLE_YELLOW else 1.2, **st)
    min_w = 0.004 * smax                     # keep short obstacles visible on a long route
    for ob in sc.world.obstacles:
        w = max(ob.length, min_w)
        ax2.add_patch(Rectangle((ob.s - 0.5 * w, ob.d - 0.5 * ob.width), w,
                                ob.width, fc=OBS_COLOR.get(ob.obj_class, "#999"), ec="black", lw=0.5))
    for p in sc.world.pedestrians:
        ax2.plot([p.s, p.s], [p.d_start, p.d_end] if p.crosses else [p.d_start, p.d_start + 0.01],
                 color="#9467bd", ls=":", marker="o", ms=3)
    _stop_lines_v(ax2, route)
    ax2.set_xlim(0, smax)
    ax2.set_ylim(-3.2, 6.6 if route.left_lane else 3.0)
    ax2.set_ylabel("d [m] (left +)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_title("Frenet view: lateral offset from your lane center", fontsize=10)

    # ---- speed
    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ss = np.linspace(0, smax, 800)
    ax3.plot(ss, [route.speed_limit_at(si) for si in ss], color="#d62728", ls="--", lw=1,
             label="speed limit")
    if len(s_t):
        ax3.plot(s_t, [tk.v for tk in ticks], color="#1f77b4", lw=1.5, label="speed")
    _stop_lines_v(ax3, route, label=True)
    if g.fail_xy and len(s_t):
        sfail, _ = geom.project_points(ref, g.fail_xy[0], g.fail_xy[1], s_t[-1], 30)
        for a in (ax2, ax3):
            a.axvline(float(sfail[0]), color="red", lw=1.5)
    ax3.set_xlabel("s along route [m]")
    ax3.set_ylabel("speed [m/s]")
    ax3.set_ylim(0, vmax + 1.5)
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(alpha=0.25)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


STOP_LABEL = {"stop_sign": "STOP", "traffic_light": "light", "crosswalk": "crosswalk"}


def _stop_lines_v(ax, route, label=False):
    for sl in route.stop_lines:
        ax.axvline(sl.s, color=STOP_COLOR[sl.kind], lw=1.2, alpha=0.8)
        if sl.kind == "crosswalk":
            ax.axvspan(sl.crosswalk_s_start, sl.crosswalk_s_end, color="#9ecae1", alpha=0.4)
        if label:
            ax.annotate(STOP_LABEL[sl.kind], (sl.s, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -12), textcoords="offset points", fontsize=8,
                        color=STOP_COLOR[sl.kind])


def _corner_d(ref, tk):
    c = geom.car_corners(tk.x, tk.y, tk.psi)
    _, d = geom.project_points(ref, c[:, 0], c[:, 1], tk.s, 15.0)
    return d.min(), d.max()


def animate_run(res, g, path: str, every: int = 3, max_frames: int = 300) -> None:
    """Follow-cam GIF: road, obstacles, pedestrians, light colors, detections, plan."""
    from matplotlib.animation import PillowWriter
    sc, route = res.scenario, res.scenario.route
    ticks = res.ticks[::every][:max_frames]
    if not ticks:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    writer = PillowWriter(fps=10)
    with writer.saving(fig, path, dpi=80):
        for tk in ticks:
            ax.clear()
            s0 = tk.s
            _draw_road(ax, route, s0 - 25, s0 + 50)
            _draw_obstacles(ax, sc.world)
            for p in sc.world.pedestrians:
                px, py = sc.world.pedestrian_xy(p, tk.t)
                ax.plot(px, py, "o", color="#9467bd", ms=8, zorder=6)
            for L in sc.world.lights:
                ax.plot(L.x, L.y, "o", color=LIGHT_RGB.get(L.color(tk.t), "gray"), ms=11,
                        mec="black", zorder=6)
            if tk.dets is not None and len(tk.dets):
                ax.plot(tk.dets[:, 0], tk.dets[:, 1], "x", color="black", ms=5, zorder=7)
            if tk.plan is not None:
                ax.plot(tk.plan[:, 0], tk.plan[:, 1], color="#1f77b4", lw=1.5, zorder=5)
            ax.add_patch(Polygon(geom.car_corners(tk.x, tk.y, tk.psi), closed=True,
                                 fc="#1f77b4", ec="black", alpha=0.8, zorder=8))
            ax.set_xlim(tk.x - 30, tk.x + 30)
            ax.set_ylim(tk.y - 20, tk.y + 20)
            ax.set_aspect("equal")
            ax.set_title(f"{sc.name}  t={tk.t:5.1f}s  v={tk.v:4.1f} m/s   (x = detections)",
                         fontsize=10)
            writer.grab_frame()
        if g.fail_xy:
            ax.plot(*g.fail_xy, "x", color="red", ms=20, mew=4, zorder=9)
            ax.set_title(f"{sc.name}: FAIL, {g.reason[:50]}", fontsize=10, color="red")
            for _ in range(10):
                writer.grab_frame()
    plt.close(fig)
