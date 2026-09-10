"""One figure per scenario: the lot, the path, and footprints along it."""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

from .lot import ParkingProblem, footprints  # noqa: E402

FILL = {"car": ("#c9d6e3", "#56687a"), "curb": ("#6b6b6b", "#444444"),
        "wall": ("#6b6b6b", "#444444"), "island": ("#9ccc8a", "#5a8a48"),
        "cone": ("#ff9933", "#b35900"), "cart": ("#b39ddb", "#5e35b1"),
        "person": ("#ef7d7d", "#b71c1c")}
FWD, REV = "#1f77b4", "#ff7f0e"


def _poly(ax, corners, face, edge, lw=1.0, alpha=1.0, ls="-", z=2):
    ax.add_patch(Polygon(corners, closed=True, facecolor=face, edgecolor=edge,
                         lw=lw, alpha=alpha, ls=ls, zorder=z))


def plot_problem(ax, problem: ParkingProblem) -> None:
    xmin, xmax, ymin, ymax = problem.bounds
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color="#333333", lw=2.5)
    for o in problem.obstacles:
        face, edge = FILL.get(o.kind, ("#dddddd", "#666666"))
        _poly(ax, o.corners(), face, edge)
    sp = problem.spot
    _poly(ax, sp.corners(), "#d9f2d0", "#2e7d32", lw=2.0, ls="--", z=1)
    ax.annotate("", xy=(sp.x + 1.6 * math.cos(sp.yaw), sp.y + 1.6 * math.sin(sp.yaw)),
                xytext=(sp.x - 0.8 * math.cos(sp.yaw), sp.y - 0.8 * math.sin(sp.yaw)),
                arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2), zorder=3)
    st = problem.start
    _poly(ax, footprints([st.x], [st.y], [st.yaw])[0], "none", "black", lw=1.8, z=4)
    ax.plot(st.x, st.y, "o", color="black", ms=4, zorder=5)


def _direct_shot(problem):
    """The shortest forward Dubins path and the pose where it first hits something (or None)."""
    try:
        import tools   # parking/tools.py (on the path when run from run.py / check.py)
    except ImportError:
        return None
    d = tools.dubins_path(problem.start, problem.spot.goal_pose())
    if d is None:
        return None
    hit = problem.collides_many(d.x, d.y, d.yaw)
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    return d, (float(d.x[i]), float(d.y[i]), float(d.yaw[i]))


def plot_run(problem: ParkingProblem, result, score_total: float, path: str) -> None:
    xmin, xmax, ymin, ymax = problem.bounds
    w = 12.0
    h = min(12.0, max(4.5, w * (ymax - ymin) / (xmax - xmin) + 1.2))
    fig, ax = plt.subplots(figsize=(w, h))
    plot_problem(ax, problem)

    shot = _direct_shot(problem)
    if shot is not None and len(result.dense) == 1:
        # your path IS the direct shot (the fallback): no need to draw it twice
        seg0 = result.dense[0]
        if abs(seg0["s"][-1] - shot[0].length) < 0.05 and math.hypot(
                seg0["x"][-1] - shot[0].x[-1], seg0["y"][-1] - shot[0].y[-1]) < 0.05:
            shot = None
    if shot is not None:
        d, pose = shot
        ax.plot(d.x, d.y, color="0.45", lw=1.2, ls="--", zorder=4)
        _poly(ax, footprints(*[[v] for v in pose])[0], "none", "#c62828", lw=1.0, ls="--", alpha=0.7, z=4)
    for seg in result.dense:
        col = REV if seg["reverse"] else FWD
        ax.plot(seg["x"], seg["y"], color=col, lw=2.2, zorder=6)
        s = seg["s"]
        marks = np.searchsorted(s, np.arange(0.0, s[-1] + 1e-9, 1.5))
        marks = np.unique(np.clip(marks, 0, len(s) - 1))
        for poly in footprints(seg["x"][marks], seg["y"][marks], seg["yaw"][marks]):
            _poly(ax, poly, "none", col, lw=0.7, alpha=0.45, z=5)
    for cx, cy in result.cusps:
        ax.plot(cx, cy, "D", color="black", ms=7, zorder=8)
    if result.final_pose is not None:
        fx, fy, fyaw = result.final_pose
        ok = result.passed
        _poly(ax, footprints([fx], [fy], [fyaw])[0], "none", "#2e7d32" if ok else "#c62828", lw=2.2, z=7)
    if getattr(result, "fail_pose", None) is not None:
        _poly(ax, footprints(*[[v] for v in result.fail_pose])[0], "#ef9a9a", "#c62828", lw=2.0, alpha=0.6, z=7)
    if result.fail_xy is not None:
        ax.plot(*result.fail_xy, "x", color="red", ms=18, mew=3.5, zorder=9)

    handles = [Line2D([], [], color=FWD, lw=2.2, label="your path, forward"),
               Line2D([], [], color=REV, lw=2.2, label="your path, reverse"),
               Line2D([], [], color="black", marker="D", ms=6, ls="none", label="direction change")]
    if shot is not None:
        handles.append(Line2D([], [], color="0.45", lw=1.2, ls="--",
                              label="shortest forward path, and where it would hit (red dashed)"))
    if getattr(result, "fail_pose", None) is not None:
        handles.append(Patch(facecolor="#ef9a9a", edgecolor="#c62828", label="your car at first contact (X)"))
    ax.set_aspect("equal")
    pad = 1.0
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x East [m]")
    ax.set_ylabel("y North [m]")
    ax.grid(alpha=0.25)
    status = "PASS" if result.passed else f"FAILED: {result.reason}"
    ax.set_title(f"{problem.name}: {status}, score {score_total:.1f}/100\n{problem.description}", fontsize=11)
    # legend in its own strip under the axes, sized in inches so it never hits the x label
    rows = 2 if len(handles) > 3 else 1
    strip = (0.28 * rows + 0.15) / h
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, strip, 1, 1))
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_lot(problem: ParkingProblem, path: str, title: Optional[str] = None) -> None:
    """Just the lot (handy while designing scenarios)."""
    xmin, xmax, ymin, ymax = problem.bounds
    fig, ax = plt.subplots(figsize=(12, max(4.5, 12 * (ymax - ymin) / (xmax - xmin) + 1.0)))
    plot_problem(ax, problem)
    ax.set_aspect("equal")
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.grid(alpha=0.25)
    ax.set_title(title or problem.name)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
