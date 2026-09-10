"""One diagnostic figure per run."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import vehicle as V  # noqa: E402
from .runner import RunLog  # noqa: E402

LANE_HALF = 1.8


def plot_run(log: RunLog, total: float, path: str) -> None:
    scn, d = log.scenario, log.data
    route = scn.route
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_lat)
    ax_st = fig.add_subplot(gs[2, 1], sharex=ax_lat)

    # --- map --------------------------------------------------------------
    s = route.s
    for off in (-LANE_HALF, LANE_HALF):
        ex, ey = route.frenet_to_xy(s, np.full_like(s, off))
        ax_map.plot(ex, ey, color="0.6", lw=1.0)
    ax_map.plot(route.x, route.y, "--", color="0.45", lw=1.0, label="reference path")
    for snap in log.snapshots:
        ax_map.plot(snap["x"], snap["y"], color="tab:orange", lw=3, alpha=0.15)
    if len(d.get("x", [])):
        ax_map.plot(d["x"], d["y"], color="tab:blue", lw=1.6, label="car (rear axle)")
        ax_map.plot(d["x"][0], d["y"][0], "o", color="tab:blue")
        if log.status != "ok":
            ax_map.plot(d["x"][-1], d["y"][-1], "x", color="red", ms=14, mew=3)
    for stp in log.stops:
        if stp["label"] == "end of route":
            continue
        sx, sy, sh = (float(v) for v in route.interp(stp["s"]))
        nx, ny = -math.sin(sh), math.cos(sh)
        ax_map.plot([sx - LANE_HALF * nx, sx + LANE_HALF * nx],
                    [sy - LANE_HALF * ny, sy + LANE_HALF * ny], color="red", lw=3)
        ax_map.annotate(stp["label"], (sx + 2.2 * nx, sy + 2.2 * ny), color="red", fontsize=8)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("x East [m]")
    ax_map.set_ylabel("y North [m]")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(alpha=0.3)

    if len(d.get("t", [])):
        t = d["t"]
        ax_lat.plot(t, d["d"], color="tab:blue")
        for k, yv in enumerate((-0.9, 0.9)):
            ax_lat.axhline(yv, color="red", lw=0.8, ls=":",
                           label="car body touches the lane line" if k == 0 else None)
        ax_lat.legend(fontsize=8, loc="upper right")
        ax_lat.axhline(0, color="0.5", lw=0.6)
        ax_lat.set_ylabel("lateral error [m]\n(+ = left of path)")
        ax_lat.grid(alpha=0.3)

        ax_v.plot(t, d["v_ref"], "--", color="0.3", label="reference")
        ax_v.plot(t, d["v"], color="tab:blue", label="car")
        ax_v.set_ylabel("speed [m/s]")
        ax_v.legend(fontsize=8, loc="best")
        ax_v.grid(alpha=0.3)

        ax_st.plot(t, np.degrees(np.radians(d["cmd_s"] / V.STEERING_RATIO)),
                   color="tab:orange", lw=1.0, label="commanded")
        ax_st.plot(t, np.degrees(d["delta"]), color="tab:blue", lw=1.2, label="actual")
        ax_st.set_ylabel("road-wheel angle [deg]")
        ax_st.set_xlabel("time [s]")
        ax_st.legend(fontsize=8, loc="best")
        ax_st.grid(alpha=0.3)

    status = "completed" if log.completed else f"FAILED ({log.status})"
    fig.suptitle(f"{scn.name}: {status}, score {total:.1f}/100\n{scn.description}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
