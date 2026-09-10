"""One figure per run: speed, acceleration, and position or gap."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .runner import RunLog  # noqa: E402


def plot_run(log: RunLog, total: float, path: str) -> None:
    d, scn = log.data, log.scenario
    follow = scn.lead is not None
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax_v, ax_a, ax_x = axes
    if len(d.get("t", [])):
        t = d["t"]
        ax_v.plot(t, d["limit"], color="red", lw=1, ls=":", label="speed limit")
        ax_v.plot(t, d["v_target"], color="0.3", lw=1.2, ls="--", label="target")
        ax_v.plot(t, d["v"], color="tab:blue", lw=1.6, label="car")
        if follow:
            ax_v.plot(t, d["lead_v"], color="tab:purple", lw=1, label="car ahead")
        ax_v.set_ylabel("speed [m/s]")
        ax_v.legend(fontsize=8, loc="upper right")

        ax_a.plot(t, d["cmd_t"] + d["cmd_b"], color="tab:orange", lw=1, label="requested (t + b)")
        ax_a.plot(t, d["accel"], color="tab:blue", lw=1.4, label="actual")
        ax_a.axhline(0, color="0.5", lw=0.6)
        ax_a.set_ylabel("acceleration [m/s²]")
        ax_a.legend(fontsize=8, loc="upper right")

        if follow:
            ax_x.plot(t, d["lead_gap"], color="tab:blue", lw=1.4, label="gap to the car ahead")
            safe = np.where(np.isfinite(d["lead_gap"]), 1.5 * np.maximum(d["v"], 2.0), np.nan)
            ax_x.plot(t, safe, color="red", lw=1, ls=":", label="1.5 s gap")
            ax_x.set_ylabel("gap [m]")
        else:
            ax_x.plot(t, d["x"], color="tab:blue", lw=1.4, label="position")
            for s in log.stops:
                ax_x.axhline(s["x"], color="red", lw=0.8, ls="--")
                ax_x.annotate(s["label"], (t[0], s["x"]), color="red", fontsize=8, va="bottom")
            ax_x.set_ylabel("position [m]")
        ax_x.legend(fontsize=8, loc="lower right")
        ax_x.set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    status = "completed" if log.completed else f"FAILED ({log.status}): {log.message.strip().splitlines()[-1] if log.message else ''}"
    fig.suptitle(f"{scn.name}: {status}, score {total:.0f}/100\n{scn.description}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
