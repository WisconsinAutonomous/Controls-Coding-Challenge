"""Metrics and the 0-100 score."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .runner import DT, RunLog

# (points, good, bad): full points at `good`, zero at `bad`, linear in between.
SCORING = {
    "speed_rms":   (30, 0.15, 1.00),   # [m/s] vs the target, away from stops
    "speeding":    (15, 0.20, 1.50),   # [m/s] worst speed above the limit
    "stop_error":  (25, 0.20, 1.50),   # [m] mean distance from the stop points
    "jerk_rms":    (10, 1.00, 4.00),   # [m/s^3] what the passengers feel
    "cmd_rate":    (10, 8.00, 30.0),   # [m/s^3] how jumpy your throttle/brake requests are
    "time_ratio":  (10, 1.05, 1.40),   # finish time / planned time
}
# Adaptive cruise (follow_the_leader) swaps speed tracking for gap keeping.
SCORING_FOLLOW = {
    "min_time_gap": (30, 1.50, 0.70),  # [s] smallest gap / speed (higher is better)
    "gap_rms":      (15, 0.30, 1.20),  # [s] RMS of (time gap - 2 s) while following
    "speeding":     (10, 0.20, 1.50),
    "stop_error":   (15, 0.20, 1.50),
    "jerk_rms":     (15, 1.00, 4.00),
    "time_ratio":   (15, 1.05, 1.40),
}


def _lin(val: float, good: float, bad: float) -> float:
    if not math.isfinite(val):
        return 0.0
    return float(np.clip((bad - val) / (bad - good), 0.0, 1.0))


def metrics(log: RunLog) -> Dict[str, float]:
    d = log.data
    if len(d.get("t", [])) < 2:
        return {}
    g = d["graded"] > 0.5
    err = (d["v"] - d["v_target"])[g]
    errs = [abs(s["error"]) for s in log.stops if s["error"] is not None]
    m = {
        "speed_rms": float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0,
        "speeding": float(max(0.0, np.max(d["v"] - d["limit"]))),
        "stop_error": float(np.mean(errs)) if errs else math.nan,
        "overruns": int(sum(1 for s in log.stops if s["overrun"])),
        "jerk_rms": float(np.sqrt(np.mean((np.diff(d["accel"]) / DT) ** 2))),
        "cmd_rate": float(np.sqrt(np.mean((np.diff(d["cmd_t"] + d["cmd_b"]) / DT) ** 2))),
        "time_ratio": log.finish_time / log.nominal_time if log.completed else math.nan,
        "ms_p95": float(np.percentile(d["ms"], 95)),
    }
    gap = d["lead_gap"]
    has = np.isfinite(gap)
    if has.any():
        th = gap[has] / np.maximum(d["v"][has], 2.0)
        m["min_time_gap"] = float(np.min(th))
        following = th < 4.0
        m["gap_rms"] = float(np.sqrt(np.mean((th[following] - 2.0) ** 2))) if following.any() else 0.0
    return m


def score(log: RunLog, m: Dict[str, float]) -> Dict[str, float]:
    if not log.completed or log.status != "ok" or not m:
        return {"total": 0.0}
    table = SCORING_FOLLOW if log.scenario.lead is not None else SCORING
    pts = {}
    for key, (w, good, bad) in table.items():
        if key == "stop_error" and m.get("overruns", 0):
            pts[key] = 0.0
        elif key == "min_time_gap":
            pts[key] = w * float(np.clip((m[key] - bad) / (good - bad), 0.0, 1.0))
        else:
            pts[key] = w * _lin(m.get(key, math.nan), good, bad)
    pts["total"] = float(sum(pts.values()))
    return pts
