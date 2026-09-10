"""The 0-100 score for a plan that passed every rule."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .checker import CheckResult

# metric: (points, good, bad).  Full points at `good`, zero at `bad`, linear between.
SCORING = {
    "length_ratio":    (30, 1.05, 2.00),   # path length / par length
    "min_clearance_m": (20, 0.30, 0.05),
    "lateral_m":       (10, 0.10, 0.45),   # car center to spot centerline
    "heading_deg":     (10, 1.0, 5.0),
    "plan_time_s":     (10, 2.0, 20.0),   # core; stretch uses STRETCH_TIME below
}
STRETCH_TIME = (5.0, 25.0)                 # full / zero planning-time points on stretch scenarios
CUSP_POINTS, CUSP_PENALTY = 15, 5.0        # full at <= par direction changes, -5 per extra
STEER_POINTS, STEER_PENALTY = 5, 1.0       # full at 0 steer-in-place events, -1 each


def _lin(value: float, good: float, bad: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def score(result: CheckResult, par_length: float, par_cusps: int, core: bool = True) -> Dict[str, float]:
    """Per-component points and the total.  A failed plan scores 0."""
    if not result.passed:
        return {"total": 0.0}
    m = dict(result.metrics)
    m["length_ratio"] = m["length_m"] / par_length
    table = dict(SCORING)
    if not core:
        table["plan_time_s"] = (SCORING["plan_time_s"][0],) + STRETCH_TIME
    pts = {k: w * _lin(m[k], good, bad) for k, (w, good, bad) in table.items()}
    pts["cusps"] = max(0.0, CUSP_POINTS - CUSP_PENALTY * max(0, m["cusps"] - par_cusps))
    pts["steer_events"] = max(0.0, STEER_POINTS - STEER_PENALTY * m["steer_events"])
    pts["total"] = float(sum(pts.values()))
    return pts
