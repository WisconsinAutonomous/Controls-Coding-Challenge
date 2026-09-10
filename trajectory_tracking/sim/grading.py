"""Metrics and the 0-100 score for one run."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .runner import CONTROL_DT, RunLog

# component: (weight, good, bad).  Score is linear from full weight at `good`
# to zero at `bad`.
SCORING = {
    "lat_rms_m":        (20, 0.03, 0.30),
    "lat_max_m":        (15, 0.10, 0.80),
    "speed_rms_mps":    (20, 0.10, 0.70),
    "stop_err_m":       (15, 0.10, 1.00),
    "jerk_rms":         (10, 1.00, 4.00),
    "steer_rate_rms":   (10, 0.03, 0.15),
    "time_ratio":       (10, 1.05, 1.40),
}


def _lin(value: float, good: float, bad: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def _masked_rms(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x[mask] ** 2))) if mask.any() else 0.0


def metrics(log: RunLog) -> Dict[str, float]:
    d = log.data
    if len(d.get("t", [])) < 2:
        return {}
    mask = d["t"] > log.scenario.lat_grace
    lat = np.abs(d["d"][mask]) if mask.any() else np.abs(d["d"])
    jerk = np.diff(d["accel"]) / CONTROL_DT
    steer_rate = np.diff(d["delta"]) / CONTROL_DT
    stop_errs = [abs(s["error"]) for s in log.stops if s["error"] is not None]
    m = {
        "lat_rms_m": float(np.sqrt(np.mean(lat ** 2))),
        "lat_max_m": float(np.max(lat)),
        "yaw_rms_deg": float(np.degrees(np.sqrt(np.mean(d["e_yaw"][mask] ** 2)))) if mask.any() else math.nan,
        "speed_rms_mps": _masked_rms(d["v"] - d["v_ref"], d["speed_graded"] > 0.5),
        "stop_err_m": float(np.mean(stop_errs)) if stop_errs else math.nan,
        "stops_overrun": int(sum(1 for s in log.stops if s["overrun"])),
        "jerk_rms": float(np.sqrt(np.mean(jerk ** 2))),
        "steer_rate_rms": float(np.sqrt(np.mean(steer_rate ** 2))),
        "lat_accel_max": float(np.max(np.abs(d["v"] ** 2 * d["kappa"]))),
        "time_ratio": (log.finish_time / log.nominal_time) if log.completed else math.nan,
        "compute_ms_p95": float(np.percentile(d["compute_ms"], 95)),
    }
    return m


def score(log: RunLog, m: Dict[str, float]) -> Dict[str, float]:
    """Per-component points and the total.  A failed run scores 0."""
    if log.status != "ok" or not log.completed or not m:
        return {"total": 0.0}
    pts = {}
    for key, (w, good, bad) in SCORING.items():
        val = m.get(key, math.nan)
        if key == "stop_err_m" and m.get("stops_overrun", 0) > 0:
            pts[key] = 0.0
        else:
            pts[key] = w * _lin(val, good, bad)
    pts["total"] = float(sum(pts.values()))
    return pts
