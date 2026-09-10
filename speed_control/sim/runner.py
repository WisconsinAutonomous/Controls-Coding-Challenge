"""Closed loop at 50 Hz: road/planner -> your controller -> car."""

from __future__ import annotations

import copy
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common import vehicle as V
from common.messages import CarTBS
from .car import Car, CarParams
from .road import STOP_WINDOW, STOPPED, Road
from .scenarios import Scenario

DT = 0.02
END_HOLD = 1.0
COLLISION_GAP = 0.3     # [m]
END_OVERRUN = 10.0      # [m] past the end of the road = never really stopped


@dataclass
class RunLog:
    scenario: Scenario
    status: str = "ok"          # ok | timeout | collision | overran | bad_output | exception | not_implemented
    message: str = ""
    completed: bool = False
    finish_time: float = math.nan
    nominal_time: float = math.nan
    stops: List[dict] = field(default_factory=list)
    data: Dict[str, np.ndarray] = field(default_factory=dict)


def run(scn: Scenario, controller_factory, params: Optional[CarParams] = None, seed: int = 0) -> RunLog:
    scn = copy.deepcopy(scn)
    params = params or scn.params
    road = Road(scn.length, scn.speed_limit, scn.stops, scn.v0, lead=scn.lead, seed=seed)
    car = Car(scn.v0, params, grade=scn.grade, seed=seed)
    log = RunLog(scn, nominal_time=scn.par_time or road.nominal_time())
    limit = 1.5 * log.nominal_time + 20.0
    try:
        ctrl = controller_factory()
    except Exception:
        log.status, log.message = "exception", traceback.format_exc(limit=3)
        return log

    rows = {k: [] for k in ("t", "x", "v", "v_meas", "v_target", "a_target", "limit", "accel",
                            "cmd_t", "cmd_b", "stop_distance", "lead_gap", "lead_v", "graded", "ms")}
    t, still_end = 0.0, 0.0
    while t < limit:
        meas = car.measure()
        tgt = road.target(car.x, meas.v)
        t0 = time.perf_counter()
        try:
            cmd = ctrl.compute(meas, tgt)
            if not isinstance(cmd, CarTBS):
                raise TypeError(f"compute() must return a CarTBS, got {type(cmd).__name__}")
            finite = math.isfinite(float(cmd.t)) and math.isfinite(float(cmd.b))
        except NotImplementedError as e:
            log.status, log.message = "not_implemented", str(e) or "a TODO in controller.py is not implemented yet"
            break
        except Exception:
            log.status, log.message = "exception", traceback.format_exc(limit=3)
            break
        ms = 1e3 * (time.perf_counter() - t0)
        if not finite:
            log.status = "bad_output"
            log.message = f"compute() returned t={cmd.t}, b={cmd.b} at t = {t:.2f} s; both must be finite numbers"
            break
        cmd = car.apply(cmd)
        car.step(DT)
        if road.lead is not None:
            road.lead.step(t, DT)
        t += DT
        stop = road.next_stop()
        x_stop = stop.x if stop is not None else scn.length
        holding = stop is not None and abs(car.x - stop.x) <= STOP_WINDOW and car.v < 0.3
        graded = (x_stop - car.x > 3.0) and not holding and car.x < scn.length - 3.0
        gap = lead_v = math.nan
        if road.lead is not None and not road.lead.gone:
            gap = road.lead.x - (car.x + V.REAR_AXLE_TO_FRONT_BUMPER)
            lead_v = road.lead.v
            if gap < 4.0 * max(car.v, 2.0):
                graded = False          # following: the gap decides the speed, not the plan
        road.update(car.x, car.v, DT)
        for k, val in (("t", t), ("x", car.x), ("v", car.v), ("v_meas", meas.v), ("v_target", tgt.v),
                       ("a_target", tgt.a), ("limit", road.limit_at(car.x)), ("accel", car.accel),
                       ("cmd_t", cmd.t), ("cmd_b", cmd.b),
                       ("stop_distance", math.nan if tgt.stop_distance is None else tgt.stop_distance),
                       ("lead_gap", gap), ("lead_v", lead_v), ("graded", float(graded)), ("ms", ms)):
            rows[k].append(val)
        if not math.isnan(gap) and gap <= COLLISION_GAP:
            log.status, log.message = "collision", f"hit the car ahead at t = {t:.1f} s"
            break
        if car.x > scn.length + END_OVERRUN:
            log.status = "overran"
            log.message = f"drove past the end of the road (more than {END_OVERRUN:.0f} m beyond it)"
            break
        if car.x >= scn.length - STOP_WINDOW and car.v < STOPPED:
            still_end += DT
            if still_end >= END_HOLD:
                log.completed, log.finish_time = True, t
                break
        else:
            still_end = 0.0
    else:
        log.status = "timeout"
        log.message = f"did not finish within {limit:.0f} s (reached {car.x:.0f} of {scn.length:.0f} m)"

    log.data = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    for s in road.stops:
        log.stops.append({"label": s.label, "x": s.x, "overrun": s.overrun, "error": s.error})
    if log.completed:
        log.stops.append({"label": "end of road", "x": scn.length, "overrun": False,
                          "error": float(car.x - scn.length)})
    return log
