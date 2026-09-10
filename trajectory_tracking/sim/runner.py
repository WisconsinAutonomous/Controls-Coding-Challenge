"""Closed-loop simulation: reference -> your controller -> plant, at 50 Hz."""

from __future__ import annotations

import copy
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from common.geometry import wrap_to_pi
from common.messages import CarTBS
from .plant import NOMINAL, Plant, PlantParams, TrueState
from .reference import STOP_WINDOW, STOPPED_SPEED, ReferenceProvider
from .scenarios import Scenario

CONTROL_DT = 0.02        # [s] 50 Hz, like the real MPC node
PUBLISH_EVERY = 5        # reference published every 5 control ticks (10 Hz)
OFF_ROAD = 2.5           # [m] lateral error that ends the run
END_HOLD = 1.0           # [s] standstill at the end of the route to finish
STOP_APPROACH = 2.0      # [m] before a stop, speed error is not graded


@dataclass
class RunLog:
    scenario: Scenario
    params: PlantParams
    status: str = "ok"               # ok | off_road | timeout | exception | not_implemented | bad_output
    message: str = ""
    completed: bool = False
    finish_time: float = math.nan
    nominal_time: float = math.nan
    stops: List[dict] = field(default_factory=list)
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: List[dict] = field(default_factory=list)


def make_plant(scn: Scenario, params: PlantParams, seed: int, backend: str):
    x, y, yaw = (float(a) for a in scn.route.interp(0.0))
    x -= scn.d0 * math.sin(yaw)
    y += scn.d0 * math.cos(yaw)
    x0 = TrueState(x=x, y=y, psi=yaw + scn.dpsi0, v=scn.v0)
    if backend == "chrono":
        from .chrono_plant import ChronoPlant
        return ChronoPlant(x0, params, seed=seed)
    return Plant(x0, params, grade_fn=scn.grade, seed=seed)


def run(scn: Scenario, controller_factory: Callable[[], object],
        params: PlantParams = NOMINAL, seed: int = 0, backend: str = "numpy") -> RunLog:
    scn = copy.deepcopy(scn)
    ref = ReferenceProvider(scn.route, scn.speed_limit, scn.stops, scn.v0)
    log = RunLog(scenario=scn, params=params, nominal_time=ref.nominal_time())
    time_limit = 1.5 * log.nominal_time + 15.0
    plant = make_plant(scn, params, seed, backend)

    try:
        ctrl = controller_factory()
    except Exception:
        log.status, log.message = "exception", traceback.format_exc(limit=4)
        return log

    rows: Dict[str, list] = {k: [] for k in (
        "t", "x", "y", "psi", "v", "delta", "accel", "s", "d", "e_yaw", "v_ref",
        "kappa_ref", "kappa", "cmd_t", "cmd_b", "cmd_s", "compute_ms", "speed_graded")}
    s_true, t, tick, still_end = 0.0, 0.0, 0, 0.0
    traj = None
    L = scn.route.length

    while t < time_limit:
        meas = plant.measure()
        if tick % PUBLISH_EVERY == 0:
            traj = ref.publish(meas, stamp=t)
            if tick % (PUBLISH_EVERY * 20) == 0:
                arr = traj.as_arrays()
                log.snapshots.append({"t": t, "x": arr["x"], "y": arr["y"]})
        t0 = time.perf_counter()
        try:
            cmd = ctrl.compute(meas, traj)
            if not isinstance(cmd, CarTBS):
                raise TypeError(f"compute() must return CarTBS, got {type(cmd).__name__}")
            finite = all(math.isfinite(float(v)) for v in (cmd.t, cmd.b, cmd.s))
        except NotImplementedError as e:
            log.status, log.message = "not_implemented", str(e) or "a TODO in controller.py is not written yet"
            break
        except Exception:
            log.status, log.message = "exception", traceback.format_exc(limit=4)
            break
        compute_ms = 1e3 * (time.perf_counter() - t0)
        if not finite:
            log.status = "bad_output"
            log.message = (f"compute() returned t={cmd.t}, b={cmd.b}, s={cmd.s} at t = {t:.2f} s; "
                           "all three must be finite numbers")
            break
        cmd = plant.apply(cmd)
        plant.step(CONTROL_DT)
        t += CONTROL_DT
        tick += 1

        st = plant.s
        s_true, d_true, _ = scn.route.project(st.x, st.y, s_hint=s_true, window=15.0)
        # The last few meters before a stop are graded by stop precision, not
        # by speed error (the profile is a steep sqrt there).
        stp = ref.active_stop()
        graded = not ((stp is not None and s_true > stp.s - STOP_APPROACH) or s_true > L - STOP_APPROACH)
        ref.update(s_true, st.v, CONTROL_DT)
        _, _, yaw_ref = scn.route.interp(s_true)
        for k, val in (("t", t), ("x", st.x), ("y", st.y), ("psi", st.psi), ("v", st.v),
                       ("delta", st.delta), ("accel", st.accel), ("s", s_true),
                       ("d", d_true), ("e_yaw", float(wrap_to_pi(st.psi - yaw_ref))),
                       ("v_ref", ref.v_ref_at(s_true)),
                       ("kappa_ref", float(scn.route.curvature_at(s_true))),
                       ("kappa", plant.curvature()), ("cmd_t", cmd.t), ("cmd_b", cmd.b),
                       ("cmd_s", cmd.s), ("compute_ms", compute_ms),
                       ("speed_graded", float(graded))):
            rows[k].append(val)

        if abs(d_true) > OFF_ROAD and t > scn.lat_grace:
            log.status = "off_road"
            log.message = f"left the road: {abs(d_true):.1f} m from the path at s = {s_true:.0f} m"
            break
        if s_true >= L - STOP_WINDOW and st.v < STOPPED_SPEED:
            still_end += CONTROL_DT
            if still_end >= END_HOLD:
                log.completed = True
                log.finish_time = t
                break
        else:
            still_end = 0.0
    else:
        log.status = "timeout"
        log.message = f"did not finish within {time_limit:.0f} s (reached s = {s_true:.0f} of {L:.0f} m)"

    log.data = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    for stp in ref.stops:
        log.stops.append({"label": stp.label, "s": stp.s, "released": stp.released,
                          "overrun": stp.overrun, "error": stp.stop_error})
    if log.completed:
        log.stops.append({"label": "end of route", "s": L, "released": True,
                          "overrun": False, "error": float(s_true - L)})
    return log
