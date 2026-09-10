"""Optional PyChrono backend: the same controller on a multibody sedan.

This swaps the NumPy kinematic plant for Chrono's ``Sedan`` model on flat
rigid ground: real tires (slip, load transfer), suspension, a geared
powertrain with engine braking, and Ackermann steering geometry.  Your
controller sees the same ``CarState`` and sends the same ``CarTBS``; the
same steering delay / lag / rate limit is applied in front of Chrono's
steering, and a small drive-by-wire loop turns the acceleration request into
throttle and brake pedal, like the real car's DBW does.

It is NOT our car: wheelbase 2.78 m (ours 2.67), max road-wheel angle 25 deg
(ours 30), different mass and powertrain.  That mismatch is the point.

Install (conda only):
    conda install -c conda-forge -c projectchrono pychrono
"""

from __future__ import annotations

import math

import numpy as np

try:
    import pychrono as chrono
    import pychrono.vehicle as veh
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The chrono backend needs PyChrono: conda install -c conda-forge -c projectchrono pychrono"
    ) from e

from common import vehicle as V
from common.messages import CarState, CarTBS
from .plant import PlantParams, TrueState, _DelayLine, _finite

STEP = 1e-3            # [s] Chrono integration step
COAST_DECEL = 0.4      # [m/s^2] engine braking + drag when coasting (measured)


class ChronoPlant:
    def __init__(self, x0: TrueState, params: PlantParams, seed: int = 0):
        veh.SetVehicleDataPath(chrono.GetChronoDataPath() + "vehicle/")
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        car = veh.Sedan()
        car.SetContactMethod(chrono.ChContactMethod_NSC)
        car.SetChassisFixed(False)
        half_wb = 0.5 * 2.776   # chassis reference is midway between the axles
        cx = x0.x + half_wb * math.cos(x0.psi)
        cy = x0.y + half_wb * math.sin(x0.psi)
        car.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(cx, cy, 0.4),
                                                chrono.QuatFromAngleZ(x0.psi)))
        car.SetInitFwdVel(x0.v)
        car.SetTireType(veh.TireModelType_TMEASY)
        car.SetTireStepSize(STEP)
        car.Initialize()
        self.car = car
        self.veh = car.GetVehicle()
        self.max_steer = self.veh.GetMaxSteeringAngle()
        self.wheelbase = self.veh.GetWheelbase()

        self.terrain = veh.RigidTerrain(car.GetSystem())
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(0.9)
        mat.SetRestitution(0.01)
        self.terrain.AddPatch(mat, chrono.ChCoordsysd(chrono.ChVector3d(x0.x, x0.y, 0), chrono.QUNIT),
                              3000, 3000)
        self.terrain.Initialize()
        self.inputs = veh.DriverInputs()

        self._steer_line = _DelayLine(params.steer_delay, 0.0)
        self._accel_line = _DelayLine(params.accel_delay, 0.0)
        self.delta = 0.0          # road-wheel angle after the actuator model
        self._a_meas = 0.0     # DBW's acceleration estimate (fast)
        self._a_comfort = 0.0  # ~1 Hz filtered acceleration, used for the jerk metric
        self._a_int = 0.0
        self._v_prev = x0.v
        self.last_cmd = CarTBS()
        self.s = TrueState(x=x0.x, y=x0.y, psi=x0.psi, v=x0.v)
        self._update_state()

    # ------------------------------------------------------------------ #
    def apply(self, cmd: CarTBS) -> CarTBS:
        t = float(np.clip(_finite(cmd.t), 0.0, V.MAX_THROTTLE))
        b = float(np.clip(_finite(cmd.b), -V.MAX_BRAKE, 0.0))
        s = float(np.clip(_finite(cmd.s), -V.MAX_STEER_COLUMN_DEG, V.MAX_STEER_COLUMN_DEG))
        self.last_cmd = CarTBS(t=t, b=b, s=s)
        self._steer_line.push(self.t, math.radians(s / self.p.steer_ratio))
        self._accel_line.push(self.t, t + b)
        return self.last_cmd

    def _dbw(self, a_req: float) -> tuple:
        """Acceleration request -> (throttle, braking).

        One signed pedal effort (positive = throttle, negative = brake) from a
        feedforward map measured on this vehicle plus PI on measured
        acceleration, so small decel requests can still reach the brake.
        """
        err = a_req - self._a_meas
        self._a_int = float(np.clip(self._a_int + STEP * err, -1.5, 1.5))
        coast = COAST_DECEL * min(1.0, self.s.v / 4.0)   # engine braking fades at crawl
        if a_req > -coast:
            effort = 0.12 * (a_req + coast)
        else:
            effort = -(-a_req - coast) / 10.7
        effort += 0.05 * err + 0.25 * self._a_int
        return float(np.clip(effort, 0.0, 1.0)), float(np.clip(-effort, 0.0, 1.0))

    def step(self, duration: float) -> None:
        p = self.p
        n = max(1, int(round(duration / STEP)))
        for _ in range(n):
            target = float(np.clip(self._steer_line.get(self.t) + p.steer_offset, -p.max_steer, p.max_steer))
            rate = max(-p.steer_rate_max, min(p.steer_rate_max, (target - self.delta) / p.steer_tau))
            self.delta += STEP * rate
            a_req = self._accel_line.get(self.t)
            thr, brk = self._dbw(a_req)
            if self.s.v < 0.05 and a_req <= COAST_DECEL:
                thr, brk = 0.0, max(brk, 0.3)   # hold the car at a standstill
            self.inputs.m_steering = float(np.clip(self.delta / self.max_steer, -1.0, 1.0))
            self.inputs.m_throttle = thr
            self.inputs.m_braking = brk
            self.car.Synchronize(self.t, self.inputs, self.terrain)
            self.terrain.Synchronize(self.t)
            self.car.Advance(STEP)
            self.terrain.Advance(STEP)
            self.t += STEP
            v = max(0.0, self.veh.GetSpeed())
            self._a_meas += (STEP / 0.05) * ((v - self._v_prev) / STEP - self._a_meas)
            self._a_comfort += (STEP / 0.15) * (self._a_meas - self._a_comfort)
            self._v_prev = v
            self.s.odometer += STEP * v
        self._update_state()

    def _update_state(self) -> None:
        rl = self.veh.GetSpindlePos(1, veh.LEFT)
        rr = self.veh.GetSpindlePos(1, veh.RIGHT)
        self.s.x = 0.5 * (rl.x + rr.x)
        self.s.y = 0.5 * (rl.y + rr.y)
        self.s.psi = self.veh.GetRot().GetCardanAnglesZYX().z
        self.s.v = max(0.0, self.veh.GetSpeed())
        self.s.delta = self.delta
        self.s.accel = self._a_comfort
        self._pitch = -self.veh.GetRot().GetCardanAnglesZYX().y

    def measure(self) -> CarState:
        p, st, r = self.p, self.s, self.rng
        psi = (st.psi + r.normal(0.0, p.yaw_noise) + math.pi) % (2 * math.pi) - math.pi
        return CarState(x=st.x + r.normal(0.0, p.pos_noise), y=st.y + r.normal(0.0, p.pos_noise),
                        v=max(0.0, st.v + r.normal(0.0, p.v_noise)), psi=psi,
                        pitch=self._pitch + r.normal(0.0, p.pitch_noise), stamp=self.t)

    def curvature(self) -> float:
        return math.tan(self.delta) / self.wheelbase
