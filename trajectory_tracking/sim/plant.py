"""NumPy vehicle plant: what your CarTBS commands actually do to the car.

The motion model is a kinematic bicycle about the rear axle.  What makes it
behave like our real car (and unlike a textbook bicycle) is the actuator and
sensor layer around it:

* Steering: CarTBS.s is a steering COLUMN angle.  The road-wheel angle follows
  ``column / STEERING_RATIO`` after a transport delay, through a first-order
  lag, and cannot move faster than a rate limit.  It also saturates at
  ``MAX_STEER``.
* Throttle/brake: acceleration requests, also delayed and lagged.  Rolling
  resistance, aerodynamic drag and road grade act on top.  The car never rolls
  backward (automatic hill hold).
* Understeer: at speed the car turns a little less than the kinematic model
  says for the same wheel angle.
* Localization: CarState has small Gaussian noise, like our GPS/INS.

The nominal numbers below are what we believe the car does; the grader also
runs a set of perturbed plants (see ``PERTURBATION_RANGES``) because on the
real car those numbers are only approximately known.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from common import vehicle as V
from common.messages import CarState, CarTBS


@dataclass
class PlantParams:
    wheelbase: float = V.WHEELBASE
    max_steer: float = V.MAX_STEER            # [rad] road wheel
    steer_ratio: float = V.STEERING_RATIO
    steer_tau: float = 0.20                   # [s] first-order steering lag
    steer_rate_max: float = 0.35              # [rad/s] road-wheel rate limit
    steer_delay: float = 0.08                 # [s] command -> actuator delay
    steer_offset: float = 0.0                 # [rad] alignment bias at the road wheel
    accel_tau: float = 0.30                   # [s] powertrain / brake lag
    accel_delay: float = 0.10                 # [s]
    throttle_gain: float = 1.0                # delivered / requested (throttle)
    brake_gain: float = 1.0                   # delivered / requested (brake)
    rolling_decel: float = 0.15               # [m/s^2]
    drag_coeff: float = 2.6e-4                # [1/m]  drag decel = c * v^2
    understeer: float = 0.0035                # [rad per m/s^2 of lateral accel]
    pos_noise: float = 0.02                   # [m] std dev on x, y
    yaw_noise: float = 0.003                  # [rad] std dev on psi
    v_noise: float = 0.05                     # [m/s] std dev on v
    pitch_noise: float = 0.002                # [rad] std dev on pitch


NOMINAL = PlantParams()

# Ranges used by `run.py --perturb`.  Each perturbed plant draws every value
# uniformly from its range.
PERTURBATION_RANGES = {
    "steer_tau": (0.14, 0.30),
    "steer_rate_max": (0.28, 0.45),
    "steer_delay": (0.04, 0.14),
    "steer_offset": (-0.01, 0.01),
    "accel_tau": (0.20, 0.45),
    "accel_delay": (0.05, 0.15),
    "throttle_gain": (0.85, 1.10),
    "brake_gain": (0.85, 1.15),
    "rolling_decel": (0.10, 0.25),
    "understeer": (0.0, 0.006),
}


def perturbed_params(rng: np.random.Generator, base: PlantParams = NOMINAL) -> PlantParams:
    vals = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PERTURBATION_RANGES.items()}
    return replace(base, **vals)


class _DelayLine:
    """Returns the command that was issued ``delay`` seconds ago."""

    def __init__(self, delay: float, initial: float):
        self.delay = delay
        self.buf = deque([(-1e9, initial)])

    def push(self, t: float, value: float) -> None:
        self.buf.append((t + self.delay, value))

    def get(self, t: float) -> float:
        while len(self.buf) > 1 and self.buf[1][0] <= t + 1e-9:
            self.buf.popleft()
        return self.buf[0][1]


@dataclass
class TrueState:
    x: float
    y: float
    psi: float
    v: float
    delta: float = 0.0      # actual road-wheel angle [rad]
    accel: float = 0.0      # actual powertrain/brake acceleration [m/s^2]
    odometer: float = 0.0   # distance driven [m]


class Plant:
    """Integrates the car in ``dt`` (2 ms) substeps between control ticks."""

    def __init__(self, x0: TrueState, params: PlantParams = NOMINAL,
                 grade_fn: Optional[Callable[[float], float]] = None,
                 seed: int = 0, dt: float = 0.002):
        self.p = params
        self.s = replace(x0)
        self.t = 0.0
        self.dt = dt
        self.grade_fn = grade_fn or (lambda odo: 0.0)
        self.rng = np.random.default_rng(seed)
        self._steer_line = _DelayLine(params.steer_delay, x0.delta - params.steer_offset)
        self._accel_line = _DelayLine(params.accel_delay, 0.0)
        self.last_cmd = CarTBS()

    # ------------------------------------------------------------------ #
    def apply(self, cmd: CarTBS) -> CarTBS:
        """Clip a command to the CarTBS limits and queue it. Returns the clipped command."""
        t = float(np.clip(_finite(cmd.t), 0.0, V.MAX_THROTTLE))
        b = float(np.clip(_finite(cmd.b), -V.MAX_BRAKE, 0.0))
        s = float(np.clip(_finite(cmd.s), -V.MAX_STEER_COLUMN_DEG, V.MAX_STEER_COLUMN_DEG))
        clipped = CarTBS(t=t, b=b, s=s)
        self.last_cmd = clipped
        delta_cmd = math.radians(s / self.p.steer_ratio)
        self._steer_line.push(self.t, delta_cmd)
        self._accel_line.push(self.t, self.p.throttle_gain * t + self.p.brake_gain * b)
        return clipped

    def step(self, duration: float) -> None:
        """Advance the true state by ``duration`` seconds."""
        n = max(1, int(round(duration / self.dt)))
        h = duration / n
        p, st = self.p, self.s
        for _ in range(n):
            delta_target = float(np.clip(self._steer_line.get(self.t) + p.steer_offset,
                                         -p.max_steer, p.max_steer))
            rate = (delta_target - st.delta) / p.steer_tau
            rate = max(-p.steer_rate_max, min(p.steer_rate_max, rate))
            st.delta = max(-p.max_steer, min(p.max_steer, st.delta + h * rate))

            a_req = self._accel_line.get(self.t)
            st.accel += h * (a_req - st.accel) / p.accel_tau

            grade = self.grade_fn(st.odometer)
            g_along = V.GRAVITY * math.sin(math.atan(grade))
            resist = p.rolling_decel + p.drag_coeff * st.v * st.v
            if st.v <= 1e-3:
                # Standing still: static friction / brakes / hill-hold keep the
                # car in place unless the powertrain beats resistance + grade.
                drive = st.accel - g_along
                a_net = drive - p.rolling_decel if drive > p.rolling_decel else 0.0
            else:
                a_net = st.accel - resist - g_along

            v_new = max(0.0, st.v + h * a_net)
            v_mid = 0.5 * (st.v + v_new)
            kappa = math.tan(st.delta) / (p.wheelbase + p.understeer * v_mid * v_mid)
            psi_mid = st.psi + 0.5 * h * v_mid * kappa
            st.x += h * v_mid * math.cos(psi_mid)
            st.y += h * v_mid * math.sin(psi_mid)
            st.psi += h * v_mid * kappa
            st.odometer += h * v_mid
            st.v = v_new
            self.t += h

    # ------------------------------------------------------------------ #
    def measure(self) -> CarState:
        """Noisy localization output, like /localization/state."""
        p, st, r = self.p, self.s, self.rng
        psi = (st.psi + r.normal(0.0, p.yaw_noise) + math.pi) % (2 * math.pi) - math.pi
        return CarState(
            x=st.x + r.normal(0.0, p.pos_noise),
            y=st.y + r.normal(0.0, p.pos_noise),
            v=max(0.0, st.v + r.normal(0.0, p.v_noise)),
            psi=psi,
            pitch=math.atan(self.grade_fn(st.odometer)) + r.normal(0.0, p.pitch_noise),
            stamp=self.t,
        )

    def curvature(self) -> float:
        """Path curvature the car is currently driving [1/m]."""
        v = self.s.v
        return math.tan(self.s.delta) / (self.p.wheelbase + self.p.understeer * v * v)


def _finite(v: float) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0
