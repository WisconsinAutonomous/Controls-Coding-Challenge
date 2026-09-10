"""The car, driving straight down a road: what your throttle and brake do.

Same longitudinal model as our Trajectory Tracking challenge (and our best
estimate of the real car):

* Your throttle/brake request reaches the car 0.10 s later (transport delay),
  then the actual acceleration follows it with a 0.30 s lag.
* Rolling resistance (0.15 m/s^2) and a little aerodynamic drag slow you down.
* On a hill, gravity pulls you back: g * sin(pitch).
* The car never rolls backward (automatic hill hold).
* The speed you measure has a little noise (0.05 m/s).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from common import vehicle as V
from common.messages import CarState, CarTBS

DT_SIM = 0.002    # [s] internal integration step


@dataclass
class CarParams:
    accel_delay: float = 0.10      # [s]
    accel_tau: float = 0.30        # [s]
    throttle_gain: float = 1.0     # delivered / requested
    brake_gain: float = 1.0
    rolling_decel: float = 0.15    # [m/s^2]
    drag_coeff: float = 2.6e-4     # [1/m], drag decel = c * v^2
    v_noise: float = 0.05          # [m/s]
    pitch_noise: float = 0.002     # [rad]


NOMINAL = CarParams()

# Used by `run.py --perturb N`: every value drawn uniformly from its range.
PERTURBATION_RANGES = {
    "accel_delay": (0.05, 0.15),
    "accel_tau": (0.20, 0.45),
    "throttle_gain": (0.80, 1.10),
    "brake_gain": (0.85, 1.15),
    "rolling_decel": (0.10, 0.25),
}


def perturbed(rng: np.random.Generator, base: CarParams = NOMINAL) -> CarParams:
    return replace(base, **{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PERTURBATION_RANGES.items()})


class Car:
    def __init__(self, v0: float = 0.0, params: CarParams = NOMINAL,
                 grade: Optional[Callable[[float], float]] = None, seed: int = 0):
        self.p = params
        self.x = 0.0          # distance along the road [m] (rear axle)
        self.v = v0           # true speed [m/s]
        self.accel = 0.0      # actual powertrain/brake acceleration [m/s^2]
        self.t = 0.0
        self.grade = grade or (lambda x: 0.0)
        self.rng = np.random.default_rng(seed)
        self._queue = deque([(-1e9, 0.0)])   # (time it takes effect, requested accel)

    def apply(self, cmd: CarTBS) -> CarTBS:
        t = float(np.clip(_finite(cmd.t), 0.0, V.MAX_THROTTLE))
        b = float(np.clip(_finite(cmd.b), -V.MAX_BRAKE, 0.0))
        self._queue.append((self.t + self.p.accel_delay, self.p.throttle_gain * t + self.p.brake_gain * b))
        return CarTBS(t=t, b=b, s=0.0)

    def _requested(self) -> float:
        q = self._queue
        while len(q) > 1 and q[1][0] <= self.t + 1e-9:
            q.popleft()
        return q[0][1]

    def step(self, duration: float) -> None:
        p = self.p
        n = max(1, int(round(duration / DT_SIM)))
        h = duration / n
        for _ in range(n):
            self.accel += h * (self._requested() - self.accel) / p.accel_tau
            g_along = V.GRAVITY * math.sin(math.atan(self.grade(self.x)))
            if self.v <= 1e-3:
                drive = self.accel - g_along
                a_net = drive - p.rolling_decel if drive > p.rolling_decel else 0.0
            else:
                a_net = self.accel - p.rolling_decel - p.drag_coeff * self.v ** 2 - g_along
            v_new = max(0.0, self.v + h * a_net)
            self.x += h * 0.5 * (self.v + v_new)
            self.v = v_new
            self.t += h

    def pitch(self) -> float:
        return math.atan(self.grade(self.x))

    def measure(self) -> CarState:
        """What localization reports.  The road points East, so y = 0 and psi = 0."""
        return CarState(x=self.x, y=0.0, v=max(0.0, self.v + self.rng.normal(0.0, self.p.v_noise)),
                        psi=0.0, pitch=self.pitch() + self.rng.normal(0.0, self.p.pitch_noise),
                        stamp=self.t)


def _finite(v) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0
