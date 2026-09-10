"""Your trajectory-tracking controller.  This is the file you edit.

Work through the parts in order (see README.md).  After each part run

    python check.py part1      (or part2, part3)

and then `python run.py` to drive the scenarios.

The simulator calls ``Controller.compute(state, ref)`` 50 times a second:

* ``state`` is a CarState from localization: ``x, y`` of the REAR AXLE (ENU:
  x = East, y = North, meters), speed ``v`` [m/s], heading ``psi`` [rad,
  0 = East, counter-clockwise positive] and ``pitch``.  It is a little noisy.
* ``ref`` is the latest ReferenceTrajectory from the planner: points every
  0.5 m from about where the car is to 50 m ahead.  ``ref.as_arrays()`` gives
  NumPy arrays ``x, y, yaw, velocity_mps, acceleration_mps2, curvature``.
  When the planner wants you to stop, the trajectory ENDS at the stop point
  with speed 0.

You return a CarTBS: throttle ``t`` (0 to 5 m/s^2), brake ``b`` (-10 to 0
m/s^2) and steering COLUMN angle ``s`` in degrees (positive = left).  The
glue at the bottom already turns an acceleration and a road-wheel angle into
a CarTBS for you.
"""

from __future__ import annotations

import math

import numpy as np

from common import vehicle
from common.geometry import wrap_to_pi
from common.messages import CarState, CarTBS, ReferenceTrajectory

DT = 0.02                     # [s] time between calls
WHEELBASE = vehicle.WHEELBASE  # [m] 2.67


class Controller:
    def __init__(self):
        # Tuning knobs.  Add your own (and any state you need) here.
        self.kp_speed = 1.0          # Part 2: (m/s^2) per (m/s) of speed error
        self.ki_speed = 0.3          # Part 2
        self.speed_integral = 0.0    # Part 2
        self.lookahead_min = 3.0     # Part 3: [m]
        self.lookahead_gain = 0.8    # Part 3: [s], lookahead = min + gain * speed

    # =========================================================================
    # Part 1: where am I relative to the path?
    # =========================================================================
    def path_errors(self, state: CarState, pts: dict):
        """Find the closest reference point and your errors relative to the path.

        Return ``(i, e_lat, e_yaw)``:
          * ``i``: index of the reference point closest to the car
          * ``e_lat``: lateral error [m], the signed distance from the path to
            the car, POSITIVE when the car is LEFT of the path
          * ``e_yaw``: heading error [rad], ``car heading - path heading at i``,
            wrapped to [-pi, pi) (use ``wrap_to_pi``)

        Hint: the path direction at point i is (cos(yaw_i), sin(yaw_i)), and
        "left of it" is (-sin(yaw_i), cos(yaw_i)).  Project the vector from
        the path point to the car onto "left".
        """
        # TODO (Part 1)
        raise NotImplementedError("Part 1: path_errors() is not written yet")

    # =========================================================================
    # Part 2: how hard to accelerate?
    # =========================================================================
    def speed_accel(self, state: CarState, pts: dict, i: int) -> float:
        """Return an acceleration request [m/s^2] that tracks the reference speed.

        The target is ``pts["velocity_mps"]`` near point i.  Use a PI
        controller on the speed error, plus the reference acceleration
        ``pts["acceleration_mps2"]`` as feedforward (the plan is telling you
        how hard it wants to speed up or slow down).

        Two things to watch:
          * Starting from rest, the speed at your own position is ~0 (the
            plan ramps up ahead of you).  Look a few points ahead, or you will
            never get going.
          * Limit the integral (windup), and do not let it grow while you are
            stopped and waiting.
        If you did the Speed Control challenge, this is the same idea.
        """
        # TODO (Part 2)
        raise NotImplementedError("Part 2: speed_accel() is not written yet")

    # =========================================================================
    # Part 3: which way to steer? (pure pursuit)
    # =========================================================================
    def pure_pursuit(self, state: CarState, pts: dict, i: int) -> float:
        """Return a road-wheel angle [rad], positive = left.

        Pure pursuit: pick a "goal point" on the path a lookahead distance
        ``ld`` ahead of the car, then steer along the circular arc that
        passes through the rear axle and that point.

            1. ld = lookahead_min + lookahead_gain * speed
            2. starting at i, walk along the path until a point is at least
               ld away from the car: that is the goal point
            3. express the goal point in the CAR's frame (x forward, y left):
               alpha = angle to it, measured from the car's heading
            4. the arc through it has curvature  k = 2 * sin(alpha) / D,
               where D is the actual distance to the goal point (a little
               more than ld, because the path points are 0.5 m apart)
            5. the road-wheel angle for that curvature is  atan(WHEELBASE * k)

        README.md has a picture.
        """
        # TODO (Part 3)
        raise NotImplementedError("Part 3: pure_pursuit() is not written yet")

    # =========================================================================
    # The glue (already written).  Change it as much as you like later.
    # =========================================================================
    def compute(self, state: CarState, ref: ReferenceTrajectory) -> CarTBS:
        pts = ref.as_arrays()
        i, e_lat, e_yaw = self.path_errors(state, pts)                    # Part 1
        accel = _if_done(self.speed_accel, state, pts, i,                  # Part 2
                         default=1.0 * (pts["velocity_mps"][min(i + 4, len(pts["x"]) - 1)] - state.v))
        delta = _if_done(self.pure_pursuit, state, pts, i,                 # Part 3
                         default=-0.3 * e_lat - 1.0 * e_yaw)
        delta = float(np.clip(delta, -vehicle.MAX_STEER, vehicle.MAX_STEER))
        return CarTBS(t=min(max(accel, 0.0), 5.0), b=max(min(accel, 0.0), -10.0),
                      s=vehicle.wheel_angle_to_column_deg(delta))


def _if_done(fn, *args, default=None):
    """Call a TODO function; until it is written, use a crude stand-in."""
    try:
        return fn(*args)
    except NotImplementedError:
        return default
