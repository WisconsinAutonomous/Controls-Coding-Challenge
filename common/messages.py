"""Plain-Python mirrors of the ROS 2 messages our control stack uses.

On the car these are ROS 2 messages from the ``wauto_control_msgs`` package.
Here they are dataclasses so that the challenges run with nothing but NumPy,
but the field names, units and sign conventions are the same as on the car.

Coordinate conventions (the same everywhere in this repo):

* World frame is local ENU: ``x`` = East [m], ``y`` = North [m].
* Heading ``psi`` / ``yaw`` is ENU: radians, 0 = East, counter-clockwise positive.
  (GPS and our waypoint files use compass heading: 0 = North, clockwise.
  ``psi = pi/2 - compass_heading_rad``.)
* Positive curvature and positive steering turn the car LEFT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CarState:
    """Mirror of ``wauto_control_msgs/CarState`` (topic ``/localization/state``).

    The reported point is the center of the REAR AXLE.
    """

    x: float = 0.0      # [m] East
    y: float = 0.0      # [m] North
    v: float = 0.0      # [m/s] speed along the heading, always >= 0 (no reverse)
    psi: float = 0.0    # [rad] ENU heading, 0 = East, CCW positive
    pitch: float = 0.0  # [rad] nose-up positive (driving uphill => pitch > 0)
    stamp: float = 0.0  # [s] time the state was measured


@dataclass
class TrajectoryPoint:
    """Mirror of ``wauto_control_msgs/TrajectoryPoint``.

    On the car the position is a ``geometry_msgs/Pose``; here it is flattened
    to ``x, y, yaw``.
    """

    x: float
    y: float
    yaw: float                     # [rad] ENU heading of the path at this point
    velocity_mps: float            # [m/s] target speed at this point
    acceleration_mps2: float = 0.0  # [m/s^2] target longitudinal acceleration
    curvature: float = 0.0         # [1/m] path curvature, positive = turning left
    relative_time_sec: float = 0.0  # [s] time to reach this point from point 0


@dataclass
class ReferenceTrajectory:
    """Mirror of ``wauto_control_msgs/ReferenceTrajectory``.

    Published by the mid planner on ``/vehicle/reference_trajectory`` and
    consumed by the MPC.  Points are ordered along the direction of travel and
    are typically ~0.5 m apart.
    """

    points: List[TrajectoryPoint] = field(default_factory=list)
    stamp: float = 0.0     # [s] time the trajectory was published
    reverse: bool = False  # always False in these challenges

    def __len__(self) -> int:
        return len(self.points)

    def as_arrays(self) -> dict:
        """Return the fields as NumPy arrays, keyed by field name."""
        if not self.points:
            keys = ("x", "y", "yaw", "velocity_mps", "acceleration_mps2",
                    "curvature", "relative_time_sec")
            return {k: np.zeros(0) for k in keys}
        return {
            "x": np.array([p.x for p in self.points], dtype=float),
            "y": np.array([p.y for p in self.points], dtype=float),
            "yaw": np.array([p.yaw for p in self.points], dtype=float),
            "velocity_mps": np.array([p.velocity_mps for p in self.points], dtype=float),
            "acceleration_mps2": np.array([p.acceleration_mps2 for p in self.points], dtype=float),
            "curvature": np.array([p.curvature for p in self.points], dtype=float),
            "relative_time_sec": np.array([p.relative_time_sec for p in self.points], dtype=float),
        }

    @classmethod
    def from_arrays(cls, x, y, yaw, velocity_mps, acceleration_mps2=None,
                    curvature=None, relative_time_sec=None,
                    stamp: float = 0.0) -> "ReferenceTrajectory":
        """Build a trajectory from equal-length arrays (missing fields = 0)."""
        n = len(x)
        zeros = np.zeros(n)
        a = zeros if acceleration_mps2 is None else acceleration_mps2
        k = zeros if curvature is None else curvature
        t = zeros if relative_time_sec is None else relative_time_sec
        pts = [TrajectoryPoint(float(x[i]), float(y[i]), float(yaw[i]),
                               float(velocity_mps[i]), float(a[i]),
                               float(k[i]), float(t[i]))
               for i in range(n)]
        return cls(points=pts, stamp=stamp)


@dataclass
class CarTBS:
    """Mirror of ``wauto_control_msgs/CarTBS`` (topic ``/control/tbs``).

    TBS = Throttle, Brake, Steer.  This is what the drive-by-wire system
    accepts.  Throttle and brake are *acceleration requests*, not pedal
    positions, and are sent on separate channels:

    * ``t`` throttle in [0, 5] m/s^2   (0 = no throttle)
    * ``b`` brake    in [-10, 0] m/s^2 (NEGATIVE numbers brake, 0 = no brake)
    * ``s`` steering COLUMN angle in [-500, 500] degrees, positive = left.
      The column angle is the road-wheel angle times the steering ratio
      (see ``common.vehicle.STEERING_RATIO``).

    Values outside these ranges are clipped by the vehicle.
    """

    t: float = 0.0
    b: float = 0.0
    s: float = 0.0


# Detection classes, copied from wauto_perception_msgs/ObjectDetection so the
# numbers match what you would see on the car.
class ObjClass:
    UNKNOWN = 0
    CAR = 1
    PEDESTRIAN = 2
    DEER = 3
    BARRICADE = 4
    TRAFFIC_LIGHT = 5
    TRAFFIC_SIGN = 6
    BARREL = 7
    RAILROAD_GATE = 8
    CONE = 9
    TUBE = 10

    NAMES = {0: "unknown", 1: "car", 2: "pedestrian", 3: "deer", 4: "barricade",
             5: "traffic_light", 6: "traffic_sign", 7: "barrel",
             8: "railroad_gate", 9: "cone", 10: "tube"}


class LightColor:
    """``custom_classification`` values for ObjClass.TRAFFIC_LIGHT."""

    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


@dataclass
class ObjectDetection:
    """Mirror of ``wauto_perception_msgs/ObjectDetection``.

    Positions are in the VEHICLE frame, measured from the center of the FRONT
    BUMPER: ``x`` forward [m], ``y`` left [m], ``z`` up [m].  ``length`` is the
    object's extent along the vehicle x axis and ``width`` along y.  (The car's
    message has no ``length`` field; we added it so boxes are fully defined.)

    ``object_id`` is stable while perception keeps tracking the object, but
    detections can be missed for a frame or two, and positions are noisy.
    """

    object_id: int
    obj_class: int
    custom_classification: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass
class ObjectArray:
    """Mirror of ``wauto_perception_msgs/ObjectArray``."""

    objects: List[ObjectDetection] = field(default_factory=list)
    stamp: float = 0.0


def trajectory_is_finite(traj: Optional[ReferenceTrajectory]) -> bool:
    """True if ``traj`` has at least 2 points and no NaN/inf values."""
    if traj is None or len(traj.points) < 2:
        return False
    arr = traj.as_arrays()
    return all(np.all(np.isfinite(v)) for v in arr.values())
