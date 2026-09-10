"""Vehicle parameters for our car (a Chevy Bolt-class hatchback).

These are the same numbers the real planner and MPC are configured with.
"""

import math

# --- Geometry ---------------------------------------------------------------
WHEELBASE = 2.67                   # [m] rear axle to front axle
LENGTH = 4.5                       # [m] bumper to bumper
WIDTH = 1.8                        # [m] mirror to mirror
REAR_AXLE_TO_FRONT_BUMPER = 3.7    # [m]
REAR_AXLE_TO_REAR_BUMPER = LENGTH - REAR_AXLE_TO_FRONT_BUMPER  # 0.8 m

# --- Steering ---------------------------------------------------------------
MAX_STEER = 0.52                   # [rad] max road-wheel angle (~30 deg)
STEERING_RATIO = 16.8              # column angle / road-wheel angle
MAX_STEER_COLUMN_DEG = 500.0       # [deg] CarTBS.s limit
# Tightest turn the car can make: curvature = tan(MAX_STEER) / WHEELBASE
MAX_CURVATURE = math.tan(MAX_STEER) / WHEELBASE   # ~0.214 1/m (radius ~4.7 m)

# --- Longitudinal -----------------------------------------------------------
MAX_THROTTLE = 5.0                 # [m/s^2] CarTBS.t upper limit
MAX_BRAKE = 10.0                   # [m/s^2] CarTBS.b lower limit is -MAX_BRAKE
GRAVITY = 9.81                     # [m/s^2]


def wheel_angle_to_column_deg(delta_rad: float) -> float:
    """Road-wheel angle [rad] -> steering column angle [deg] (CarTBS.s)."""
    return math.degrees(delta_rad) * STEERING_RATIO


def column_deg_to_wheel_angle(column_deg: float) -> float:
    """Steering column angle [deg] (CarTBS.s) -> road-wheel angle [rad]."""
    return math.radians(column_deg / STEERING_RATIO)
