"""Local Planning: your planner.  This is the file you edit.

There are three parts for you to write, marked PART 1, PART 2 and PART 3.
Everything else is glue that is already written.  Until a part is done the
glue falls back to something naive, so ``python run.py`` always works.

    python check.py          # checks your parts one by one, with hints
    python run.py            # drives the car through the scenarios

Conventions (README.md has pictures):
  * Frenet: s = distance along your lane's centerline [m], d = sideways offset
    from it [m], positive to the LEFT.
  * The car's position is its REAR AXLE.  The front bumper is FRONT = 3.7 m
    ahead of it, the rear bumper REAR = 0.8 m behind it, and it is 1.8 m wide.
"""

from __future__ import annotations

import math

import numpy as np

from common import vehicle as V
from tools import (ObstacleMemory, build_trajectory, lateral_transition,  # noqa: F401
                   path_curvature, stations, transition_length)

A_LAT_MAX = 1.5      # [m/s^2] most sideways acceleration you want in a curve
ACCEL = 1.5          # [m/s^2] comfortable speeding up
DECEL = 2.0          # [m/s^2] comfortable slowing down
CLEARANCE = 0.6      # [m] gap to leave next to an obstacle (the grader wants 0.5 or more)
STOP_HOLD_S = 3.0    # [s] how long to stay stopped at a stop sign
HORIZON = 50.0       # [m] how far ahead we plan
FRONT = V.REAR_AXLE_TO_FRONT_BUMPER    # 3.7 m
REAR = V.REAR_AXLE_TO_REAR_BUMPER      # 0.8 m
HALF_WIDTH = 0.5 * V.WIDTH             # 0.9 m


# ====================================================================== #
# PART 1: the speed profile
# ====================================================================== #
def speed_profile(s, v_limit, curvature, v0, stop_s=None):
    r"""How fast to go at each station.

    Inputs (arrays all have the same length N, one value per station):
      s          stations along the route [m], about 0.5 m apart. s[0] is where the car is now.
      v_limit    the speed limit at each station [m/s]
      curvature  how sharply the path bends at each station [1/m] (that is 1 / turn radius)
      v0         the car's speed right now [m/s]
      stop_s     where the car must be stopped (a stop line, the goal), or None.  When it
                 is not None it lies within s (usually it is the last station).
    Return: a numpy array of N speeds [m/s].

    Your speeds must follow these rules (check.py tests every one):
      1. v[0] == v0.  The car cannot change speed instantly.
      2. Never above v_limit, and never so fast in a curve that the sideways acceleration
         v**2 * |curvature| goes above A_LAT_MAX.  In other words
         v <= sqrt(A_LAT_MAX / |curvature|)   (careful when curvature is 0).
      3. From one station to the next, speed up by at most ACCEL and slow down by at most
         DECEL.  With constant acceleration a over a distance ds:  v_next**2 = v**2 + 2*a*ds.
      4. v == 0 at stop_s and at every station after it.
      5. Otherwise go as fast as rules 1 to 4 allow.

    One way to do it:
      a. Make an array of the most you could ever go at each station (rules 2 and 4).
      b. FORWARD pass (rule 3, speeding up): walk from the car forward.  Each station can be
         at most as fast as the one before it plus what ACCEL gives you over ds.
      c. BACKWARD pass (rule 3, slowing down): walk from the far end back toward the car.
         Each station can be at most as fast as still lets you slow down to the next one
         with DECEL.  This is how the car "sees" a stop or a curve ahead and brakes early.
      d. Take the smaller of the two at every station, and put v0 back at v[0].

          speed
            |   __________              __________        forward pass: the / ramps
            |  /          \____________/          \       backward pass: the \ ramps
            | /             the curve              \
            |/______________________________________\ stop    s -->

    Plain Python loops over the stations are fine (N is about 100).
    """
    raise NotImplementedError("Part 1: speed_profile")


# ====================================================================== #
# PART 2: stop signs
# ====================================================================== #
class StopSignLogic:
    """Stop at stop signs.

    ``route_map.stop_lines`` lists every stop line on the route.  Lines with
    ``kind == 'stop_sign'`` need a FULL stop: the car stopped (speed below
    0.05 m/s) for STOP_HOLD_S = 3 seconds with its FRONT BUMPER no more than
    4 m before the line.  Only then may it drive across.

    This is a tiny state machine, one pass per stop sign:

        APPROACHING ──(stopped close to the line)──► HOLDING ──(3 s)──► DONE
             ▲                                          │
             └────────────(started moving again)────────┘

    ``update`` is called every tick (0.1 s).  Return the s where the REAR AXLE
    should stop, or None when no stop sign needs you to stop.

    Inputs:
      t          time [s]
      s_car      where the car is along the route (its rear axle) [m]
      v          the car's speed [m/s]
      route_map  the map. Each stop line has .id, .s (where the line is) and .kind.

    Hints:
      * Aim to stop with the front bumper about 1 m before the line:
        stop_s = line.s - 1.0 - FRONT.
      * Remember which signs are DONE (for example a set of ids), so you do not stop twice.
      * Only count time while you are stopped close to the line.  Stopping 10 m early
        (maybe something was in the way) does not count.
      * Ignore lines behind you and lines of other kinds ('traffic_light' and
        'crosswalk' are stretch goals).
    """

    def __init__(self):
        pass   # TODO: add whatever state you need

    def update(self, t, s_car, v, route_map):
        raise NotImplementedError("Part 2: StopSignLogic.update")


# ====================================================================== #
# PART 3: nudging around obstacles
# ====================================================================== #
def choose_offset(s, obstacles, route_map):
    r"""Sideways offsets that steer around things sticking into your lane.

    Inputs:
      s          stations along the route [m]
      obstacles  what ObstacleMemory remembers: a list of TrackedObject, each with
                 s_min, s_max (its extent along the road), d_min, d_max (across it),
                 obj_class and is_pedestrian
      route_map  the map (route_map.lane_width is 3.6 m)
    Return: a numpy array d, one offset [m] per station (positive = left).

    In the Frenet frame (s along the road, d sideways) a nudge looks like this:

         d
        +1.8  ═══════════════════════════════════════   double yellow (do not cross)
                              ______________
         0.0  ───────────────/              \────────   your d(s)
                            [barrel]
        -1.8  ───────────────────────────────────────   edge line
                                                    s ──►

    Rules (check.py tests each):
      1. Stay in your lane: the car is 1.8 m wide, so its center has to stay within
         |d| <= 1.8 - 0.9 = 0.9 m.  Keep about 0.2 m to spare, so |d| <= 0.7: while
         the car is angled mid-shift its front corner swings further out than its
         center, and perception's idea of where a barrel is wobbles a little.
      2. While any part of the car is beside an obstacle, keep at least CLEARANCE between
         them.  At station s the car's body covers s - REAR to s + FRONT along the road,
         and d - 0.9 to d + 0.9 across it.
      3. Ignore obstacles that do not reach into your lane (d_max < -1.8 or d_min > 1.8),
         and ignore pedestrians (those are a stretch goal).
      4. Move smoothly.  Use lateral_transition() to shift over and back, and make each
         shift long enough: transition_length() tells you how long at a given speed.
         For the speed, use the speed limit there: route_map.speed_limit_at(ob.s).
      5. Come back to d = 0 after the obstacle.

    Hints:
      * An obstacle on your right (d < 0) is passed on the left, at about
        d = d_max + CLEARANCE + 0.9.  One on your left: d = d_min - CLEARANCE - 0.9.
      * Be fully shifted by the time the front of the car reaches s_min, and stay shifted
        until the rear of the car is past s_max.  Add a meter or two of buffer: the
        positions come from noisy perception.
      * Tie the shift to the obstacle's s, not to the car, so the plan does not slide
        forward every tick.
      * Adding two lateral_transition() calls makes one nudge (see its docstring).
    """
    raise NotImplementedError("Part 3: choose_offset")


# ====================================================================== #
# Glue (already written).  You can change it, but you should not need to.
# ====================================================================== #
FALLBACK_SPEED = 6.0     # [m/s] used while Part 1 is not done


class PartError(ValueError):
    """One of your parts returned something the planner cannot use."""


def _as_array(name, value, n):
    """Check what a part returned: n finite numbers."""
    if value is None:
        raise PartError(f"{name} returned None (did you forget `return`?)")
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise PartError(f"{name} returned {type(value).__name__}; return an array of {n} numbers")
    if arr.shape != (n,):
        raise PartError(f"{name} returned shape {arr.shape}; expected {n} values, one per station")
    if not np.all(np.isfinite(arr)):
        raise PartError(f"{name} returned NaN or inf at s index {int(np.argmin(np.isfinite(arr)))}")
    return arr


def _fallback_speed(s, v0, stop_s):
    """Naive speeds used until Part 1 works: a fixed cap, and stop at stop_s."""
    ahead = np.maximum(s - s[0], 0.0)
    v = np.minimum(FALLBACK_SPEED, np.sqrt(v0 ** 2 + 2 * ACCEL * ahead))
    if stop_s is not None:
        v = np.minimum(v, np.sqrt(2 * DECEL * np.maximum(stop_s - s, 0.0)))
    v[0] = v0
    return v


class Planner:
    def __init__(self, route_map):
        self.route = route_map
        self.memory = ObstacleMemory()
        self.stop_signs = StopSignLogic()
        self.missing = set()   # parts that are not done yet (run.py reports them)
        self.s_hint = None

    def plan(self, obs):
        ego = obs.ego
        ref = self.route.reference
        s0, d0, _ = ref.project(ego.x, ego.y, s_hint=self.s_hint)
        self.s_hint = s0

        # What is around us, remembered in world and Frenet coordinates.
        obstacles = self.memory.update(obs, self.route)
        # for ob in obstacles:
        #     print(ob.name, ob.s_min, ob.s_max, ob.d_min, ob.d_max, "seen" if ob.seen else "remembered")

        # Where must we stop?  The goal, and (Part 2) stop signs.
        stop_s = self.route.goal_s
        try:
            sign = self.stop_signs.update(obs.t, s0, ego.v, self.route)
            if sign is not None:
                if not isinstance(sign, (int, float, np.floating)) or not math.isfinite(sign):
                    raise PartError(f"StopSignLogic.update returned {sign!r}; return a number or None")
                stop_s = min(stop_s, float(sign))
        except NotImplementedError:
            self.missing.add("Part 2 (StopSignLogic)")

        s = stations(s0, HORIZON, stop_s)
        stop_here = stop_s if stop_s <= s[-1] + 1e-6 else None   # stations() ends exactly on it

        # Where in the lane do we drive?  (Part 3)
        try:
            d = _as_array("choose_offset", choose_offset(s, obstacles, self.route), len(s))
        except NotImplementedError:
            self.missing.add("Part 3 (choose_offset)")
            d = np.zeros_like(s)

        # How fast?  (Part 1)
        v_limit = np.array([self.route.speed_limit_at(si) for si in s])
        curvature = path_curvature(self.route, s, d)
        try:
            v = _as_array("speed_profile", speed_profile(s, v_limit, curvature, ego.v, stop_here), len(s))
        except NotImplementedError:
            self.missing.add("Part 1 (speed_profile)")
            v = _fallback_speed(s, ego.v, stop_here)

        return build_trajectory(self.route, s, d, v, ego)
