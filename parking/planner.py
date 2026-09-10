"""Your parking planner.  Work through Parts 1 to 3 below (see the README).

Run ``python check.py`` to test each part, and ``python run.py`` to run the
scenarios.  Until a part is done the glue at the bottom falls back to a plain
Dubins path, so everything always runs.

Conventions: a pose is ``(x, y, yaw)``, the REAR AXLE in meters (x = East,
y = North) and ``yaw`` the direction the car's nose points (radians, 0 = East,
counter-clockwise positive).  The car only drives forward in Parts 1 to 3;
backing up is the stretch part.
"""

from __future__ import annotations

import math

import numpy as np

from common import vehicle
from tools import dubins_path, dubins_paths, get_checker, join, staging_candidates, to_segments  # noqa: F401

MAX_CURVATURE = vehicle.MAX_CURVATURE     # 0.214 1/m: the tightest the car can turn


# ====================================================================== #
# Part 1: how the car moves
# ====================================================================== #
def drive(pose, curvature, distance, step=0.1):
    """Poses of the rear axle while driving with the steering wheel held still.

    THE IDEA.  With the steering held still, the rear axle drives along a
    circle (or a straight line when the wheels are straight).  ``curvature``
    is 1 / (radius of that circle): 0 means straight, positive means turning
    LEFT, negative means turning right.  Every meter the car drives, its
    heading changes by ``curvature`` radians.  The car cannot turn tighter
    than MAX_CURVATURE (a circle of about 4.7 m radius).

    For a small distance ds driven forward:
        x   changes by  ds * cos(yaw)
        y   changes by  ds * sin(yaw)
        yaw changes by  ds * curvature

    Backing up (negative ``distance``) uses the same three rules with ds < 0:
    the car moves opposite to where it points, and with the wheels turned
    left the nose swings to the RIGHT.

    INPUTS
        pose       (x, y, yaw) where you start.  Also accept anything with
                   .x .y .yaw attributes (like problem.start).
        curvature  1/m, must satisfy |curvature| <= MAX_CURVATURE, otherwise
                   raise ValueError (the car cannot do it).
        distance   meters along the path; negative means drive backward.
        step       spacing between returned poses, meters.

    RETURNS
        Three numpy arrays (x, y, yaw) of the same length.  The first pose
        is the start pose, the last is where you end up, and consecutive
        poses are at most ``step`` apart.

    HINTS
        * np.linspace(0, distance, n + 1) gives n equal steps; pick
          n = ceil(|distance| / step).
        * Either integrate the three rules above exactly (a little calculus:
          sin and cos show up), or take many tiny steps (1 cm, say) and keep
          every few poses.  The checks allow 2 cm and 0.5 degrees of error.
        * Handle curvature == 0 separately if your formula divides by it.
    """
    raise NotImplementedError("Part 1: drive() is not written yet")


# ====================================================================== #
# Part 2: is a path safe?
# ====================================================================== #
def path_is_free(problem, x, y, yaw):
    """True if the car can drive this path without touching anything.

    THE IDEA.  The car is a 4.5 m x 1.8 m rectangle, not a point.  Its rear
    axle can be a meter away from a parked car while its front corner is
    already scraping it.  So check the whole footprint, at every pose of the
    path.  Leaving the lot counts as a collision too.

    INPUTS
        problem    the ParkingProblem.
        x, y, yaw  numpy arrays: the poses along the path.

    RETURNS
        True if no pose collides, else False.

    HINTS
        * ``get_checker(problem)`` gives you a FastChecker for this lot.
          ``checker.collides(x, y, yaw)`` returns one True/False per pose,
          and ``checker.path_is_free(x, y, yaw)`` also checks between your
          poses (every 2.5 cm) so a corner cannot slip through a gap.
        * ``problem.collides_many(x, y, yaw)`` checks poses exactly but slower.
        * The grader checks every 5 cm along your path.  If you check fewer
          poses than that, you can miss a scrape the grader will see.
    """
    raise NotImplementedError("Part 2: path_is_free() is not written yet")


def min_clearance(problem, x, y, yaw):
    """Smallest distance [m] between the car and anything, along the path.

    THE IDEA.  A path that misses a parked car by 2 cm is legal but scary
    (and scores worse).  This tells you how close the path gets.

    INPUTS / RETURNS
        Same poses as ``path_is_free``; return one float (0 if it touches).

    HINTS
        * ``get_checker(problem).clearance(x, y, yaw)`` gives a distance per
          pose (0 where the car touches something, otherwise within about
          0.1 m); take the smallest.
        * ``problem.clearance_many(x, y, yaw)`` is exact but slower.
    """
    raise NotImplementedError("Part 2: min_clearance() is not written yet")


# ====================================================================== #
# Part 3: find a path into the spot
# ====================================================================== #
def plan_path(problem):
    """A forward-only, collision-free path from problem.start into the spot.

    THE IDEA.  A Dubins path is the shortest way for a car that only drives
    forward to get from one pose to another: a turn, a straight (or a turn),
    and a turn.  ``dubins_path(start, goal)`` computes one for you, but it
    ignores obstacles.  When the direct path hits something, go somewhere
    else first: pick a "staging" pose in the open aisle, drive there, then
    drive into the spot.  Try many staging poses and keep the best path.

    STEPS
        1. goal = problem.spot.goal_pose()  (parks you exactly centered).
        2. Try the direct Dubins path from problem.start to goal.  If your
           path_is_free() says it is clear, that is a good answer.
        3. Otherwise loop over staging_candidates(problem): for each
           staging pose, make a Dubins path start -> staging and another
           staging -> goal.  If both are free, join() them into one path.
        4. Keep the shortest free path you found and return it.

    INPUTS
        problem    the ParkingProblem.

    RETURNS
        One continuous path from problem.start to the goal pose: either three
        numpy arrays (x, y, yaw), or the Path that ``join(p1, p2)`` gives you
        (the glue below accepts both).

    HINTS
        * ``path.length`` tells you how long a Path is.
        * ``dubins_paths(a, b)`` (plural) lists every Dubins path from a to
          b, shortest first.  When the shortest hits something, a longer one
          might not.
        * Shorter scores better, but so does clearance: a path that is 1 m
          longer and stays 0.3 m from everything beats one that scrapes by.
        * If nothing works, raise an error that says so.  It is easier to
          debug than a path through a parked car.
    """
    raise NotImplementedError("Part 3: plan_path() is not written yet")


# ====================================================================== #
# Glue (already written): turns your path into what the grader expects.
# ====================================================================== #
FALLBACK_USED = set()    # why the plain Dubins path was used (run.py prints these)


def _as_xyz(result):
    """plan_path() may return (x, y, yaw) arrays or a Path from the toolbox."""
    if all(hasattr(result, a) for a in ("x", "y", "yaw")):
        return result.x, result.y, result.yaw
    if isinstance(result, (tuple, list)) and len(result) == 3:
        return result
    raise TypeError("plan_path() must return three arrays (x, y, yaw) or a Path from join(); "
                    f"it returned {type(result).__name__}")


class Planner:
    def plan(self, problem):
        try:
            x, y, yaw = _as_xyz(plan_path(problem))
        except NotImplementedError as e:
            # Something plan_path() needs is not written yet (it could be
            # Part 2 or Part 3): use the plain Dubins path, which ignores obstacles.
            FALLBACK_USED.add(str(e) or "plan_path() is not written yet")
            path = dubins_path(problem.start, problem.spot.goal_pose())
            x, y, yaw = path.x, path.y, path.yaw
        return to_segments(x, y, yaw)
