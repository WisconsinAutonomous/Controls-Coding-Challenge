"""Parking scenarios.  Every lot is synthetic.  ``seed`` 0 is the public
layout.  Every seed (0 included) jitters the parked cars (+/- 0.2 m, +/- 3 deg,
varied car sizes); seeds other than 0 also move the start pose and, for some
scenarios, which spot is free.  The cars right next to the target spot in
``angled``, ``perpendicular`` and ``tight_aisle`` are fixed on purpose."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

from .lot import OrientedBox, ParkingProblem, ParkingSpot, Pose

HALF_PI = 0.5 * math.pi

# Par length [m] and par number of direction changes, per scenario.  Set from
# the reference solution over seeds 0..9: the median path length rounded up
# to the next meter, and the number of direction changes it needed on most
# seeds.  (Full length points go up to 1.05 x par.)
PAR = {
    "empty_lot": (18.0, 0),
    "perpendicular": (20.0, 0),
    "angled": (18.0, 0),
    "around_the_row": (68.0, 0),
    "parallel": (8.0, 1),
    "back_in": (23.0, 1),
    "tight_aisle": (23.0, 2),
    "cluttered": (19.0, 0),
}


def _car(rng, x, y, yaw, jitter=True, size=((4.3, 4.9), (1.75, 1.95))) -> OrientedBox:
    """A parked car centered at (x, y) along ``yaw``, with seed-dependent jitter."""
    length = float(rng.uniform(*size[0]))
    width = float(rng.uniform(*size[1]))
    if jitter:
        c, s = math.cos(yaw), math.sin(yaw)
        dl, dw = rng.uniform(-0.2, 0.2, size=2)
        x, y = x + dl * c - dw * s, y + dl * s + dw * c
        yaw += math.radians(float(rng.uniform(-3.0, 3.0)))
        if rng.random() < 0.5:           # parked nose-in or backed in
            yaw += math.pi
    return OrientedBox(float(x), float(y), float(yaw), length, width, "car")


def _row(rng, x_first, n, pitch, y_entry, facing, depth, width, free=(),
         size=((4.3, 4.9), (1.75, 1.95)), pull=0.0) -> Tuple[List[ParkingSpot], List[OrientedBox]]:
    """A row of perpendicular spots opening onto an aisle edge at ``y_entry``.

    ``facing`` = +1 for spots north of the edge, -1 for spots south of it.
    ``pull`` moves parked cars that many meters toward the aisle.
    Returns the spots (as ParkingSpot, nose-in heading) and the parked cars.
    """
    spots, cars = [], []
    yaw = facing * HALF_PI
    for k in range(n):
        sp = ParkingSpot(x_first + k * pitch, y_entry + facing * 0.5 * depth, yaw, width, depth)
        spots.append(sp)
        if k not in free:
            cars.append(_car(rng, sp.x, sp.y - facing * pull, yaw, size=size))
    return spots, cars


def _fixed_flanks(obstacles, spot, offsets, shift_toward, length=4.7, width=1.85, front_gap=0.3):
    """Replace the cars in the two spots beside ``spot`` with identical,
    straight-parked cars that are the same for every seed.

    ``offsets`` maps "left"/"right" to the neighbor spot's center in the
    target spot's frame (along, across); ``shift_toward`` moves that car the
    given distance toward the target spot (a slightly sloppy parker).  The
    aisle end of each car is ``front_gap`` inside the spot line.
    """
    c, s = math.cos(spot.yaw), math.sin(spot.yaw)

    def is_flank(o):
        lx, ly = spot.to_local(o.x, o.y)
        return o.kind == "car" and any(abs(float(lx) - a) < 1.2 and abs(float(ly) - b) < 1.2
                                       for a, b in offsets.values())

    kept = [o for o in obstacles if not is_flank(o)]
    for side, (a, b) in offsets.items():
        b2 = b - math.copysign(shift_toward.get(side, 0.0), b)
        a2 = a - 0.5 * (spot.depth - length) + front_gap
        kept.append(OrientedBox(spot.x + a2 * c - b2 * s, spot.y + a2 * s + b2 * c, spot.yaw, length, width, "car"))
    return kept


def _curb(x0, x1, y, kind="curb", thickness=0.3) -> OrientedBox:
    return OrientedBox(0.5 * (x0 + x1), y, 0.0, x1 - x0, thickness, kind)


def _start(rng, seed, x, y, yaw, dx=1.0, dy=0.4, dyaw_deg=4.0) -> Pose:
    if seed == 0:
        return Pose(x, y, yaw)
    return Pose(x + float(rng.uniform(-dx, dx)), y + float(rng.uniform(-dy, dy)),
                yaw + math.radians(float(rng.uniform(-dyaw_deg, dyaw_deg))))


# ---------------------------------------------------------------------- #
def _perpendicular_lot(rng, seed, k_target, spot_w=3.0, aisle=8.0, depth=5.5, n=14,
                       size=((4.3, 4.9), (1.75, 1.95)), pull=0.0):
    x_first = 1.5 + spot_w
    north, north_cars = _row(rng, x_first, n, spot_w, aisle, +1, depth, spot_w, free={k_target}, size=size)
    south, south_cars = _row(rng, x_first, n, spot_w, 0.0, -1, depth, spot_w, size=size, pull=pull)
    x_end = x_first + (n - 0.5) * spot_w + 1.5
    obstacles = north_cars + south_cars + [
        _curb(0.0, x_end, aisle + depth + 0.15), _curb(0.0, x_end, -depth - 0.15)]
    bounds = (0.0, x_end, -depth - 0.3, aisle + depth + 0.3)
    return north[k_target], obstacles, bounds


def empty_lot(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 700)
    k = 7 if seed == 0 else int(rng.integers(5, 10))
    aisle, depth, w, n = 10.0, 5.5, 3.0, 14
    x_first = 1.5 + w
    # nobody parked within two spots of the target
    _, north_cars = _row(rng, x_first, n, w, aisle, +1, depth, w, free=set(range(k - 2, k + 3)))
    north, _ = _row(rng, x_first, n, w, aisle, +1, depth, w)
    _, south_cars = _row(rng, x_first, n, w, 0.0, -1, depth, w)
    x_end = x_first + (n - 0.5) * w + 1.5
    obstacles = north_cars + south_cars + [_curb(0.0, x_end, aisle + depth + 0.15), _curb(0.0, x_end, -depth - 0.15)]
    bounds = (0.0, x_end, -depth - 0.3, aisle + depth + 0.3)
    spot = north[k]
    start = _start(rng, seed, spot.x - 15.0, 5.0, 0.0)
    return ParkingProblem("empty_lot", "Warm-up: nose-in with no neighbors and a 10 m aisle.",
                          start, spot, obstacles, bounds, core=True)


def perpendicular(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed)
    k = 7 if seed == 0 else int(rng.integers(5, 10))
    spot, obstacles, bounds = _perpendicular_lot(rng, seed, k)
    # The car on the right of the spot is parked 10 cm toward it, the same in
    # every layout: the shortest forward path turns in at full lock and its
    # front corner clips that car, so you have to find another way in.
    obstacles = _fixed_flanks(obstacles, spot, {"left": (0.0, spot.width), "right": (0.0, -spot.width)},
                              {"right": 0.10})
    start = _start(rng, seed, spot.x - 15.0, 4.0, 0.0)
    return ParkingProblem("perpendicular", "Nose-in to a 3.0 m spot off an 8 m aisle.",
                          start, spot, obstacles, bounds, core=True)


def back_in(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 100)
    k = 7 if seed == 0 else int(rng.integers(5, 10))
    spot, obstacles, bounds = _perpendicular_lot(rng, seed, k)
    spot = ParkingSpot(spot.x, spot.y, -HALF_PI, spot.width, spot.depth)   # face the aisle
    start = _start(rng, seed, spot.x - 12.0, 4.0, 0.0)
    return ParkingProblem("back_in", "Back into a 3.0 m spot so the car faces the aisle.",
                          start, spot, obstacles, bounds, core=False)


def tight_aisle(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 200)
    k = 7 if seed == 0 else int(rng.integers(5, 10))
    w, aisle, depth = 2.6, 6.0, 5.5
    # The opposite row is parked close to the aisle line.
    spot, obstacles, bounds = _perpendicular_lot(rng, seed, k, spot_w=w, aisle=aisle, pull=0.3)
    # The two cars next to the spot are the same for every seed (they decide
    # whether one forward move can work): a sedan on the left, an SUV on the right.
    y_row = aisle + 0.5 * depth
    flank = [OrientedBox(spot.x - w, y_row, HALF_PI, 4.7, 1.85, "car"),
             OrientedBox(spot.x + w, y_row, HALF_PI, 5.0, 1.9, "car")]
    obstacles = [o for o in obstacles
                 if not (o.kind == "car" and abs(o.y - y_row) < 1.0 and 0.5 * w < abs(o.x - spot.x) < 1.5 * w)]
    obstacles += flank
    start = _start(rng, seed, spot.x - 15.0, 3.0, 0.0, dy=0.3)
    return ParkingProblem("tight_aisle", "Nose-in to a 2.6 m spot off a 6 m aisle: expect a multi-point turn.",
                          start, spot, obstacles, bounds, core=False)


def cluttered(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 300)
    k = 7 if seed == 0 else int(rng.integers(6, 10))
    spot, obstacles, bounds = _perpendicular_lot(rng, seed, k)
    start = _start(rng, seed, spot.x - 15.0, 4.0, 0.0)
    # Clutter in the aisle between the start and the spot.  Items stay either
    # in the south half of the aisle, or in the north half well before the
    # spot, so the car always has a way through.
    items = [("cone", 0.4, 0.4), ("cone", 0.4, 0.4), ("cone", 0.4, 0.4),
             ("cart", 1.0, 0.6), ("person", 0.5, 0.5)]
    placed: List[OrientedBox] = []
    x_lo, x_hi = start.x + 6.0, spot.x + 6.0
    for kind, length, width in items:
        for _ in range(100):
            if rng.random() < 0.6:
                x, y = rng.uniform(x_lo, x_hi), rng.uniform(0.5, 2.6)
            else:
                x, y = rng.uniform(x_lo, spot.x - 7.0), rng.uniform(4.8, 7.4)
            if all(math.hypot(x - o.x, y - o.y) > 2.5 for o in placed):
                break
        placed.append(OrientedBox(float(x), float(y), float(rng.uniform(0, math.pi)), length, width, kind))
    return ParkingProblem("cluttered", "The perpendicular lot with cones, a cart and a person in the aisle.",
                          start, spot, obstacles + placed, bounds, core=False)


def angled(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 400)
    theta = math.radians(60.0)
    width, depth, aisle, n = 3.0, 5.5, 6.0, 12
    pitch = width / math.sin(theta)
    # distance from a spot center to its lowest corner, measured across the aisle
    reach = 0.5 * depth * math.sin(theta) + 0.5 * width * math.cos(theta)
    k_target = 6 if seed == 0 else int(rng.integers(4, 9))
    obstacles: List[OrientedBox] = []
    target = None
    x_first = 4.0
    for k in range(n):
        # north side: cars turn left 60 deg into these (one-way aisle heading east)
        sp = ParkingSpot(x_first + k * pitch, aisle + reach, theta, width, depth)
        if k == k_target:
            target = sp
        else:
            obstacles.append(_car(rng, sp.x, sp.y, sp.yaw))
        # south side: mirror image, turn right into these
        obstacles.append(_car(rng, x_first + 1.2 + k * pitch, -reach, -theta))
    # The two cars beside the target are the same in every layout, the left one
    # parked 8 cm toward it: the shortest forward path into the spot clips it.
    stagger = pitch * math.cos(theta)
    obstacles = _fixed_flanks(obstacles, target, {"left": (-stagger, width), "right": (stagger, -width)},
                              {"left": 0.08})
    x_end = x_first + n * pitch + 2.0
    top = aisle + 2 * reach
    obstacles += [_curb(0.0, x_end, top + 0.15), _curb(0.0, x_end, -2 * reach - 0.15)]
    bounds = (0.0, x_end, -2 * reach - 0.3, top + 0.3)
    start = _start(rng, seed, target.x - 16.0, 3.0, 0.0, dy=0.3)
    return ParkingProblem("angled", "Nose-in to a 60 deg angled spot off a 6 m one-way aisle.",
                          start, target, obstacles, bounds, core=True)


def around_the_row(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 500)
    w, d = 3.0, 5.5
    n_inner, n_outer = 11, 14
    k_target = 3 if seed == 0 else int(rng.integers(2, 7))
    x_first = 1.5
    _, a1_south = _row(rng, x_first, n_outer, w, 0.0, -1, d, w)                 # below aisle 1
    _, inner_low = _row(rng, x_first, n_inner, w, 8.0, +1, d, w)                 # double row, faces aisle 1
    up_spots, inner_up = _row(rng, x_first, n_inner, w, 19.0, -1, d, w, free={k_target})  # faces aisle 2
    _, a2_north = _row(rng, x_first, n_outer, w, 27.0, +1, d, w)                 # above aisle 2
    x_row_end = x_first + (n_inner - 0.5) * w
    x_end = x_first + (n_outer - 0.5) * w
    obstacles = a1_south + inner_low + inner_up + a2_north + [
        OrientedBox(0.5 * x_row_end, 13.5, 0.0, x_row_end, 0.3, "island"),     # divider in the double row
        _curb(0.0, x_end, -d - 0.15), _curb(0.0, x_end, 27.0 + d + 0.15)]
    bounds = (0.0, x_end, -d - 0.3, 27.0 + d + 0.3)
    start = _start(rng, seed, 6.0, 4.0, 0.0, dx=1.0)
    return ParkingProblem("around_the_row", "The only free spot is on the far side of the double row.",
                          start, up_spots[k_target], obstacles, bounds, core=False)


def parallel(seed: int = 0) -> ParkingProblem:
    rng = np.random.default_rng(seed + 600)
    gap, lane = 7.0, 2.4
    x_gap0 = 15.0
    x_gap1 = x_gap0 + gap
    obstacles: List[OrientedBox] = []
    # parked cars along the curb: bumpers fixed at the gap, lengths vary outward
    x = x_gap0
    for _ in range(2):
        c = _car(rng, 0.0, 0.0, 0.0, jitter=False)
        obstacles.append(OrientedBox(x - 0.5 * c.length, 1.2 + float(rng.uniform(-0.1, 0.1)),
                                     math.radians(float(rng.uniform(-2, 2))), c.length, c.width, "car"))
        x -= c.length + float(rng.uniform(1.0, 2.0))
    x = x_gap1
    front_rear_bumper = x_gap1
    for _ in range(2):
        c = _car(rng, 0.0, 0.0, 0.0, jitter=False)
        obstacles.append(OrientedBox(x + 0.5 * c.length, 1.2 + float(rng.uniform(-0.1, 0.1)),
                                     math.radians(float(rng.uniform(-2, 2))), c.length, c.width, "car"))
        x += c.length + float(rng.uniform(1.0, 2.0))
    x_end = 40.0
    # far side of the street: another parking lane with a few cars
    for xc in (4.0, 11.0, 26.0, 33.5):
        obstacles.append(_car(rng, xc + float(rng.uniform(-1, 1)), 8.4, 0.0))
    obstacles += [_curb(0.0, x_end, -0.15), _curb(0.0, x_end, 9.75)]
    spot = ParkingSpot(0.5 * (x_gap0 + x_gap1), 0.5 * lane, 0.0, lane, gap)
    start = _start(rng, seed, front_rear_bumper + 0.8, 3.8, 0.0, dx=0.5, dy=0.2, dyaw_deg=2.0)
    return ParkingProblem("parallel", "Parallel park into a 7 m gap at the curb.",
                          start, spot, obstacles, (0.0, x_end, -0.3, 9.9), core=False)


SCENARIOS: Dict[str, Callable[[int], ParkingProblem]] = {
    "empty_lot": empty_lot,
    "angled": angled,
    "perpendicular": perpendicular,
    "around_the_row": around_the_row,
    "parallel": parallel,
    "back_in": back_in,
    "tight_aisle": tight_aisle,
    "cluttered": cluttered,
}
