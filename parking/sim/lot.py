"""Parking lot geometry: oriented boxes, the car footprint, the target spot,
and exact collision / clearance checks between them.

All poses are the REAR AXLE in local ENU (x = East, y = North) with ``yaw``
the car's heading (0 = East, counter-clockwise positive), also when reversing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from common import vehicle as V
from common.geometry import wrap_to_pi

REAR = V.REAR_AXLE_TO_REAR_BUMPER     # 0.8 m behind the rear axle
FRONT = V.REAR_AXLE_TO_FRONT_BUMPER   # 3.7 m ahead of the rear axle
HALF_W = 0.5 * V.WIDTH                # 0.9 m
CENTER_OFFSET = 0.5 * (FRONT - REAR)  # rear axle -> geometric center, 1.45 m


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


def footprints(xs, ys, yaws) -> np.ndarray:
    """Car footprint corners for many rear-axle poses, shape (N, 4, 2).

    Corner order (counter-clockwise): rear-right, front-right, front-left, rear-left.
    """
    xs, ys, yaws = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (xs, ys, yaws))
    c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    lon = np.array([-REAR, FRONT, FRONT, -REAR])
    lat = np.array([-HALF_W, -HALF_W, HALF_W, HALF_W])
    px = xs[:, None] + lon * c - lat * s
    py = ys[:, None] + lon * s + lat * c
    return np.stack([px, py], axis=-1)


def footprint(x: float, y: float, yaw: float) -> np.ndarray:
    """Car footprint corners for one rear-axle pose, shape (4, 2)."""
    return footprints([x], [y], [yaw])[0]


def _box_corners(x, y, yaw, length, width) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    lon = np.array([-hl, hl, hl, -hl])
    lat = np.array([-hw, -hw, hw, hw])
    return np.stack([x + lon * c - lat * s, y + lon * s + lat * c], axis=-1)


@dataclass
class OrientedBox:
    """A rectangular obstacle. ``yaw`` is the direction of its ``length``."""

    x: float
    y: float
    yaw: float
    length: float
    width: float
    kind: str = "car"     # car | curb | wall | cone | cart | island | person

    def corners(self) -> np.ndarray:
        return _box_corners(self.x, self.y, self.yaw, self.length, self.width)


@dataclass
class ParkingSpot:
    """The target spot.  ``yaw`` is the heading the PARKED car must face;
    ``depth`` is measured along that heading and ``width`` across it."""

    x: float
    y: float
    yaw: float
    width: float
    depth: float

    def corners(self) -> np.ndarray:
        return _box_corners(self.x, self.y, self.yaw, self.depth, self.width)

    def to_local(self, px, py) -> Tuple[np.ndarray, np.ndarray]:
        """World points -> (along-spot, across-spot) coordinates from the spot center."""
        dx, dy = np.asarray(px) - self.x, np.asarray(py) - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return dx * c + dy * s, -dx * s + dy * c

    def contains(self, x: float, y: float, yaw: float, tol: float = 1e-6) -> bool:
        """True if the whole car footprint at this rear-axle pose is inside the spot."""
        lon, lat = self.to_local(*footprint(x, y, yaw).T)
        return bool(np.all(np.abs(lon) <= 0.5 * self.depth + tol)
                    and np.all(np.abs(lat) <= 0.5 * self.width + tol))

    def goal_pose(self) -> Pose:
        """Rear-axle pose that puts the car exactly centered in the spot."""
        return Pose(self.x - CENTER_OFFSET * math.cos(self.yaw),
                    self.y - CENTER_OFFSET * math.sin(self.yaw), self.yaw)

    def placement_error(self, x: float, y: float, yaw: float) -> dict:
        """Where the car's center sits relative to the spot center, and heading error."""
        cx = x + CENTER_OFFSET * math.cos(yaw)
        cy = y + CENTER_OFFSET * math.sin(yaw)
        lon, lat = self.to_local(cx, cy)
        return {"lateral_m": float(lat), "longitudinal_m": float(lon),
                "heading_deg": float(np.degrees(wrap_to_pi(yaw - self.yaw)))}


@dataclass
class ParkingProblem:
    """Everything the planner gets.  See the README for conventions."""

    name: str
    description: str
    start: Pose
    spot: ParkingSpot
    obstacles: List[OrientedBox]
    bounds: Tuple[float, float, float, float]   # xmin, xmax, ymin, ymax
    core: bool = True
    _boxes: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.obstacles:
            self._boxes = np.stack([o.corners() for o in self.obstacles])   # (M, 4, 2)
        else:
            self._boxes = np.zeros((0, 4, 2))

    # ------------------------------------------------------------------ #
    def collides(self, x: float, y: float, yaw: float) -> bool:
        """True if the car at this rear-axle pose touches an obstacle or leaves the bounds."""
        return bool(self.collides_many([x], [y], [yaw])[0])

    def collides_many(self, xs, ys, yaws) -> np.ndarray:
        """Vectorized ``collides`` for arrays of poses.  Returns a bool array."""
        cars = footprints(xs, ys, yaws)
        out = _outside_bounds(cars, self.bounds)
        if len(self._boxes):
            for i0 in range(0, len(cars), 256):
                chunk = cars[i0:i0 + 256]
                out[i0:i0 + 256] |= _sat_overlap(chunk, self._boxes).any(axis=1)
        return out

    def clearance(self, x: float, y: float, yaw: float) -> float:
        """Distance [m] from the car footprint to the nearest obstacle or bound (0 if touching)."""
        return float(self.clearance_many([x], [y], [yaw])[0])

    def clearance_many(self, xs, ys, yaws) -> np.ndarray:
        """Vectorized ``clearance`` for arrays of poses."""
        cars = footprints(xs, ys, yaws)
        xmin, xmax, ymin, ymax = self.bounds
        px, py = cars[..., 0], cars[..., 1]
        d = np.min(np.stack([px - xmin, xmax - px, py - ymin, ymax - py]), axis=0).min(axis=1)
        d = np.maximum(d, 0.0)
        if len(self._boxes):
            for i0 in range(0, len(cars), 128):
                chunk = cars[i0:i0 + 128]
                dist = _poly_distance(chunk, self._boxes)              # (n, M)
                dist[_sat_overlap(chunk, self._boxes)] = 0.0
                d[i0:i0 + 128] = np.minimum(d[i0:i0 + 128], dist.min(axis=1))
        return d


# ---------------------------------------------------------------------- #
def _outside_bounds(cars: np.ndarray, bounds) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    px, py = cars[..., 0], cars[..., 1]
    return ((px < xmin) | (px > xmax) | (py < ymin) | (py > ymax)).any(axis=1)


def _axes(polys: np.ndarray) -> np.ndarray:
    """Two unit edge directions per rectangle, shape (K, 2, 2)."""
    e1 = polys[:, 1] - polys[:, 0]
    e2 = polys[:, 3] - polys[:, 0]
    a = np.stack([e1, e2], axis=1)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _sat_overlap(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Separating-axis test between rectangles A (N,4,2) and B (M,4,2) -> (N, M) bool."""
    ax_a, ax_b = _axes(A), _axes(B)
    # project on A's axes
    pa = np.einsum("nkd,nvd->nkv", ax_a, A)                  # (N,2,4)
    pb = np.einsum("nkd,mvd->nmkv", ax_a, B)                 # (N,M,2,4)
    sep = (pb.max(-1) < pa.min(-1)[:, None]) | (pb.min(-1) > pa.max(-1)[:, None])
    # project on B's axes
    qb = np.einsum("mkd,mvd->mkv", ax_b, B)                  # (M,2,4)
    qa = np.einsum("mkd,nvd->nmkv", ax_b, A)                 # (N,M,2,4)
    sep |= (qa.max(-1) < qb.min(-1)[None]) | (qa.min(-1) > qb.max(-1)[None])
    return ~sep.any(-1)


def _point_seg_dist(P: np.ndarray, S0: np.ndarray, S1: np.ndarray) -> np.ndarray:
    d = S1 - S0
    t = np.clip(np.sum((P - S0) * d, -1) / np.maximum(np.sum(d * d, -1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (S0 + t[..., None] * d), axis=-1)


def _poly_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance between separated convex quads A (N,4,2) and B (M,4,2) -> (N, M)."""
    a_pts = A[:, None, :, None, :]                            # (N,1,4,1,2)
    b0, b1 = B[None, :, None, :, :], np.roll(B, -1, axis=1)[None, :, None, :, :]
    d1 = _point_seg_dist(a_pts, b0, b1).min(axis=(2, 3))      # A corners to B edges
    b_pts = B[None, :, :, None, :]                            # (1,M,4,1,2)
    a0, a1 = A[:, None, None, :, :], np.roll(A, -1, axis=1)[:, None, None, :, :]
    d2 = _point_seg_dist(b_pts, a0, a1).min(axis=(2, 3))      # B corners to A edges
    return np.minimum(d1, d2)
