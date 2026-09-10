"""Small geometry kernels used by the simulator and grader."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from common import vehicle as V
from common.geometry import Polyline

HALF_W = 0.5 * V.WIDTH


def car_corners(x, y, yaw) -> np.ndarray:
    """Footprint corners for pose(s).  Returns (..., 4, 2), counter-clockwise:
    front-left, rear-left, rear-right, front-right."""
    x, y, yaw = np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    lon = np.array([V.REAR_AXLE_TO_FRONT_BUMPER, -V.REAR_AXLE_TO_REAR_BUMPER,
                    -V.REAR_AXLE_TO_REAR_BUMPER, V.REAR_AXLE_TO_FRONT_BUMPER])
    lat = np.array([HALF_W, HALF_W, -HALF_W, -HALF_W])
    cx = x[..., None] + c * lon - s * lat
    cy = y[..., None] + s * lon + c * lat
    return np.stack([cx, cy], axis=-1)


def front_bumper_xy(x, y, yaw):
    return (np.asarray(x) + V.REAR_AXLE_TO_FRONT_BUMPER * np.cos(yaw),
            np.asarray(y) + V.REAR_AXLE_TO_FRONT_BUMPER * np.sin(yaw))


def quads_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quads given as (4, 2) arrays."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))):
            return False
    return True


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(np.sum((p[:, None, :] - a[None]) * ab[None], axis=2)
                / np.maximum(np.sum(ab * ab, axis=1), 1e-12)[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - proj, axis=2)


def quad_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex quads (0 if they overlap)."""
    if quads_overlap(a, b):
        return 0.0
    ea, eb = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    return float(min(_point_seg_dist(a, b, eb).min(), _point_seg_dist(b, a, ea).min()))


def point_quad_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Distance from a point to a convex quad (0 if inside)."""
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * (p[1] - q[:, 1]) - edges[:, 1] * (p[0] - q[:, 0])
    if np.all(cross >= 0) or np.all(cross <= 0):
        return 0.0
    return float(_point_seg_dist(p[None], q, np.roll(q, -1, axis=0)).min())


def project_points(poly: Polyline, px, py, s_hint: float, window: float = 25.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Frenet projection of many points near ``s_hint``."""
    px, py = np.atleast_1d(np.asarray(px, float)), np.atleast_1d(np.asarray(py, float))
    i0 = max(0, int(np.searchsorted(poly.s, s_hint - window)) - 1)
    i1 = min(len(poly.seg_len), int(np.searchsorted(poly.s, s_hint + window)) + 1)
    x0, y0 = poly.x[i0:i1], poly.y[i0:i1]
    dx, dy = poly.x[i0 + 1:i1 + 1] - x0, poly.y[i0 + 1:i1 + 1] - y0
    L2 = np.maximum(dx * dx + dy * dy, 1e-12)
    rx, ry = px[:, None] - x0[None], py[:, None] - y0[None]
    t = np.clip((rx * dx + ry * dy) / L2, 0.0, 1.0)
    ex, ey = rx - t * dx, ry - t * dy
    dist2 = ex * ex + ey * ey
    k = np.argmin(dist2, axis=1)
    rows = np.arange(len(px))
    s = poly.s[i0 + k] + t[rows, k] * poly.seg_len[i0 + k]
    cross = dx[k] * ry[rows, k] - dy[k] * rx[rows, k]
    d = np.sqrt(dist2[rows, k]) * np.where(cross < 0, -1.0, 1.0)
    # past either end of the line: extend the end segment (like Polyline.project)
    seg = i0 + k
    tk = t[rows, k]
    ends = ((seg == 0) & (tk <= 0.0)) | ((seg == len(poly.seg_len) - 1) & (tk >= 1.0))
    if np.any(ends):
        h = poly.seg_heading[seg[ends]]
        ex_, ey_ = ex[rows, k][ends], ey[rows, k][ends]
        s[ends] += np.cos(h) * ex_ + np.sin(h) * ey_
        d[ends] = -np.sin(h) * ex_ + np.cos(h) * ey_
    return s, d


def path_curvature(x: np.ndarray, y: np.ndarray, step: float = 0.25,
                   half_window: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curvature of a driven path, measured over ~1 m windows.

    Returns (s, kappa, keep_idx_s) where ``s`` is arc length of the resampled
    points.  Headings come from chords spanning 2*half_window*step meters and
    curvature from heading change over the same span.
    """
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] < 4 * half_window * step:
        return np.zeros(0), np.zeros(0), s
    mask = np.concatenate(([True], seg > 1e-6))
    su, xu, yu = s[mask], x[mask], y[mask]
    sr = np.arange(0.0, su[-1], step)
    xr, yr = np.interp(sr, su, xu), np.interp(sr, su, yu)
    h = half_window
    hx, hy = xr[2 * h:] - xr[:-2 * h], yr[2 * h:] - yr[:-2 * h]
    head = np.unwrap(np.arctan2(hy, hx))          # heading at sr[h:-h]
    kap = (head[2 * h:] - head[:-2 * h]) / (2 * h * step)   # at sr[2h:-2h]
    return sr[2 * h:-2 * h], kap, s


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi
