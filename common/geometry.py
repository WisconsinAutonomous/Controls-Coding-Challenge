"""Geometry helpers: angle wrapping, polylines, Frenet frames, and track building.

You are free to use anything in here (the real stack has an equivalent
``ReferenceLine`` class), or to write your own.

Frenet convention: ``s`` is arc length along the polyline [m] and ``d`` is the
signed lateral offset from it [m], positive to the LEFT of the direction of
travel.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


class Polyline:
    """An ordered 2D polyline with arc length, projection and Frenet helpers."""

    def __init__(self, x, y, yaw=None, curvature=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
            raise ValueError("Polyline needs two equal-length 1D arrays with >= 2 points")
        seg = np.hypot(np.diff(x), np.diff(y))
        keep = np.concatenate(([True], seg > 1e-9))  # drop duplicate points
        self.x, self.y = x[keep], y[keep]
        dx, dy = np.diff(self.x), np.diff(self.y)
        self.seg_len = np.hypot(dx, dy)
        self.seg_heading = np.arctan2(dy, dx)
        self.s = np.concatenate(([0.0], np.cumsum(self.seg_len)))
        self.length = float(self.s[-1])
        if yaw is not None:
            self.yaw = np.unwrap(np.asarray(yaw, dtype=float)[keep])
        else:
            h = np.unwrap(self.seg_heading)
            node = np.empty(len(self.x))
            node[0], node[-1] = h[0], h[-1]
            node[1:-1] = 0.5 * (h[:-1] + h[1:])
            self.yaw = node
        if curvature is not None:
            self.curvature = np.asarray(curvature, dtype=float)[keep]
        else:
            self.curvature = np.gradient(self.yaw, self.s) if len(self.s) > 2 else np.zeros(len(self.s))

    # ------------------------------------------------------------------ #
    def project(self, px: float, py: float, s_hint: Optional[float] = None,
                window: float = 40.0) -> Tuple[float, float, int]:
        """Closest point on the polyline to (px, py).

        Returns ``(s, d, seg_idx)``.  If ``s_hint`` is given, only segments with
        arc length within ``window`` of it are searched.  Use a hint whenever
        the path passes close to itself (loops, hairpins) so you do not snap
        onto the wrong leg.
        """
        if s_hint is None:
            i0, i1 = 0, len(self.seg_len)
        else:
            i0 = max(0, int(np.searchsorted(self.s, s_hint - window)) - 1)
            i1 = min(len(self.seg_len), int(np.searchsorted(self.s, s_hint + window)) + 1)
            if i1 <= i0:
                i0, i1 = 0, len(self.seg_len)
        x0, y0 = self.x[i0:i1], self.y[i0:i1]
        dx, dy = self.x[i0 + 1:i1 + 1] - x0, self.y[i0 + 1:i1 + 1] - y0
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0)
        cx, cy = x0 + t * dx, y0 + t * dy
        dist2 = (px - cx) ** 2 + (py - cy) ** 2
        k = int(np.argmin(dist2))
        idx = i0 + k
        s = float(self.s[idx] + t[k] * self.seg_len[idx])
        cross = dx[k] * (py - y0[k]) - dy[k] * (px - x0[k])
        d = math.sqrt(float(dist2[k]))
        # Off the ends of the line, extend the end segment: s goes below 0 or
        # above `length`, and d is the signed offset from the extended line.
        if (idx == 0 and t[k] <= 0.0) or (idx == len(self.seg_len) - 1 and t[k] >= 1.0):
            h = self.seg_heading[idx]
            rx, ry = px - cx[k], py - cy[k]
            s += math.cos(h) * rx + math.sin(h) * ry
            d = -math.sin(h) * rx + math.cos(h) * ry
            return s, float(d), idx
        return s, float(math.copysign(d, cross)), idx

    def interp(self, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position and heading at arc length(s) ``s`` (clamped to the ends)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return (np.interp(s, self.s, self.x), np.interp(s, self.s, self.y),
                np.interp(s, self.s, self.yaw))

    def curvature_at(self, s):
        """Curvature at arc length(s) ``s`` [1/m]."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return np.interp(s, self.s, self.curvature)

    def frenet_to_xy(self, s, d) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Frenet (s, d) to world (x, y)."""
        x, y, h = self.interp(s)
        d = np.asarray(d, dtype=float)
        return x - d * np.sin(h), y + d * np.cos(h)

    def resampled(self, spacing: float) -> "Polyline":
        """A copy resampled at (approximately) uniform ``spacing`` in s."""
        n = max(2, int(math.ceil(self.length / spacing)) + 1)
        s = np.linspace(0.0, self.length, n)
        x, y, h = self.interp(s)
        return Polyline(x, y, yaw=h, curvature=self.curvature_at(s))


# ---------------------------------------------------------------------- #
class TrackBuilder:
    """Build a smooth road centerline from curvature pieces.

    Every piece specifies curvature as a function of arc length, and the
    heading and position are integrated from it, so heading and curvature are
    exact (not finite-differenced).

    Example::

        track = (TrackBuilder()
                 .straight(40)
                 .turn(radius=12, angle_deg=-90, transition=6)   # right turn
                 .straight(30)
                 .build())
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self._pieces: List[Tuple[float, Callable[[np.ndarray], np.ndarray]]] = []
        self._k_end = 0.0

    def _add(self, length: float, fn, k_end: float) -> "TrackBuilder":
        if length > 0:
            self._pieces.append((float(length), fn))
        self._k_end = k_end
        return self

    def straight(self, length: float) -> "TrackBuilder":
        return self._add(length, lambda u: np.zeros_like(u), 0.0)

    def arc(self, radius: float, angle_deg: float) -> "TrackBuilder":
        """Constant-curvature arc; positive angle turns left."""
        k = math.copysign(1.0 / radius, angle_deg)
        return self._add(radius * math.radians(abs(angle_deg)), lambda u: np.full_like(u, k), k)

    def clothoid(self, length: float, k_end: float) -> "TrackBuilder":
        """Curvature changes linearly from the current value to ``k_end``."""
        k0 = self._k_end
        return self._add(length, lambda u: k0 + (k_end - k0) * (u / length), k_end)

    def turn(self, radius: float, angle_deg: float, transition: float = 0.0) -> "TrackBuilder":
        """Clothoid in, arc, clothoid out.  Positive angle turns left."""
        k = math.copysign(1.0 / radius, angle_deg)
        total = math.radians(abs(angle_deg))
        arc_angle = total - abs(k) * transition   # each clothoid turns k*T/2
        if arc_angle < 0:
            raise ValueError("transition too long for this turn")
        self.clothoid(transition, k)
        self._add(radius * arc_angle, lambda u: np.full_like(u, k), k)
        return self.clothoid(transition, 0.0)

    def lane_change(self, offset: float, length: float) -> "TrackBuilder":
        """Smooth S-shaped lateral shift of ``offset`` m (positive = left)."""
        A = 2.0 * math.pi * offset / length ** 2
        return self._add(length, lambda u: A * np.sin(2.0 * math.pi * u / length), 0.0)

    def custom(self, length: float, curvature_fn) -> "TrackBuilder":
        """Arbitrary curvature(u) for u in [0, length]."""
        return self._add(length, curvature_fn, float(curvature_fn(np.array([length]))[0]))

    def build(self, spacing: float = 0.1, step: float = 0.01) -> Polyline:
        """Integrate the pieces and return a Polyline sampled every ``spacing`` m."""
        ks = [np.array([0.0])]
        s_total = 0.0
        for length, fn in self._pieces:
            n = max(1, int(round(length / step)))
            u = np.linspace(0.0, length, n + 1)[1:]
            ks.append(fn(u))
            s_total += length
        kappa = np.concatenate(ks)
        if self._pieces:
            kappa[0] = self._pieces[0][1](np.array([0.0]))[0]
        # uneven piece lengths => rebuild the s grid piece by piece
        s_parts = [np.array([0.0])]
        acc = 0.0
        for length, _ in self._pieces:
            n = max(1, int(round(length / step)))
            s_parts.append(acc + np.linspace(0.0, length, n + 1)[1:])
            acc += length
        s = np.concatenate(s_parts)
        ds = np.diff(s)
        yaw = self.yaw0 + np.concatenate(([0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)))
        c, sn = np.cos(yaw), np.sin(yaw)
        x = self.x0 + np.concatenate(([0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * ds)))
        y = self.y0 + np.concatenate(([0.0], np.cumsum(0.5 * (sn[1:] + sn[:-1]) * ds)))
        n_out = max(2, int(round(s[-1] / spacing)) + 1)
        s_out = np.linspace(0.0, s[-1], n_out)
        return Polyline(np.interp(s_out, s, x), np.interp(s_out, s, y),
                        yaw=np.interp(s_out, s, yaw), curvature=np.interp(s_out, s, kappa))
