"""Toolbox for the parking challenge.  Use as much or as little as you like.

    from tools import dubins_path, FastChecker, staging_candidates, join, to_segments

* ``dubins_path(start, goal)``        shortest FORWARD-ONLY path between two poses
                                      (ignores obstacles), or None
* ``dubins_paths(start, goal)``       all forward-only Dubins paths, shortest first
* ``FastChecker(problem)``            fast collision and clearance checks for many poses
* ``get_checker(problem)``            the same, built once per problem and reused
* ``staging_candidates(problem, n)``  collision-free poses in the open part of the lot,
                                      good places to aim for on the way to the spot
* ``join(*paths)``                    glue pieces end to end
* ``to_segments(path)``               turn a path into the list of ReferenceTrajectory that
                                      ``plan()`` returns (split at direction changes,
                                      valid spacing)

A "pose" is anything with ``.x, .y, .yaw`` (like ``problem.start`` or
``problem.spot.goal_pose()``) or a tuple ``(x, y, yaw)``: the REAR AXLE in ENU
meters and the heading of the car's nose.  ``dubins_path`` returns a ``Path``:
arrays ``x, y, yaw`` sampled every ``step`` meters (plus a ``reverse`` flag per
point), with ``path.end`` the final pose and ``path.length`` its length.
Anywhere a Path is accepted, a plain ``(x, y, yaw)`` tuple of arrays works too.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional

import numpy as np

from common import vehicle
from common.messages import ReferenceTrajectory
from sim.lot import FRONT, HALF_W, REAR, _sat_overlap, footprints

MIN_RADIUS = 1.0 / vehicle.MAX_CURVATURE    # 4.66 m at the rear axle
DEFAULT_RADIUS = 1.02 * MIN_RADIUS           # a little margin under the steering limit


class Path(NamedTuple):
    """Sampled poses.  ``reverse[i]`` is True if point i is reached driving backward."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    reverse: np.ndarray

    @property
    def end(self):
        """The last pose, as (x, y, yaw)."""
        return float(self.x[-1]), float(self.y[-1]), float(self.yaw[-1])

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))


def _pose(p):
    if hasattr(p, "x"):
        return float(p.x), float(p.y), float(p.yaw)
    x, y, yaw = p
    return float(x), float(y), float(yaw)


# ---------------------------------------------------------------------- #
# Primitives
def _arc(pose, curvature: float, length: float, reverse: bool = False, step: float = 0.1) -> Path:
    """Constant-curvature piece used to sample Dubins paths (private: writing
    this yourself is Part 1 of the challenge)."""
    x0, y0, h0 = _pose(pose)
    if abs(curvature) > vehicle.MAX_CURVATURE * 1.0001:
        raise ValueError(f"curvature {curvature:.3f} is tighter than the car can turn "
                         f"(max {vehicle.MAX_CURVATURE:.3f} 1/m)")
    n = max(1, int(math.ceil(abs(length) / step)))
    u = np.linspace(0.0, abs(length), n + 1) * (-1.0 if reverse else 1.0)   # signed distance
    h = h0 + curvature * u
    if abs(curvature) > 1e-12:
        x = x0 + (np.sin(h) - math.sin(h0)) / curvature
        y = y0 - (np.cos(h) - math.cos(h0)) / curvature
    else:
        x = x0 + u * math.cos(h0)
        y = y0 + u * math.sin(h0)
    return Path(x, y, h, np.full(len(x), bool(reverse)))


def _as_path(p) -> Path:
    """Accept a Path or an (x, y, yaw) / (x, y, yaw, reverse) tuple of arrays."""
    if isinstance(p, Path):
        return p
    if len(p) == 4:
        x, y, yaw, rev = p
    else:
        x, y, yaw = p
        rev = None
    x, y, yaw = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x, y, yaw))
    if rev is None:
        rev = _infer_reverse(x, y, yaw)
    return Path(x, y, yaw, np.broadcast_to(np.asarray(rev, dtype=bool), x.shape).copy())


def _infer_reverse(x, y, yaw) -> np.ndarray:
    """Per point: True if the step INTO it moves against the car's heading."""
    rev = np.zeros(len(x), dtype=bool)
    if len(x) > 1:
        dot = np.diff(x) * np.cos(yaw[1:]) + np.diff(y) * np.sin(yaw[1:])
        rev[1:] = dot < 0
        rev[0] = rev[1]
    return rev


def join(*paths) -> Path:
    """Glue paths end to end.  Each must start where the previous one ended.

    Pieces can be Paths or (x, y, yaw) tuples of arrays (the direction of
    travel is then worked out from the geometry).
    """
    paths = [_as_path(p) for p in paths if p is not None]
    if not paths:
        raise ValueError("join() needs at least one path")
    xs, ys, hs, rs = [paths[0].x], [paths[0].y], [paths[0].yaw], [paths[0].reverse]
    for prev, p in zip(paths[:-1], paths[1:]):
        ex, ey, eh = prev.end
        gap = math.hypot(p.x[0] - ex, p.y[0] - ey)
        dh = abs(math.remainder(p.yaw[0] - eh, 2 * math.pi))
        if gap > 1e-3 or dh > 1e-3:
            raise ValueError(f"join(): pieces do not connect ({gap:.3f} m, {math.degrees(dh):.2f} deg)")
        # keep yaw continuous across the join
        shift = eh - p.yaw[0] - math.remainder(eh - p.yaw[0], 2 * math.pi)
        xs.append(p.x[1:])
        ys.append(p.y[1:])
        hs.append(p.yaw[1:] + shift)
        rs.append(p.reverse[1:])
    return Path(np.concatenate(xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(rs))


# ---------------------------------------------------------------------- #
# Dubins paths: shortest paths for a car that only drives forward
def _m2pi(a):
    return a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi))


def _dubins_words(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    out = []
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out.append(("LSL", _m2pi(-alpha + tmp), math.sqrt(p2), _m2pi(beta - tmp)))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out.append(("RSR", _m2pi(alpha - tmp), math.sqrt(p2), _m2pi(-beta + tmp)))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(("LSR", _m2pi(-alpha + tmp), p, _m2pi(-_m2pi(beta) + tmp)))
    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(("RSL", _m2pi(alpha - tmp), p, _m2pi(beta - tmp)))
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1:
        p = _m2pi(2 * math.pi - math.acos(tmp))
        t = _m2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        out.append(("RLR", t, p, _m2pi(alpha - beta - t + p)))
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1:
        p = _m2pi(2 * math.pi - math.acos(tmp))
        t = _m2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        out.append(("LRL", t, p, _m2pi(_m2pi(beta) - alpha - t + p)))
    return out


def dubins_paths(start, goal, radius: Optional[float] = None, step: float = 0.05) -> List[Path]:
    """All forward-only Dubins paths from ``start`` to ``goal``, shortest first.

    Each is an arc, a straight or an arc, and an arc (six "words": LSL, RSR,
    LSR, RSL, RLR, LRL).  They ignore obstacles: check them yourself.
    """
    r = DEFAULT_RADIUS if radius is None else float(radius)
    if r < MIN_RADIUS * 0.9999:
        raise ValueError(f"radius {r:.2f} m is tighter than the car can turn (min {MIN_RADIUS:.2f} m)")
    x0, y0, h0 = _pose(start)
    x1, y1, h1 = _pose(goal)
    d = math.hypot(x1 - x0, y1 - y0) / r
    th = _m2pi(math.atan2(y1 - y0, x1 - x0)) if d > 1e-12 else 0.0
    out = []
    for word, *lens in _dubins_words(_m2pi(h0 - th), _m2pi(h1 - th), d):
        pieces, pose = [], (x0, y0, h0)
        for ch, ln in zip(word, lens):
            if ln * r < 1e-9:
                continue
            k = {"L": 1.0 / r, "R": -1.0 / r, "S": 0.0}[ch]
            seg = _arc(pose, k, ln * r, step=step)
            pieces.append(seg)
            pose = seg.end
        if pieces:
            out.append(join(*pieces))
    out.sort(key=lambda p: p.length)
    return out


def dubins_path(start, goal, radius: Optional[float] = None, step: float = 0.05) -> Optional[Path]:
    """The shortest forward-only Dubins path from ``start`` to ``goal`` (or None)."""
    paths = dubins_paths(start, goal, radius, step)
    return paths[0] if paths else None


# ---------------------------------------------------------------------- #
def to_segments(x, y=None, yaw=None, reverse=None, spacing: float = 0.1) -> List[ReferenceTrajectory]:
    """Build what ``plan()`` returns from sampled poses (a Path, or x, y, yaw arrays).

    Splits wherever the direction of travel changes and resamples every
    ``spacing`` meters (keeping the exact end points), so the output always
    has valid spacing and connected segments.  ``reverse`` (bool or bool
    array) is optional: if you leave it out, the direction of each step is
    worked out from the geometry.
    """
    if isinstance(x, (Path, tuple)) and y is None:
        x, y, yaw, rev_path = _as_path(x)
        reverse = rev_path if reverse is None else reverse
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    yaw = np.unwrap(np.asarray(yaw, dtype=float))
    if reverse is None:
        reverse = _infer_reverse(x, y, yaw)
    rev = np.broadcast_to(np.asarray(reverse, dtype=bool), x.shape).copy()
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate(([True], step > 1e-9))
    x, y, yaw, rev = x[keep], y[keep], yaw[keep], rev[keep]
    if len(x) < 2:
        raise ValueError("to_segments(): the path does not move")
    # the direction of the step into point i is rev[i]; split where it changes
    step_rev = rev[1:]
    cuts = [0] + [i + 1 for i in range(len(step_rev) - 1) if step_rev[i] != step_rev[i + 1]] + [len(x) - 1]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sx, sy, sh = x[a:b + 1], y[a:b + 1], yaw[a:b + 1]
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(sx), np.diff(sy)))))
        n = max(1, int(math.ceil(s[-1] / spacing)))
        si = np.linspace(0.0, s[-1], n + 1)
        tr = ReferenceTrajectory.from_arrays(np.interp(si, s, sx), np.interp(si, s, sy),
                                             np.interp(si, s, sh), np.full(n + 1, 1.0))
        tr.reverse = bool(step_rev[a])
        segs.append(tr)
    return segs


# ---------------------------------------------------------------------- #
class FastChecker:
    """Fast collision checks for many poses at once.

    Builds a distance grid of the lot once (about 0.1 s).  Then each pose is
    first tested with four circles that together cover the car: if all four
    are clear of everything, the car is certainly free; if the middle of the
    car is inside something, it certainly collides.  Only the poses in
    between get the exact rectangle test.  So ``collides`` agrees with
    ``problem.collides``, it is just faster when you check thousands of poses.
    """

    def __init__(self, problem, resolution: float = 0.1):
        self.problem = problem
        self.res = resolution
        xmin, xmax, ymin, ymax = problem.bounds
        self.x0, self.y0 = xmin, ymin
        self.nx = int(math.ceil((xmax - xmin) / resolution)) + 1
        self.ny = int(math.ceil((ymax - ymin) / resolution)) + 1
        X, Y = np.meshgrid(xmin + resolution * np.arange(self.nx),
                           ymin + resolution * np.arange(self.ny), indexing="ij")
        D = np.minimum(np.minimum(X - xmin, xmax - X), np.minimum(Y - ymin, ymax - Y))
        for o in problem.obstacles:
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            u = (X - o.x) * c + (Y - o.y) * s
            v = -(X - o.x) * s + (Y - o.y) * c
            D = np.minimum(D, np.hypot(np.maximum(np.abs(u) - 0.5 * o.length, 0.0),
                                       np.maximum(np.abs(v) - 0.5 * o.width, 0.0)))
        self.D = D
        # Four circles along the car's centerline that cover the whole footprint,
        # and the largest circle at each center that fits INSIDE the footprint.
        length = FRONT + REAR
        self._circ = np.array([-REAR + length * (2 * i + 1) / 8.0 for i in range(4)])
        self._r_out = math.hypot(length / 8.0, HALF_W)
        self._r_in = np.minimum(HALF_W, np.minimum(self._circ + REAR, FRONT - self._circ))
        self._tol = 0.5 * math.hypot(resolution, resolution)    # grid lookup error
        lx, ly = np.linspace(-REAR, FRONT, 16), np.linspace(-HALF_W, HALF_W, 7)
        self._perim = np.concatenate([np.stack([lx, np.full_like(lx, s_)], 1) for s_ in (-HALF_W, HALF_W)]
                                     + [np.stack([np.full_like(ly, e), ly], 1) for e in (-REAR, FRONT)])
        self._boxes = problem._boxes
        if len(self._boxes):
            self._box_c = self._boxes.mean(axis=1)
            self._box_r = np.linalg.norm(self._boxes - self._box_c[:, None, :], axis=-1).max(axis=1)

    def _lookup(self, px, py):
        ix = np.clip(np.rint((px - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(np.rint((py - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return self.D[ix, iy]

    def collides(self, xs, ys, yaws) -> np.ndarray:
        """Bool array: True where the car at that rear-axle pose hits something."""
        xs, ys, yaws = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (xs, ys, yaws))
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        d = self._lookup(xs[:, None] + self._circ * c, ys[:, None] + self._circ * s)   # (N, 4)
        sure_free = np.all(d > self._r_out + self._tol, axis=1)
        sure_hit = np.any(d < self._r_in - self._tol, axis=1)
        out = sure_hit.copy()
        idx = np.nonzero(~sure_free & ~sure_hit)[0]
        if len(idx):
            out[idx] = ~self._exact_free(xs[idx], ys[idx], yaws[idx])
        return out

    def path_is_free(self, x, y=None, yaw=None, spacing: float = 0.025) -> bool:
        """True if the car can drive the whole path (a Path, or x, y, yaw arrays)
        without touching anything.  Also checks BETWEEN your poses, every
        ``spacing`` meters, so a corner cannot slip through a gap."""
        if y is None:
            p = _as_path(x)
            x, y, yaw = p.x, p.y, p.yaw
        x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        yaw = np.unwrap(np.atleast_1d(np.asarray(yaw, dtype=float)))
        if len(x) > 1:
            s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
            if s[-1] > 0:
                si = np.linspace(0.0, s[-1], int(math.ceil(s[-1] / spacing)) + 1)
                x, y, yaw = np.interp(si, s, x), np.interp(si, s, y), np.interp(si, s, yaw)
        return not bool(self.collides(x, y, yaw).any())

    def clearance(self, xs, ys, yaws) -> np.ndarray:
        """Approximate distance [m] from the footprint to the nearest obstacle,
        per pose: 0 wherever the car touches something, otherwise within about
        0.1 m (``problem.clearance_many`` is exact but slower)."""
        xs, ys, yaws = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (xs, ys, yaws))
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        px = xs[:, None] + self._perim[:, 0] * c - self._perim[:, 1] * s
        py = ys[:, None] + self._perim[:, 0] * s + self._perim[:, 1] * c
        d = self._lookup(px, py).min(axis=1)
        # The outline can be clear of something that sits entirely under the
        # car (a cone), so a colliding pose is reported as 0 explicitly.
        d[self.collides(xs, ys, yaws)] = 0.0
        return d

    def _exact_free(self, xs, ys, yaws) -> np.ndarray:
        cars = footprints(xs, ys, yaws)
        xmin, xmax, ymin, ymax = self.problem.bounds
        px, py = cars[..., 0], cars[..., 1]
        ok = ~((px < xmin) | (px > xmax) | (py < ymin) | (py > ymax)).any(axis=1)
        if len(self._boxes) == 0:
            return ok
        cc = cars.mean(axis=1)
        near = np.linalg.norm(cc[:, None, :] - self._box_c[None], axis=-1) < self._box_r[None] + 2.45
        cols = np.nonzero(near.any(axis=0))[0]
        if len(cols):
            ok &= ~(_sat_overlap(cars, self._boxes[cols]) & near[:, cols]).any(axis=1)
        return ok


_CACHE: dict = {}


def get_checker(problem) -> FastChecker:
    """A FastChecker for ``problem``, built the first time and reused after that."""
    hit = _CACHE.get(id(problem))
    if hit is not None and hit[0] is problem:
        return hit[1]
    if len(_CACHE) > 16:
        _CACHE.clear()
    chk = FastChecker(problem)
    _CACHE[id(problem)] = (problem, chk)
    return chk


# ---------------------------------------------------------------------- #
def staging_candidates(problem, n: int = 300, seed: int = 0, min_clearance: float = 0.3,
                       checker: Optional[FastChecker] = None) -> list:
    """Up to ``n`` collision-free poses in the open part of the lot.

    Useful as intermediate targets: when the direct path to the spot hits
    something, try going start -> staging pose -> spot.  Half of the headings
    point roughly toward the spot, the rest are random.  Every pose keeps at
    least ``min_clearance`` meters from everything.  Returned as (x, y, yaw)
    tuples, nearest to the spot first.  Deterministic for a given ``seed``.
    """
    chk = checker or get_checker(problem)
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = problem.bounds
    sp = problem.spot
    out = []
    for _ in range(40):
        m = 4 * n
        xs = rng.uniform(xmin, xmax, m)
        ys = rng.uniform(ymin, ymax, m)
        toward = np.arctan2(sp.y - ys, sp.x - xs)
        yaws = np.where(rng.random(m) < 0.5, toward + rng.uniform(-1.0, 1.0, m), rng.uniform(-np.pi, np.pi, m))
        ok = ~chk.collides(xs, ys, yaws)
        ok[ok] = chk.clearance(xs[ok], ys[ok], yaws[ok]) >= min_clearance
        out += [(float(a), float(b), float(c)) for a, b, c in zip(xs[ok], ys[ok], yaws[ok])]
        if len(out) >= n:
            break
    out = out[:n]
    out.sort(key=lambda p: math.hypot(p[0] - sp.x, p[1] - sp.y))
    return out
