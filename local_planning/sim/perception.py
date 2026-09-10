"""Fake perception: turns the true world into noisy ObjectDetections.

Detections are reported in the vehicle frame measured from the center of the
front bumper (x forward, y left), like ``/perception/objectdetection`` on the
car.  The model has the usual problems of a real perception stack: a limited
field of view, position noise that wanders instead of being white, size
jitter, missed frames, and a traffic-light classifier that is sometimes wrong.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from common import vehicle as V
from common.messages import (CarState, LightColor, ObjClass, ObjectArray,
                             ObjectDetection)

from .world import World

MAX_RANGE = 45.0          # [m] from the front bumper
MAX_LATERAL = 15.0        # [m]
HALF_FOV = math.radians(60.0)
DROP_PROB = 0.05
POS_SIGMA_STATIC = 0.10   # [m]
POS_SIGMA_PED = 0.15      # [m]
RANGE_SIGMA_PER_M = 0.003  # extra forward noise per meter of range
NOISE_CORR = 0.7          # AR(1) correlation between frames
SIZE_JITTER = 0.05
LIGHT_WRONG_PROB = 0.03
LIGHT_DROP_PROB = 0.05


def front_bumper(ego: CarState):
    c, s = math.cos(ego.psi), math.sin(ego.psi)
    return ego.x + V.REAR_AXLE_TO_FRONT_BUMPER * c, ego.y + V.REAR_AXLE_TO_FRONT_BUMPER * s


def world_to_vehicle(ego: CarState, wx, wy):
    """World point(s) -> vehicle frame (front-bumper origin, x fwd, y left)."""
    fx, fy = front_bumper(ego)
    c, s = math.cos(ego.psi), math.sin(ego.psi)
    dx, dy = np.asarray(wx) - fx, np.asarray(wy) - fy
    return c * dx + s * dy, -s * dx + c * dy


def visible(xr: float, yr: float) -> bool:
    r = math.hypot(xr, yr)
    return (xr > 0.0 and r <= MAX_RANGE and abs(yr) <= MAX_LATERAL
            and abs(math.atan2(yr, xr)) <= HALF_FOV)


class PerceptionModel:
    """Stateful noise model (the noise is correlated in time)."""

    def __init__(self, world: World, seed: int):
        self.world = world
        self.rng = np.random.default_rng(seed + 7919)
        self._ar: Dict[int, np.ndarray] = {}

    def _ar_noise(self, key: int, sx: float, sy: float) -> np.ndarray:
        w = self.rng.standard_normal(2)
        prev = self._ar.get(key)
        if prev is None:
            n = w
        else:
            n = NOISE_CORR * prev + math.sqrt(1.0 - NOISE_CORR ** 2) * w
        self._ar[key] = n
        return n * np.array([sx, sy])

    def observe(self, t: float, ego: CarState) -> ObjectArray:
        out = []
        rng = self.rng
        for ob in self.world.obstacles:
            cx, cy = world_to_vehicle(ego, ob.corners()[:, 0], ob.corners()[:, 1])
            xc, yc = 0.5 * (cx.min() + cx.max()), 0.5 * (cy.min() + cy.max())
            if not visible(xc, yc):
                continue
            key = 100 + ob.id
            n = self._ar_noise(key, POS_SIGMA_STATIC + RANGE_SIGMA_PER_M * math.hypot(xc, yc),
                               POS_SIGMA_STATIC)
            if rng.random() < DROP_PROB:
                continue
            L = (cx.max() - cx.min()) * (1.0 + SIZE_JITTER * rng.standard_normal())
            W = (cy.max() - cy.min()) * (1.0 + SIZE_JITTER * rng.standard_normal())
            out.append(ObjectDetection(object_id=key, obj_class=ob.obj_class,
                                       x=float(xc + n[0]), y=float(yc + n[1]),
                                       z=0.5 * ob.height, length=float(max(L, 0.1)),
                                       width=float(max(W, 0.1)), height=ob.height))
        for p in self.world.pedestrians:
            px, py = self.world.pedestrian_xy(p, t)
            xr, yr = world_to_vehicle(ego, px, py)
            xr, yr = float(xr), float(yr)
            if not visible(xr, yr):
                continue
            key = 300 + p.id
            n = self._ar_noise(key, POS_SIGMA_PED + RANGE_SIGMA_PER_M * math.hypot(xr, yr),
                               POS_SIGMA_PED)
            if rng.random() < DROP_PROB:
                continue
            out.append(ObjectDetection(object_id=key, obj_class=ObjClass.PEDESTRIAN,
                                       x=xr + float(n[0]), y=yr + float(n[1]), z=0.85,
                                       length=0.5, width=0.5, height=1.7))
        for L in self.world.lights:
            xr, yr = world_to_vehicle(ego, L.x, L.y)
            xr, yr = float(xr), float(yr)
            if not visible(xr, yr) or rng.random() < LIGHT_DROP_PROB:
                continue
            color = L.color(t)
            if rng.random() < LIGHT_WRONG_PROB:
                others = [c for c in (LightColor.RED, LightColor.YELLOW, LightColor.GREEN)
                          if c != color]
                color = others[int(rng.integers(len(others)))]
            out.append(ObjectDetection(object_id=500 + L.id, obj_class=ObjClass.TRAFFIC_LIGHT,
                                       custom_classification=int(color),
                                       x=xr + 0.2 * float(rng.standard_normal()),
                                       y=yr + 0.2 * float(rng.standard_normal()),
                                       z=L.z, length=0.4, width=0.4, height=1.1))
        return ObjectArray(objects=out, stamp=t)
