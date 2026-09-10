"""Speed Control scenarios.  Every road is straight and ends with a stop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

from .car import NOMINAL, CarParams
from .road import LeadCar, Stop


@dataclass
class Scenario:
    name: str
    description: str
    length: float
    speed_limit: Callable[[float], float]
    stops: List[Stop] = field(default_factory=list)
    v0: float = 0.0
    grade: Optional[Callable[[float], float]] = None
    lead: Optional[LeadCar] = None
    params: CarParams = field(default_factory=lambda: NOMINAL)
    core: bool = True
    par_time: Optional[float] = None    # used instead of the plan's nominal time (follow_the_leader)


def _zones(default: float, zones) -> Callable[[float], float]:
    def f(x: float) -> float:
        for x0, x1, v in zones:
            if x0 <= x < x1:
                return v
        return default
    return f


def _hill(x: float) -> float:
    pts = [(0, 0.0), (40, 0.0), (50, 0.08), (130, 0.08), (150, -0.08), (210, -0.08), (220, 0.0)]
    if x <= pts[0][0] or x >= pts[-1][0]:
        return 0.0
    for (x0, g0), (x1, g1) in zip(pts[:-1], pts[1:]):
        if x0 <= x < x1:
            return g0 + (g1 - g0) * (x - x0) / (x1 - x0)
    return 0.0


def cruise() -> Scenario:
    return Scenario("cruise", "Speed limits of 8, 13, 5 and 10 m/s, then stop at the end.", 600.0,
                    _zones(10.0, [(0, 150, 8.0), (150, 350, 13.0), (350, 480, 5.0)]))


def stop_and_go() -> Scenario:
    return Scenario("stop_and_go", "A stop sign (hold 3 s) and a red light (hold 5 s), then stop at the end.",
                    260.0, _zones(10.0, []), [Stop(90.0, 3.0, "stop sign"), Stop(180.0, 5.0, "red light")])


def hill() -> Scenario:
    return Scenario("hill", "8% climb with a stop sign halfway up, then an 8% descent.", 260.0,
                    _zones(8.0, []), [Stop(100.0, 3.0, "stop sign")], grade=_hill)


def follow_the_leader() -> Scenario:
    lead = LeadCar(x0=3.7 + 25.0, schedule=[(0, 8.0), (8, 12.0), (20, 12.0), (23, 3.0), (30, 3.0),
                                            (32, 0.0), (38, 0.0), (44, 10.0), (200, 10.0)],
                   leaves_at=560.0)
    return Scenario("follow_the_leader",
                    "Adaptive cruise: a car ahead speeds up, brakes hard, stops at a light and goes. Keep a safe gap.",
                    620.0, _zones(15.0, []), lead=lead, v0=8.0, core=False, par_time=75.0)


def loaded_car() -> Scenario:
    heavy = replace(NOMINAL, throttle_gain=0.8, brake_gain=0.85, rolling_decel=0.24,
                    accel_tau=0.42, accel_delay=0.14)
    return Scenario("loaded_car", "The hill again, but the car carries 400 kg of cargo and responds slower.",
                    260.0, _zones(8.0, []), [Stop(100.0, 3.0, "stop sign")], grade=_hill,
                    params=heavy, core=False)


SCENARIOS: Dict[str, Callable[[], Scenario]] = {
    "cruise": cruise,
    "stop_and_go": stop_and_go,
    "hill": hill,
    "follow_the_leader": follow_the_leader,
    "loaded_car": loaded_car,
}
