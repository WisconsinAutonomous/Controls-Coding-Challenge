"""Small helpers shared by run.py and check.py."""

from __future__ import annotations

import math
import numbers
import os
import signal
import traceback
from contextlib import contextmanager


class TimeLimit(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Stop the block after ``seconds`` (Unix; elsewhere the caller checks the time afterward)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeLimit()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def describe_exception(exc: BaseException, candidate_file: str) -> str:
    """One readable line: the error, plus where in YOUR file it happened."""
    frames = traceback.extract_tb(exc.__traceback__)
    cand = os.path.abspath(candidate_file)
    mine = [f for f in frames if os.path.abspath(f.filename) == cand]
    msg = f"{type(exc).__name__}: {exc}".strip()
    if mine:
        f = mine[-1]
        where = f"{os.path.basename(f.filename)} line {f.lineno}, in {f.name}()"
        if f.line:
            where += f": {f.line.strip()}"
        return f"{msg}  [at {where}]"
    if frames:
        f = frames[-1]
        return f"{msg}  [at {os.path.basename(f.filename)} line {f.lineno}, in {f.name}()]"
    return msg


def strict_json(obj):
    """Copy of ``obj`` that is valid strict JSON: NaN and inf become None."""
    if isinstance(obj, dict):
        return {str(k): strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return f if math.isfinite(f) else None
