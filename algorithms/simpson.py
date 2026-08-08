"""Composite Simpson integration exposed via the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.integration.simpson import simpson_integrate as _simpson_integrate


def simpson_integrate(f: Callable[[float], float], a: float, b: float, n: int) -> Any:
    """Approximate an integral with composite Simpson's rule."""
    return _simpson_integrate(f, a, b, n)
