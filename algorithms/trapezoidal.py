"""Trapezoidal integration wrapper for the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.integration.trapezoid import trapezoid_integrate as _trapezoid


def trapezoidal(f: Callable[[float], float], a: float, b: float, n: int = 100) -> Any:
    """Approximate an integral with the composite trapezoidal rule."""
    return _trapezoid(f, a, b, n)