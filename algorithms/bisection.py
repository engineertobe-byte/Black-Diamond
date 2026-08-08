"""Bisection wrapper for the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.roots.bisection import bisection as _bisection


def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-10, max_iter: int = 100) -> Any:
    """Find a root of f in [a, b] using the bisection method."""
    return _bisection(f, a, b, tol=tol, max_iter=max_iter)
