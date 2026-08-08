"""Fixed-point wrapper for the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.roots.fixed_point import fixed_point as _fixed_point


def fixed_point(g: Callable[[float], float], x0: float, tol: float = 1e-10, max_iter: int = 100) -> Any:
    """Find a fixed point of g using fixed-point iteration."""
    return _fixed_point(g, x0, tol=tol, max_iter=max_iter)
