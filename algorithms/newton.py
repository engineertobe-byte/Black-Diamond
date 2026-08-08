"""Newton-Raphson root finding exposed through the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.roots.newton import newton_raphson as _newton_raphson


def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-12, max_iter: int = 100) -> Any:
    """Find a scalar root using Newton-Raphson."""
    return _newton_raphson(f, df, x0, tol=tol, max_iter=max_iter)
