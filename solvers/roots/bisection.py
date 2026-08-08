"""
Bisection method for root finding on bracketed intervals.

Complexity: O(log((b-a)/tol)) iterations, O(1) per step.
Error order: Linear convergence O(2^{-k}) — halving interval each step.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from black_diamond.core.result import SolverResult


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> SolverResult[float]:
    """
    Find a root of f in [a, b] where f(a)·f(b) < 0.

    Complexity: O(log((b-a)/tol))
    Error order: O(h_k) where h_k = (b-a)/2^k (linear / bracket halving).
    """
    fa = float(f(a))
    fb = float(f(b))
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs (root not bracketed).")
    if a >= b:
        raise ValueError("Require a < b.")
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    left, right = float(a), float(b)
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (left + right)
        fmid = float(f(mid))
        interval = right - left

        if abs(fmid) < tol or interval / 2.0 < tol:
            error = max(abs(fmid), interval / 2.0, np.finfo(np.float64).eps)
            return SolverResult(
                value=mid,
                error=error,
                iterations=iteration,
                metadata={"residual": abs(fmid), "interval_width": interval},
            )

        if fa * fmid < 0:
            right = mid
            fb = fmid
        else:
            left = mid
            fa = fmid

    mid = 0.5 * (left + right)
    return SolverResult(
        value=mid,
        error=max(abs(float(f(mid))), np.finfo(np.float64).eps),
        iterations=max_iter,
        metadata={"residual": abs(float(f(mid))), "interval_width": right - left},
    )
