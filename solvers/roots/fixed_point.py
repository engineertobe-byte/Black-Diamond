"""
Fixed-point iteration x_{n+1} = g(x_n) for root finding.

Complexity: O(k) iterations.
Error order: Linear O(|g'(ξ)|^k) when |g'(x*)| < 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from black_diamond.core.result import SolverResult


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> SolverResult[float]:
    """
    Find a fixed point x* = g(x*) (equivalently a root of x - g(x) = 0).

    Complexity: O(k)
    Error order: O(ρ^k) linear convergence with ρ = |g'(x*)| < 1.
    """
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    x = float(x0)
    for iteration in range(1, max_iter + 1):
        x_new = float(g(x))
        step = abs(x_new - x)
        residual = abs(x_new - float(g(x_new)))
        if step < tol:
            error = max(residual, step, np.finfo(np.float64).eps)
            return SolverResult(
                value=x_new,
                error=error,
                iterations=iteration,
                metadata={"step_size": step, "residual": residual},
            )
        x = x_new

    raise RuntimeError(f"Fixed-point iteration did not converge within {max_iter} iterations.")
