"""
Composite trapezoidal rule for numerical integration.

Complexity: O(n)
Error order: O(h²)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from black_diamond.core.result import SolverResult


def trapezoid_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> SolverResult[float]:
    """
    Approximate ∫ₐᵇ f(x) dx using the composite trapezoidal rule.

    Complexity: O(n)
    Error order: O(h²) where h = (b-a)/n.
    """
    if a >= b:
        raise ValueError("Lower limit a must be strictly less than upper limit b.")
    if n < 1:
        raise ValueError("Number of subintervals n must be at least 1.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(float(xi)) for xi in x], dtype=np.float64)

    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    # Richardson: trapezoid error O(h²), compare n vs n/2
    if n >= 2:
        n_half = n // 2
        h2 = (b - a) / n_half
        x2 = np.linspace(a, b, n_half + 1)
        y2 = np.array([f(float(xi)) for xi in x2], dtype=np.float64)
        integral_half = h2 * (0.5 * y2[0] + np.sum(y2[1:-1]) + 0.5 * y2[-1])
        richardson_error = abs(integral_half - integral) / 3.0
    else:
        richardson_error = h**2

    error = max(richardson_error, np.finfo(np.float64).eps)

    return SolverResult(
        value=float(integral),
        error=float(error),
        iterations=n + 1,
        metadata={"h": h, "n": n, "richardson_estimate": float(richardson_error)},
    )
