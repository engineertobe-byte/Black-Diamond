"""
Composite Simpson's rule for numerical integration.

Complexity: O(n) where n is the number of subintervals.
Error order: O(h⁴) on uniformly spaced grids with an even number of subintervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from black_diamond.core.result import SolverResult


def simpson_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> SolverResult[float]:
    """
    Approximate ∫ₐᵇ f(x) dx using the composite Simpson's rule.

    Parameters
    ----------
    f : callable
        Integrand f(x).
    a, b : float
        Integration limits (a < b).
    n : int
        Number of subintervals. Must be even and ≥ 2.

    Returns
    -------
    SolverResult
        value : float
            Integral estimate.
        error : float
            Richardson extrapolation error estimate between n and n/2 panels.
        iterations : int
            Number of function evaluations (n + 1).
        metadata : dict
            Contains ``h`` (step size), ``n``, and ``richardson_estimate``.

    Complexity
    ----------
    O(n) function evaluations.

    Error order
    -----------
    O(h⁴) where h = (b - a) / n, provided f ∈ C⁴[a, b].
    """
    if a >= b:
        raise ValueError("Lower limit a must be strictly less than upper limit b.")
    if n < 2 or n % 2 != 0:
        raise ValueError("Number of subintervals n must be even and at least 2.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(float(xi)) for xi in x], dtype=np.float64)

    # Composite Simpson: (h/3)[y₀ + 4Σy_odd + 2Σy_even + y_n]
    integral = (h / 3.0) * (
        y[0]
        + 4.0 * np.sum(y[1:-1:2])
        + 2.0 * np.sum(y[2:-2:2])
        + y[-1]
    )

    # Richardson extrapolation: compare n panels vs n/2 panels for O(h⁴) error
    n_half = n // 2
    h_half = (b - a) / n_half
    x_half = np.linspace(a, b, n_half + 1)
    y_half = np.array([f(float(xi)) for xi in x_half], dtype=np.float64)
    integral_half = (h_half / 3.0) * (
        y_half[0]
        + 4.0 * np.sum(y_half[1:-1:2])
        + 2.0 * np.sum(y_half[2:-2:2])
        + y_half[-1]
    )

    # Simpson error scales as O(h⁴); Richardson: (I_half - I_n) / 15
    richardson_error = abs(integral_half - integral) / 15.0
    error = max(richardson_error, np.finfo(np.float64).eps)

    return SolverResult(
        value=float(integral),
        error=float(error),
        iterations=n + 1,
        metadata={
            "h": h,
            "n": n,
            "richardson_estimate": float(richardson_error),
        },
    )
