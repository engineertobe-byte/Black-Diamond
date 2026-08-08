"""
Finite-difference numerical differentiation.

Complexity: O(1) per derivative evaluation.
Error order: O(h²) central, O(h) forward/backward.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np

from black_diamond.core.result import SolverResult


class DifferenceScheme(str, Enum):
    """Finite-difference stencil selection."""

    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


def differentiate(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    scheme: DifferenceScheme | str = DifferenceScheme.CENTRAL,
) -> SolverResult[float]:
    """
    Approximate f'(x) by finite differences.

    Parameters
    ----------
    f : callable
        Differentiable function.
    x : float
        Point of evaluation.
    h : float
        Step size (must be > 0).
    scheme : {'forward', 'backward', 'central'}
        Difference stencil.

    Returns
    -------
    SolverResult
        value : float — derivative estimate.
        error : float — Richardson estimate from h and h/2.
        iterations : int — number of function evaluations.

    Complexity: O(1)
    Error order: O(h²) central, O(h) forward/backward.
    """
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    scheme = DifferenceScheme(scheme)

    if scheme == DifferenceScheme.FORWARD:
        df = (f(x + h) - f(x)) / h
        df_half = (f(x + h / 2) - f(x)) / (h / 2)
        iterations = 3
        error_order = 1
    elif scheme == DifferenceScheme.BACKWARD:
        df = (f(x) - f(x - h)) / h
        df_half = (f(x) - f(x - h / 2)) / (h / 2)
        iterations = 3
        error_order = 1
    else:
        df = (f(x + h) - f(x - h)) / (2.0 * h)
        df_half = (f(x + h / 2) - f(x - h / 2)) / h
        iterations = 4
        error_order = 2

    richardson = abs(df - df_half) / (2**error_order - 1)
    error = max(richardson, np.finfo(np.float64).eps)

    return SolverResult(
        value=float(df),
        error=float(error),
        iterations=iterations,
        metadata={"scheme": scheme.value, "h": h, "richardson_estimate": float(richardson)},
    )
