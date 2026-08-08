"""
Newton divided-difference interpolation.

Complexity: O(n²) to build table, O(n) per evaluation.
Error order: O(h^{n+1}) for the n-th divided difference remainder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def _divided_differences(x: FloatArray, y: FloatArray) -> FloatArray:
    """Build the Newton divided-difference coefficient table."""
    n = len(x)
    coef = y.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef


def newton_interpolate(
    x_nodes: ArrayLike,
    y_nodes: ArrayLike,
    x_eval: ArrayLike,
) -> SolverResult[FloatArray]:
    """
    Evaluate the Newton form interpolating polynomial.

    Complexity: O(n² + m·n)
    Error order: O(h^{n+1})
    """
    xs = np.asarray(x_nodes, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_nodes, dtype=np.float64).reshape(-1)
    n = xs.shape[0]
    if ys.shape[0] != n or n < 2:
        raise ValueError("Need at least two matching nodes.")
    if len(np.unique(xs)) != n:
        raise ValueError("Nodes must be distinct.")

    coef = _divided_differences(xs, ys)
    xq = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))
    result = np.zeros_like(xq)
    iterations = n * (n - 1) // 2  # table construction cost

    for k, x in enumerate(xq):
        value = coef[-1]
        for i in range(n - 2, -1, -1):
            value = coef[i] + (x - xs[i]) * value
            iterations += 1
        result[k] = value

    nodal_error = float(np.max(np.abs(result[:n] - ys))) if xq.shape[0] >= n else 0.0
    error = max(nodal_error, np.finfo(np.float64).eps)

    return SolverResult(
        value=result,
        error=error,
        iterations=iterations,
        metadata={"n_nodes": n, "leading_coef": float(coef[0])},
    )
