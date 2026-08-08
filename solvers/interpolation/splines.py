"""
Natural cubic spline interpolation.

Complexity: O(n) to solve tridiagonal system, O(log n) per evaluation.
Error order: O(h⁴) for natural cubic splines on uniform grids.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def cubic_spline(
    x_nodes: ArrayLike,
    y_nodes: ArrayLike,
    x_eval: ArrayLike,
) -> SolverResult[FloatArray]:
    """
    Construct and evaluate a natural cubic spline S(x) with S''(x₀) = S''(x_n) = 0.

    Complexity: O(n) setup, O(m log n) evaluation via search.
    Error order: O(h⁴) for smooth functions on uniform meshes.
    """
    xs = np.asarray(x_nodes, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_nodes, dtype=np.float64).reshape(-1)
    n = xs.shape[0] - 1  # number of intervals
    if ys.shape[0] != n + 1 or n < 1:
        raise ValueError("Need at least two nodes.")
    if not np.all(np.diff(xs) > 0):
        raise ValueError("Nodes must be strictly increasing.")

    h = np.diff(xs)
    # Tridiagonal system for second derivatives M
    alpha = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (ys[i + 1] - ys[i]) - (3.0 / h[i - 1]) * (ys[i] - ys[i - 1])

    l = np.ones(n + 1, dtype=np.float64)
    mu = np.zeros(n + 1, dtype=np.float64)
    z = np.zeros(n + 1, dtype=np.float64)
    iterations = 0

    for i in range(1, n):
        iterations += 1
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    m = np.zeros(n + 1, dtype=np.float64)
    for j in range(n - 1, -1, -1):
        iterations += 1
        m[j] = z[j] - mu[j] * m[j + 1]

    xq = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))
    result = np.zeros_like(xq)

    for k, x in enumerate(xq):
        if x < xs[0] or x > xs[-1]:
            raise ValueError(f"Evaluation point {x} outside spline domain [{xs[0]}, {xs[-1]}].")
        idx = int(np.searchsorted(xs, x) - 1)
        idx = min(max(idx, 0), n - 1)
        dx = x - xs[idx]
        hi = h[idx]
        a = ys[idx]
        b = (ys[idx + 1] - ys[idx]) / hi - hi * (2.0 * m[idx] + m[idx + 1]) / 6.0
        c = m[idx] / 2.0
        d = (m[idx + 1] - m[idx]) / (6.0 * hi)
        result[k] = a + b * dx + c * dx**2 + d * dx**3
        iterations += 1

    # Nodal error (only at points that are both evaluation and nodal points)
    mask = np.isin(xq, xs)
    if np.any(mask):
        nodal_err = float(np.max(np.abs(result[mask] - ys[np.isin(xs, xq)])))
    else:
        nodal_err = 0.0
    error = max(nodal_err, np.finfo(np.float64).eps)

    return SolverResult(
        value=result,
        error=error,
        iterations=iterations,
        metadata={"n_intervals": n, "n_eval": xq.shape[0]},
    )
