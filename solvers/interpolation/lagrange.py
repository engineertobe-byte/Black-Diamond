"""
Lagrange polynomial interpolation.

Complexity: O(n²) per evaluation, O(n) to build coefficients.
Error order: O(h^{n+1}) for n+1 nodes (Runge phenomenon on equispaced nodes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def lagrange_interpolate(
    x_nodes: ArrayLike,
    y_nodes: ArrayLike,
    x_eval: ArrayLike,
) -> SolverResult[FloatArray]:
    """
    Evaluate the Lagrange interpolating polynomial through (x_i, y_i).

    Parameters
    ----------
    x_nodes : array_like, shape (n,)
        Distinct node abscissae.
    y_nodes : array_like, shape (n,)
        Node ordinates.
    x_eval : array_like, shape (m,)
        Points at which to evaluate the interpolant.

    Returns
    -------
    SolverResult
        value : ndarray — interpolated values at x_eval.
        error : float — maximum nodal interpolation residual (0 for exact fit).
        iterations : int — n(n-1)/2 basis evaluations per point.

    Complexity: O(m · n²)
    Error order: O(h^{n+1}) f^{(n+1)} term for smooth f.
    """
    xs = np.asarray(x_nodes, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_nodes, dtype=np.float64).reshape(-1)
    n = xs.shape[0]
    if ys.shape[0] != n:
        raise ValueError("x_nodes and y_nodes must have the same length.")
    if n < 2:
        raise ValueError("At least two nodes required.")
    if len(np.unique(xs)) != n:
        raise ValueError("Nodes must be distinct.")

    xq = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))
    result = np.zeros_like(xq)
    iterations = 0

    for k, x in enumerate(xq):
        value = 0.0
        for i in range(n):
            basis = 1.0
            for j in range(n):
                if j != i:
                    iterations += 1
                    basis *= (x - xs[j]) / (xs[i] - xs[j])
            value += ys[i] * basis
        result[k] = value

    # Nodal fit error (should be ~0)
    nodal = np.zeros(n)
    for k, x in enumerate(xs):
        s = 0.0
        for i in range(n):
            basis = 1.0
            for j in range(n):
                if j != i:
                    basis *= (x - xs[j]) / (xs[i] - xs[j])
            s += ys[i] * basis
        nodal[k] = s
    nodal_error = float(np.max(np.abs(nodal - ys)))
    error = max(nodal_error, np.finfo(np.float64).eps)

    return SolverResult(
        value=result,
        error=error,
        iterations=iterations,
        metadata={"n_nodes": n, "n_eval": xq.shape[0]},
    )
