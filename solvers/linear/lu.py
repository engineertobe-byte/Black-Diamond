"""
LU decomposition with partial pivoting (Doolittle).

Complexity: O(n³) factorization, O(n²) solve.
Error order: O(ε · cond(A))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult
from black_diamond.solvers.linear.cholesky import condition_number

FloatArray = NDArray[np.floating]


def lu_decompose(a: ArrayLike) -> SolverResult[tuple[FloatArray, FloatArray, NDArray[np.intp]]]:
    """
    Compute PA = LU via Doolittle LU with partial pivoting.

    Returns
    -------
    SolverResult
        value : tuple (L, U, P) where P is permutation indices.
        error : float — ||P A - L U||_F.
        iterations : int — n(n-1)/2 elimination steps.

    Complexity: O(n³)
    """
    matrix = np.asarray(a, dtype=np.float64).copy()
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    perm = np.arange(n)
    lower = np.eye(n, dtype=np.float64)
    iterations = 0

    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(matrix[k:, k])))
        if abs(matrix[pivot, k]) < np.finfo(np.float64).eps:
            raise ValueError("Matrix is singular.")
        if pivot != k:
            matrix[[k, pivot]] = matrix[[pivot, k]]
            lower[[k, pivot], :k] = lower[[pivot, k], :k]
            perm[[k, pivot]] = perm[[pivot, k]]
        for i in range(k + 1, n):
            iterations += 1
            lower[i, k] = matrix[i, k] / matrix[k, k]
            matrix[i, k:] -= lower[i, k] * matrix[k, k:]

    upper = matrix
    p_matrix = np.eye(n)[perm]
    residual = float(np.linalg.norm(p_matrix @ np.asarray(a, dtype=np.float64) - lower @ upper, ord="fro"))
    error = max(residual, np.finfo(np.float64).eps)

    return SolverResult(
        value=(lower, upper, perm),
        error=error,
        iterations=iterations,
        metadata={"residual_norm": residual, "n": n},
    )


def lu_solve(a: ArrayLike, b: ArrayLike) -> SolverResult[FloatArray]:
    """
    Solve A x = b using LU decomposition.

    Complexity: O(n³) factorize + O(n²) solve.
    Error order: O(ε · cond(A))
    """
    decomp = lu_decompose(a)
    lower, upper, perm = decomp.value
    n = lower.shape[0]
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    pb = rhs[perm]

    # Forward: L y = P b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = (pb[i] - float(np.dot(lower[i, :i], y[:i]))) / lower[i, i]

    # Backward: U x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - float(np.dot(upper[i, i + 1 :], x[i + 1 :]))) / upper[i, i]

    matrix = np.asarray(a, dtype=np.float64)
    residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
    kappa = condition_number(matrix)
    error = max(residual * kappa, np.finfo(np.float64).eps)

    return SolverResult(
        value=x,
        error=error,
        iterations=decomp.iterations + 2 * n,
        metadata={"residual_norm": residual, "condition_number": kappa, "n": n},
    )
