"""
Gaussian elimination with partial pivoting for linear systems.

Complexity: O(n³)
Error order: O(ε · cond(A) · ||x||)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult
from black_diamond.solvers.linear.cholesky import condition_number

FloatArray = NDArray[np.floating]


def gauss_solve(a: ArrayLike, b: ArrayLike) -> SolverResult[FloatArray]:
    """
    Solve A x = b by Gaussian elimination with partial pivoting.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    SolverResult
        value : ndarray — solution x.
        error : float — ||A x - b||_∞ · cond(A).
        iterations : int — number of elimination steps (n(n-1)/2 pivots).

    Complexity: O(n³)
    Error order: O(ε · cond(A))
    """
    aug = np.array(a, dtype=np.float64).copy()
    n = aug.shape[0]
    if aug.shape[0] != aug.shape[1]:
        raise ValueError("Matrix must be square.")
    rhs = np.asarray(b, dtype=np.float64).reshape(-1).copy()
    if rhs.shape[0] != n:
        raise ValueError(f"Vector length must be {n}.")

    iterations = 0
    for col in range(n - 1):
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col]))) 
        if abs(aug[pivot_row, col]) < np.finfo(np.float64).eps:
            raise ValueError("Matrix is singular or numerically singular.")
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
            rhs[[col, pivot_row]] = rhs[[pivot_row, col]]
        for row in range(col + 1, n):
            iterations += 1
            factor = aug[row, col] / aug[col, col]
            aug[row, col:] -= factor * aug[col, col:]
            rhs[row] -= factor * rhs[col]

    x = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        iterations += 1
        if abs(aug[row, row]) < np.finfo(np.float64).eps:
            raise ValueError("Matrix is singular.")
        x[row] = (rhs[row] - float(np.dot(aug[row, row + 1 :], x[row + 1 :]))) / aug[row, row]

    matrix = np.asarray(a, dtype=np.float64)
    residual = float(np.linalg.norm(matrix @ x - np.asarray(b, dtype=np.float64).reshape(-1), ord=np.inf))
    kappa = condition_number(matrix)
    error = max(residual * kappa, np.finfo(np.float64).eps)

    return SolverResult(
        value=x,
        error=error,
        iterations=iterations,
        metadata={"residual_norm": residual, "condition_number": kappa, "n": n},
    )
