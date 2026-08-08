"""
Gauss-Seidel iterative method for diagonally dominant linear systems.

Complexity: O(k · n²) per iteration k.
Error order: Linear convergence, typically faster than Jacobi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def gauss_seidel(
    a: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult[FloatArray]:
    """
    Solve A x = b by the Gauss-Seidel iteration using updated components immediately.

    Complexity: O(k · n²)
    Error order: O(ρ^k) linear convergence, ρ typically smaller than Jacobi.
    """
    matrix = np.asarray(a, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if rhs.shape[0] != n:
        raise ValueError(f"Vector length must be {n}.")

    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    x_old = x.copy()

    for iteration in range(1, max_iter + 1):
        for i in range(n):
            if abs(matrix[i, i]) < np.finfo(np.float64).eps:
                raise ValueError("Zero diagonal entry.")
            sigma = float(np.dot(matrix[i, :i], x[:i]) + np.dot(matrix[i, i + 1 :], x[i + 1 :]))
            x[i] = (rhs[i] - sigma) / matrix[i, i]

        step = float(np.linalg.norm(x - x_old, ord=np.inf))
        x_old = x.copy()
        residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
        if step < tol and residual < tol:
            return SolverResult(
                value=x,
                error=max(residual, np.finfo(np.float64).eps),
                iterations=iteration,
                metadata={"step_norm": step, "residual_norm": residual},
            )

    raise RuntimeError(f"Gauss-Seidel did not converge within {max_iter} iterations.")
