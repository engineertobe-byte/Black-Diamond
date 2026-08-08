"""
Successive Over-Relaxation (SOR) for linear systems.

Complexity: O(k · n²) per iteration k.
Error order: O(ρ_ω^k) with optimal ω accelerating convergence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def sor(
    a: ArrayLike,
    b: ArrayLike,
    omega: float = 1.25,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult[FloatArray]:
    """
    Solve A x = b by SOR with relaxation parameter ω ∈ (0, 2).

    Parameters
    ----------
    omega : float
        Relaxation factor. ω = 1 recovers Gauss-Seidel; ω > 1 over-relaxation.

    Complexity: O(k · n²)
    Error order: O(ρ_ω^k) linear convergence.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("Relaxation parameter omega must be in (0, 2).")

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
            gs_step = (rhs[i] - sigma) / matrix[i, i]
            x[i] = (1.0 - omega) * x[i] + omega * gs_step

        step = float(np.linalg.norm(x - x_old, ord=np.inf))
        x_old = x.copy()
        residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
        if step < tol and residual < tol:
            return SolverResult(
                value=x,
                error=max(residual, np.finfo(np.float64).eps),
                iterations=iteration,
                metadata={"step_norm": step, "residual_norm": residual, "omega": omega},
            )

    raise RuntimeError(f"SOR did not converge within {max_iter} iterations.")
