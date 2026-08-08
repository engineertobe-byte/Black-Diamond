"""
Jacobi iterative method for diagonally dominant linear systems.

Complexity: O(k · n²) per iteration k.
Error order: Linear convergence O(ρ^k) with ρ = ||D⁻¹(L+U)||.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def jacobi(
    a: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult[FloatArray]:
    """
    Solve A x = b by the Jacobi iteration x^{k+1} = D⁻¹(b - (L+U) x^k).

    Parameters
    ----------
    a : array_like, shape (n, n)
        Coefficient matrix (should be strictly or irreducibly diagonally dominant).
    b : array_like, shape (n,)
        Right-hand side.
    x0 : array_like, optional
        Initial guess (default: zero vector).
    tol : float
        Convergence tolerance on ||x^{k+1} - x^k||_∞.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    SolverResult
        value : ndarray — approximate solution.
        error : float — ||A x - b||_∞.
        iterations : int — Jacobi iterations performed.

    Complexity: O(k · n²)
    Error order: O(ρ^k) linear convergence when spectral radius ρ < 1.
    """
    matrix = np.asarray(a, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if rhs.shape[0] != n:
        raise ValueError(f"Vector length must be {n}.")

    diag = np.diag(matrix)
    if np.any(np.abs(diag) < np.finfo(np.float64).eps):
        raise ValueError("Zero diagonal entry; Jacobi cannot proceed.")

    d_inv = 1.0 / diag
    off_diag = matrix - np.diag(diag)
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    for iteration in range(1, max_iter + 1):
        x_new = d_inv * (rhs - off_diag @ x)
        step = float(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new
        residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
        if step < tol and residual < tol:
            return SolverResult(
                value=x,
                error=max(residual, np.finfo(np.float64).eps),
                iterations=iteration,
                metadata={"step_norm": step, "residual_norm": residual},
            )

    raise RuntimeError(f"Jacobi did not converge within {max_iter} iterations.")
