"""
Cholesky decomposition and linear system solver for symmetric positive-definite matrices.

Complexity: O(n³) for decomposition, O(n²) for solve.
Error: Machine precision O(ε) for well-conditioned systems; error grows with cond(A).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def _as_matrix(a: ArrayLike) -> FloatArray:
    """Convert input to 2-D float64 array."""
    matrix = np.asarray(a, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square and two-dimensional.")
    return matrix


def _as_vector(b: ArrayLike, n: int) -> FloatArray:
    """Convert input to 1-D float64 vector of length n."""
    vector = np.asarray(b, dtype=np.float64).reshape(-1)
    if vector.shape[0] != n:
        raise ValueError(f"Vector length must be {n}, got {vector.shape[0]}.")
    return vector


def cholesky_decompose(a: ArrayLike) -> SolverResult[FloatArray]:
    """
    Compute the Cholesky factorization A = L Lᵀ for a symmetric positive-definite matrix.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Symmetric positive-definite matrix.

    Returns
    -------
    SolverResult
        value : ndarray, shape (n, n)
            Lower-triangular Cholesky factor L.
        error : float
            Frobenius norm of residual ||A - L Lᵀ||_F.
        iterations : int
            Number of inner-loop steps (n(n+1)/2).
        metadata : dict
            Contains ``residual_norm`` and ``n``.

    Complexity
    ----------
    O(n³)

    Error order
    -----------
    O(ε · cond(A)) in floating-point arithmetic (ε ≈ machine epsilon).
    """
    matrix = _as_matrix(a)
    n = matrix.shape[0]

    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("Matrix must be symmetric.")

    lower = np.zeros((n, n), dtype=np.float64)
    iterations = 0

    for i in range(n):
        for j in range(i + 1):
            iterations += 1
            if i == j:
                diag_sum = float(np.dot(lower[i, :j], lower[i, :j]))
                pivot = matrix[i, i] - diag_sum
                if pivot <= 0.0:
                    raise ValueError(
                        "Matrix is not positive definite "
                        f"(non-positive pivot at index {i})."
                    )
                lower[i, j] = math.sqrt(pivot)
            else:
                off_sum = float(np.dot(lower[i, :j], lower[j, :j]))
                lower[i, j] = (matrix[i, j] - off_sum) / lower[j, j]

    reconstructed = lower @ lower.T
    residual = float(np.linalg.norm(matrix - reconstructed, ord="fro"))
    error = max(residual, np.finfo(np.float64).eps)

    return SolverResult(
        value=lower,
        error=error,
        iterations=iterations,
        metadata={"residual_norm": residual, "n": n},
    )


def _forward_substitution(lower: FloatArray, b: FloatArray) -> FloatArray:
    """Solve L y = b where L is lower triangular."""
    n = lower.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - float(np.dot(lower[i, :i], y[:i]))) / lower[i, i]
    return y


def _backward_substitution(upper: FloatArray, y: FloatArray) -> FloatArray:
    """Solve U x = y where U is upper triangular."""
    n = upper.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - float(np.dot(upper[i, i + 1 :], x[i + 1 :]))) / upper[i, i]
    return x


def cholesky_solve(a: ArrayLike, b: ArrayLike) -> SolverResult[FloatArray]:
    """
    Solve the linear system A x = b via Cholesky decomposition.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Symmetric positive-definite coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    SolverResult
        value : ndarray, shape (n,)
            Solution vector x.
        error : float
            Estimated absolute error ||A x - b||_∞ scaled by cond(A).
        iterations : int
            Total Cholesky steps plus 2n substitutions.
        metadata : dict
            Contains ``residual_norm``, ``condition_number``, and ``n``.

    Complexity
    ----------
    O(n³) decomposition + O(n²) solve = O(n³) overall.

    Error order
    -----------
    O(ε · cond(A) · ||x||) for well-scaled systems.
    """
    matrix = _as_matrix(a)
    n = matrix.shape[0]
    rhs = _as_vector(b, n)

    decomp = cholesky_decompose(matrix)
    lower = decomp.value

    # Solve L y = b, then Lᵀ x = y
    y = _forward_substitution(lower, rhs)
    x = _backward_substitution(lower.T, y)

    residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
    kappa = condition_number(matrix)
    error = max(residual * kappa, np.finfo(np.float64).eps)

    return SolverResult(
        value=x,
        error=error,
        iterations=decomp.iterations + 2 * n,
        metadata={
            "residual_norm": residual,
            "condition_number": kappa,
            "n": n,
        },
    )


def condition_number(a: ArrayLike, p: float | str = 2) -> float:
    """
    Compute the condition number κ(A) = ||A||_p · ||A⁻¹||_p.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Square matrix.
    p : {None, 1, 2, np.inf, 'fro'}, optional
        Norm order. Default is 2 (spectral norm).

    Returns
    -------
    float
        Condition number (≥ 1). Returns ``inf`` for singular matrices.

    Complexity
    ----------
    O(n³) for p=2 (via SVD); O(n³) for general p via inversion.

    Notes
    -----
    A large condition number indicates ill-conditioning: small perturbations
    in b can cause large changes in x.
    """
    matrix = _as_matrix(a)

    if p == 2:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps * max(singular_values[0], 1.0):
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    norm_a = float(np.linalg.norm(matrix, ord=p))
    try:
        inv_a = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return float("inf")
    norm_inv = float(np.linalg.norm(inv_a, ord=p))
    return norm_a * norm_inv
