"""
QR decomposition via Gram-Schmidt and least-squares solve.

Complexity: O(n³) for m×n with m ≥ n.
Error order: O(ε · cond(A)) for least-squares.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


def qr_decompose(a: ArrayLike) -> SolverResult[tuple[FloatArray, FloatArray]]:
    """
    Compute the reduced QR factorization A = Q R via modified Gram-Schmidt.

    Parameters
    ----------
    a : array_like, shape (m, n) with m ≥ n

    Returns
    -------
    SolverResult
        value : tuple (Q, R) with Q (m, n) orthonormal and R (n, n) upper triangular.
        error : float — ||A - Q R||_F.
        iterations : int — m·n orthogonalization steps.

    Complexity: O(m n²)
    """
    matrix = np.asarray(a, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional.")
    m, n = matrix.shape
    q = np.zeros((m, n), dtype=np.float64)
    r = np.zeros((n, n), dtype=np.float64)
    iterations = 0

    for j in range(n):
        q[:, j] = matrix[:, j]
        for i in range(j):
            iterations += 1
            r[i, j] = float(np.dot(q[:, i], q[:, j]))
            q[:, j] -= r[i, j] * q[:, i]
        r[j, j] = float(np.linalg.norm(q[:, j]))
        if r[j, j] < np.finfo(np.float64).eps:
            raise ValueError(f"Rank-deficient column at index {j}.")
        q[:, j] /= r[j, j]

    residual = float(np.linalg.norm(matrix - q @ r, ord="fro"))
    error = max(residual, np.finfo(np.float64).eps)

    return SolverResult(
        value=(q, r),
        error=error,
        iterations=iterations,
        metadata={"residual_norm": residual, "m": m, "n": n},
    )


def qr_solve(a: ArrayLike, b: ArrayLike) -> SolverResult[FloatArray]:
    """
    Solve the overdetermined least-squares problem min ||A x - b||₂ via QR.

    Complexity: O(m n²)
    Error order: O(ε · cond(A))
    """
    decomp = qr_decompose(a)
    q, r = decomp.value
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = q.shape
    if rhs.shape[0] != m:
        raise ValueError(f"Vector length must be {m}.")

    qt_b = q.T @ rhs
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (qt_b[i] - float(np.dot(r[i, i + 1 :], x[i + 1 :]))) / r[i, i]

    matrix = np.asarray(a, dtype=np.float64)
    residual = float(np.linalg.norm(matrix @ x - rhs))
    error = max(residual, np.finfo(np.float64).eps)

    return SolverResult(
        value=x,
        error=error,
        iterations=decomp.iterations + n,
        metadata={"residual_norm": residual, "m": m, "n": n},
    )
