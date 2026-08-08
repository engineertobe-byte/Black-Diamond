"""Jacobi iterative solver wrapper."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.iterative.jacobi import jacobi as _jacobi


def jacobi(a: Any, b: Any, x0: Any, tol: float = 1e-10, max_iter: int = 100) -> Any:
    """Solve A x = b with the Jacobi method."""
    return _jacobi(a, b, x0=x0, tol=tol, max_iter=max_iter)
