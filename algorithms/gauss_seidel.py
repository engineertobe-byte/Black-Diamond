"""Gauss-Seidel wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.iterative.gauss_seidel import gauss_seidel as _gauss_seidel


def gauss_seidel(a: Any, b: Any, x0: Any, tol: float = 1e-10, max_iter: int = 100) -> Any:
    """Solve A x = b with the Gauss-Seidel method."""
    return _gauss_seidel(a, b, x0=x0, tol=tol, max_iter=max_iter)
