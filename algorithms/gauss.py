"""Gaussian elimination wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.linear.gauss import gauss_solve as _gauss_solve


def gauss_elimination(a: Any, b: Any) -> Any:
    """Solve A x = b using Gaussian elimination with partial pivoting."""
    return _gauss_solve(a, b)
