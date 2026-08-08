"""LU decomposition wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.linear.lu import lu_decompose as _lu_decompose


def lu_decompose(a: Any) -> Any:
    """Return the LU decomposition of a matrix as (L, U, P)."""
    return _lu_decompose(a)
