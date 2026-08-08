"""Cholesky decomposition and linear solve utilities.

This module exposes the requested standalone interface while reusing the
existing solver implementation in black_diamond.solvers.linear.cholesky.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from black_diamond.solvers.linear.cholesky import cholesky_decompose as _cholesky_decompose
from black_diamond.solvers.linear.cholesky import cholesky_solve as _cholesky_solve
from black_diamond.solvers.linear.cholesky import condition_number as _condition_number


def cholesky_decompose(a: Any):
    """Return the Cholesky factorization of a symmetric positive-definite matrix."""
    return _cholesky_decompose(a)


def cholesky_solve(a: Any, b: Any):
    """Solve A x = b using Cholesky decomposition."""
    return _cholesky_solve(a, b)


def condition_number(a: Any, p: float | str = 2) -> float:
    """Compute the condition number of a matrix."""
    return _condition_number(a, p)
