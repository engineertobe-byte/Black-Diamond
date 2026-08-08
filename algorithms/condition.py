"""Condition-number wrappers for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.linear.cholesky import condition_number as _condition_number


def condition_number(a: Any, p: float | str = 2) -> float:
    """Return the matrix condition number."""
    return _condition_number(a, p)


def error_bound(a: Any, b: Any, delta_b: Any, x: Any) -> float:
    """Estimate the relative solution error bound from a perturbation in b."""
    cond = condition_number(a)
    return cond * (float(__import__("numpy").linalg.norm(delta_b)) / float(__import__("numpy").linalg.norm(b)))
