"""Finite-difference derivatives wrapper for the standalone algorithms API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from black_diamond.solvers.differentiation.finite_diff import (
    differentiate as _differentiate,
    DifferenceScheme,
)


def forward_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> Any:
    """Forward difference derivative: O(h) error."""
    return _differentiate(f, x, h, scheme=DifferenceScheme.FORWARD)


def backward_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> Any:
    """Backward difference derivative: O(h) error."""
    return _differentiate(f, x, h, scheme=DifferenceScheme.BACKWARD)


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> Any:
    """Central difference derivative: O(h²) error."""
    return _differentiate(f, x, h, scheme=DifferenceScheme.CENTRAL)


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> Any:
    """Second derivative using central difference: O(h²) error."""
    # Use central difference formula for second derivative
    return _differentiate(
        lambda x: central_difference(f, x, h).value,
        x,
        h,
        scheme=DifferenceScheme.CENTRAL,
    )