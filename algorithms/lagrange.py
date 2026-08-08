"""Lagrange interpolation wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.interpolation.lagrange import lagrange_interpolate as _lagrange


def lagrange(x_points: Any, y_points: Any, x: Any) -> Any:
    """Evaluate the Lagrange interpolating polynomial at x."""
    return _lagrange(x_points, y_points, x)