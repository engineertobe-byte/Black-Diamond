"""Natural cubic spline interpolation wrapper."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.interpolation.splines import cubic_spline as _splines


def cubic_splines(x_points: Any, y_points: Any, x: Any) -> Any:
    """Construct and evaluate a natural cubic spline at x."""
    return _splines(x_points, y_points, x)