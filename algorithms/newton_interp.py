"""Newton divided-difference interpolation wrapper."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.interpolation.newton_interp import newton_interpolate as _newton


def newton_interp(x_points: Any, y_points: Any, x: Any) -> Any:
    """Evaluate the Newton form interpolating polynomial at x."""
    return _newton(x_points, y_points, x)