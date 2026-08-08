"""Interpolation methods."""

from black_diamond.solvers.interpolation.lagrange import lagrange_interpolate
from black_diamond.solvers.interpolation.newton_interp import newton_interpolate
from black_diamond.solvers.interpolation.splines import cubic_spline

__all__ = ["lagrange_interpolate", "newton_interpolate", "cubic_spline"]
