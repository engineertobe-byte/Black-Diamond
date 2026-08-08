"""Numerical integration."""

from black_diamond.solvers.integration.simpson import simpson_integrate
from black_diamond.solvers.integration.trapezoid import trapezoid_integrate

__all__ = ["simpson_integrate", "trapezoid_integrate"]
