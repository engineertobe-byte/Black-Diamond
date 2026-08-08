"""Root-finding methods."""

from black_diamond.solvers.roots.bisection import bisection
from black_diamond.solvers.roots.fixed_point import fixed_point
from black_diamond.solvers.roots.newton import newton_raphson

__all__ = ["bisection", "fixed_point", "newton_raphson"]
