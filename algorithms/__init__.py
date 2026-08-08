"""Standalone numerical algorithms exposed for classical applications."""

from black_diamond.algorithms.cholesky import cholesky_decompose, cholesky_solve, condition_number
from black_diamond.algorithms.gauss import gauss_elimination
from black_diamond.algorithms.lu import lu_decompose
from black_diamond.algorithms.householder import householder_qr
from black_diamond.algorithms.condition import condition_number as cond_number, error_bound
from black_diamond.algorithms.jacobi import jacobi
from black_diamond.algorithms.gauss_seidel import gauss_seidel
from black_diamond.algorithms.sor import sor
from black_diamond.algorithms.bisection import bisection
from black_diamond.algorithms.newton import newton_raphson
from black_diamond.algorithms.fixed_point import fixed_point
from black_diamond.algorithms.lagrange import lagrange
from black_diamond.algorithms.newton_interp import newton_interp
from black_diamond.algorithms.splines import cubic_splines
from black_diamond.algorithms.derivatives import forward_difference, backward_difference, central_difference, second_derivative
from black_diamond.algorithms.trapezoidal import trapezoidal
from black_diamond.algorithms.simpson import simpson_integrate

__all__ = [
    "cholesky_decompose",
    "cholesky_solve",
    "condition_number",
    "gauss_elimination",
    "lu_decompose",
    "householder_qr",
    "error_bound",
    "jacobi",
    "gauss_seidel",
    "sor",
    "bisection",
    "newton_raphson",
    "fixed_point",
    "lagrange",
    "newton_interp",
    "cubic_splines",
    "forward_difference",
    "backward_difference",
    "central_difference",
    "second_derivative",
    "trapezoidal",
    "simpson_integrate",
]
