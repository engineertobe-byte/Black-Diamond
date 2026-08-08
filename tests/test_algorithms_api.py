"""Tests for the standalone algorithms-facing API."""

import math

import numpy as np

from black_diamond.algorithms import cholesky_decompose, cholesky_solve, condition_number, newton_raphson, simpson_integrate
from black_diamond.applications.math import Integrator, LinearSolver, RootFinder


def test_cholesky_decompose_returns_factor():
    a = np.array([[4.0, 2.0], [2.0, 3.0]])
    result = cholesky_decompose(a)
    assert result.value.shape == (2, 2)
    assert np.allclose(result.value @ result.value.T, a, atol=1e-10)


def test_cholesky_solve_matches_known_solution():
    a = np.array([[4.0, 2.0], [2.0, 3.0]])
    b = np.array([8.0, 7.0])
    result = cholesky_solve(a, b)
    assert np.allclose(result.value, np.array([1.25, 1.5]), atol=1e-10)


def test_condition_number_is_finite_for_spd():
    a = np.array([[4.0, 2.0], [2.0, 3.0]])
    assert np.isfinite(condition_number(a))


def test_simpson_integrate_matches_exact_value():
    result = simpson_integrate(math.sin, 0.0, math.pi, n=100)
    assert abs(result.value - 2.0) < 1e-6


def test_newton_raphson_converges_to_root():
    result = newton_raphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, 1.5)
    assert abs(result.value - math.sqrt(2.0)) < 1e-10


def test_linear_solver_wrapper_uses_cholesky():
    result = LinearSolver.solve(np.array([[4.0, 2.0], [2.0, 3.0]]), np.array([8.0, 7.0]))
    assert np.allclose(result.value, np.array([1.25, 1.5]), atol=1e-10)


def test_root_finder_wrapper_uses_newton():
    result = RootFinder.solve(lambda x: x**2 - 2.0, method="newton", df=lambda x: 2.0 * x, x0=1.5)
    assert abs(result.value - math.sqrt(2.0)) < 1e-10


def test_integrator_wrapper_uses_simpson():
    result = Integrator.integrate(math.sin, 0.0, math.pi, method="simpson", n=100)
    assert abs(result.value - 2.0) < 1e-6
