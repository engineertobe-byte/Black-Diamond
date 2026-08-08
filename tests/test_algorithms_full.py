"""Comprehensive tests for all Black Diamond algorithms."""

import math

import numpy as np
import pytest

from black_diamond.algorithms import (
    bisection,
    central_difference,
    cholesky_decompose,
    cholesky_solve,
    condition_number,
    cubic_splines,
    error_bound,
    fixed_point,
    forward_difference,
    backward_difference,
    gauss_elimination,
    gauss_seidel,
    householder_qr,
    jacobi,
    lagrange,
    lu_decompose,
    newton_interp,
    newton_raphson,
    second_derivative,
    simpson_integrate,
    sor,
    trapezoidal,
)


class TestAllAlgorithms:
    """Test all algorithms in the Black Diamond library."""

    def test_cholesky_decompose(self):
        """Test Cholesky decomposition."""
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        result = cholesky_decompose(A)
        assert result.value.shape == (2, 2)
        assert np.allclose(result.value @ result.value.T, A, atol=1e-10)

    def test_cholesky_solve(self):
        """Test Cholesky linear solve."""
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 7.0])
        result = cholesky_solve(A, b)
        assert np.allclose(result.value, np.array([1.25, 1.5]), atol=1e-10)

    def test_condition_number(self):
        """Test condition number computation."""
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        cond = condition_number(A)
        assert np.isfinite(cond)
        assert cond >= 1.0

    def test_error_bound(self):
        """Test error bound estimation."""
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 7.0])
        delta_b = np.array([0.1, 0.1])
        x = np.array([1.25, 1.5])
        bound = error_bound(A, b, delta_b, x)
        assert np.isfinite(bound)
        assert bound >= 0.0

    def test_gauss_elimination(self):
        """Test Gaussian elimination."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])
        result = gauss_elimination(A, b)
        assert np.allclose(result.value, [1.0, 1.0], atol=1e-10)

    def test_lu_decompose(self):
        """Test LU decomposition."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = lu_decompose(A)
        L, U, P = result.value
        PA = A[P]
        assert np.allclose(L @ U, PA, atol=1e-10)

    def test_householder_qr(self):
        """Test QR decomposition."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = householder_qr(A)
        Q, R = result.value
        assert np.allclose(Q @ R, A, atol=1e-10)
        # Check Q is orthogonal
        assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-10)

    def test_jacobi(self):
        """Test Jacobi iterative method."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)
        result = jacobi(A, b, x0, max_iter=1000)
        expected = np.linalg.solve(A, b)
        assert np.allclose(result.value, expected, rtol=1e-3)

    def test_gauss_seidel(self):
        """Test Gauss-Seidel iterative method."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)
        result = gauss_seidel(A, b, x0, max_iter=1000)
        expected = np.linalg.solve(A, b)
        assert np.allclose(result.value, expected, rtol=1e-3)

    def test_sor(self):
        """Test SOR iterative method."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)
        result = sor(A, b, x0, omega=1.25, max_iter=1000)
        expected = np.linalg.solve(A, b)
        assert np.allclose(result.value, expected, rtol=1e-3)

    def test_bisection(self):
        """Test bisection root finding."""
        def f(x):
            return x**3 - x - 2
        result = bisection(f, 1.0, 2.0)
        assert abs(result.value - 1.5213797068045676) < 1e-3

    def test_fixed_point(self):
        """Test fixed-point iteration."""
        def g(x):
            return math.cos(x)
        result = fixed_point(g, 0.5)
        assert abs(result.value - 0.7390851332151607) < 1e-3

    def test_newton_raphson(self):
        """Test Newton-Raphson root finding."""
        def f(x):
            return x**2 - 2.0
        def df(x):
            return 2.0 * x
        result = newton_raphson(f, df, 1.5)
        assert abs(result.value - math.sqrt(2.0)) < 1e-10

    def test_lagrange(self):
        """Test Lagrange interpolation."""
        xp = [0.0, 1.0, 2.0]
        yp = [1.0, 2.0, 5.0]
        result = lagrange(xp, yp, 1.5)
        assert abs(result.value[0] - 3.25) < 1e-10

    def test_newton_interp(self):
        """Test Newton divided-difference interpolation."""
        xp = [0.0, 1.0, 2.0]
        yp = [1.0, 2.0, 5.0]
        result = newton_interp(xp, yp, 1.5)
        assert abs(result.value[0] - 3.25) < 1e-10

    def test_cubic_splines(self):
        """Test cubic spline interpolation."""
        xp = [0.0, 1.0, 2.0]
        yp = [1.0, 2.0, 5.0]
        result = cubic_splines(xp, yp, 1.5)
        # Cubic spline gives different value than polynomial interpolation
        # Just verify it evaluates and is reasonable
        assert np.isfinite(result.value[0])
        # At nodal points, spline should match exactly
        result_nodal = cubic_splines(xp, yp, xp)
        assert np.allclose(result_nodal.value, yp, atol=1e-10)

    def test_forward_difference(self):
        """Test forward difference derivative."""
        def f(x):
            return x**3
        result = forward_difference(f, 1.0)
        assert abs(result.value - 3.0) < 1e-4

    def test_backward_difference(self):
        """Test backward difference derivative."""
        def f(x):
            return x**3
        result = backward_difference(f, 1.0)
        assert abs(result.value - 3.0) < 1e-4

    def test_central_difference(self):
        """Test central difference derivative."""
        def f(x):
            return x**3
        result = central_difference(f, 1.0)
        assert abs(result.value - 3.0) < 1e-6

    def test_second_derivative(self):
        """Test second derivative."""
        def f(x):
            return x**3
        result = second_derivative(f, 1.0)
        assert abs(result.value - 6.0) < 1e-4

    def test_simpson_integrate(self):
        """Test Simpson integration."""
        result = simpson_integrate(math.sin, 0.0, math.pi, n=100)
        assert abs(result.value - 2.0) < 1e-6

    def test_trapezoidal(self):
        """Test trapezoidal integration."""
        def f(x):
            return x**2
        result = trapezoidal(f, 0.0, 1.0, n=1000)
        assert abs(result.value - 1.0/3.0) < 1e-4


class TestApplications:
    """Test application modules."""

    def test_linear_solver(self):
        """Test LinearSolver application."""
        from black_diamond.applications.math import LinearSolver
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 7.0])
        result = LinearSolver.solve(A, b, method="cholesky")
        assert np.allclose(result.value, np.array([1.25, 1.5]), atol=1e-10)

    def test_root_finder(self):
        """Test RootFinder application."""
        from black_diamond.applications.math import RootFinder
        def f(x):
            return x**2 - 2.0
        def df(x):
            return 2.0 * x
        result = RootFinder.solve(f, method="newton", df=df, x0=1.5)
        assert abs(result.value - math.sqrt(2.0)) < 1e-10

    def test_integrator(self):
        """Test Integrator application."""
        from black_diamond.applications.math import Integrator
        result = Integrator.integrate(math.sin, 0.0, math.pi, method="simpson", n=100)
        assert abs(result.value - 2.0) < 1e-6

    def test_mechanics_pendulum(self):
        """Test pendulum simulation."""
        from black_diamond.applications.physics import Mechanics
        t, theta = Mechanics.pendulum(1.0, 9.81, 0.1, 1.0, dt=0.01)
        assert len(t) == len(theta)
        assert theta[0] == 0.1

    def test_kinetics_order1(self):
        """Test first-order reaction kinetics."""
        from black_diamond.applications.chemistry import Kinetics
        t, A = Kinetics.reaction_order1(0.1, 1.0, 1.0, dt=0.01)
        assert len(t) == len(A)
        assert A[0] == 1.0
        # Should decay exponentially
        assert A[-1] < A[0]

    def test_thermodynamics_interpolation(self):
        """Test thermodynamics interpolation."""
        from black_diamond.applications.chemistry import Thermodynamics
        T_data = [300.0, 400.0, 500.0]
        P_data = [1.0, 2.0, 3.0]
        result = Thermodynamics.interpolate_data(T_data, P_data, 350.0, method="linear")
        assert abs(result - 1.5) < 1e-10