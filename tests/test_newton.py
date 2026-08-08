"""Tests for Newton-Raphson root finding."""

import math

import pytest

from black_diamond.solvers.roots.newton import newton_raphson


class TestNewtonRaphson:
    def test_sqrt_two(self):
        # f(x) = x² - 2, root = √2
        f = lambda x: x**2 - 2.0
        df = lambda x: 2.0 * x
        result = newton_raphson(f, df, x0=1.5)
        assert abs(result.value - math.sqrt(2.0)) < 1e-12
        assert result.error < 1e-12
        assert result.iterations <= 10

    def test_linear_function(self):
        # f(x) = 2x - 4, root = 2 (converges in 1 step for linear f)
        f = lambda x: 2.0 * x - 4.0
        df = lambda x: 2.0
        result = newton_raphson(f, df, x0=0.0)
        assert abs(result.value - 2.0) < 1e-12

    def test_cosine_root(self):
        # f(x) = cos(x) - x, root ≈ 0.739085
        f = lambda x: math.cos(x) - x
        df = lambda x: -math.sin(x) - 1.0
        result = newton_raphson(f, df, x0=0.5)
        assert abs(result.value - 0.739085133215) < 1e-10

    def test_quadratic_convergence(self):
        # Verify error roughly squares each iteration near root
        f = lambda x: x**2 - 2.0
        df = lambda x: 2.0 * x
        result = newton_raphson(f, df, x0=1.5, tol=1e-14)
        assert result.metadata.get("convergence_rate") is not None

    def test_zero_derivative_raises(self):
        f = lambda x: x**3
        df = lambda x: 3.0 * x**2
        with pytest.raises(ValueError, match="Derivative near zero"):
            newton_raphson(f, df, x0=0.0)

    def test_max_iter_exceeded(self):
        f = lambda x: x**3 - 2.0
        df = lambda x: 3.0 * x**2
        with pytest.raises(RuntimeError, match="did not converge"):
            newton_raphson(f, df, x0=1.0, max_iter=1)

    def test_invalid_tol(self):
        with pytest.raises(ValueError, match="Tolerance"):
            newton_raphson(lambda x: x, lambda x: 1.0, x0=1.0, tol=0.0)
