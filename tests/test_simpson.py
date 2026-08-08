"""Tests for composite Simpson integration."""

import math

import pytest

from black_diamond.solvers.integration.simpson import simpson_integrate


class TestSimpsonIntegrate:
    def test_polynomial_exact(self):
        # ∫₀¹ x² dx = 1/3 — Simpson is exact for cubics
        result = simpson_integrate(lambda x: x**2, 0.0, 1.0, n=100)
        assert abs(result.value - 1.0 / 3.0) < 1e-10
        assert result.iterations == 101

    def test_sine_integral(self):
        # ∫₀^π sin(x) dx = 2
        result = simpson_integrate(math.sin, 0.0, math.pi, n=200)
        assert abs(result.value - 2.0) < 1e-8
        assert result.error < 1e-6

    def test_gaussian_integral(self):
        # ∫_{-1}^{1} exp(-x²) dx ≈ 1.493648266
        result = simpson_integrate(lambda x: math.exp(-x**2), -1.0, 1.0, n=200)
        expected = 1.493648266
        assert abs(result.value - expected) < 1e-6

    def test_error_decreases_with_n(self):
        f = lambda x: math.exp(x)
        r1 = simpson_integrate(f, 0.0, 1.0, n=20)
        r2 = simpson_integrate(f, 0.0, 1.0, n=200)
        exact = math.e - 1.0
        assert abs(r2.value - exact) < abs(r1.value - exact)

    def test_invalid_n_odd(self):
        with pytest.raises(ValueError, match="even"):
            simpson_integrate(lambda x: x, 0.0, 1.0, n=3)

    def test_invalid_limits(self):
        with pytest.raises(ValueError, match="Lower limit"):
            simpson_integrate(lambda x: x, 1.0, 0.0, n=10)

    def test_metadata(self):
        result = simpson_integrate(lambda x: x, 0.0, 2.0, n=10)
        assert result.metadata["n"] == 10
        assert result.metadata["h"] == pytest.approx(0.2)
