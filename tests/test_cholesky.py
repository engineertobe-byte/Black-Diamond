"""Tests for Cholesky decomposition and condition number."""

import math

import numpy as np
import pytest

from black_diamond.solvers.linear.cholesky import (
    cholesky_decompose,
    cholesky_solve,
    condition_number,
)


class TestCholeskyDecompose:
    def test_2x2_spd(self):
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        result = cholesky_decompose(a)
        lower = result.value
        assert lower.shape == (2, 2)
        assert np.allclose(lower @ lower.T, a)
        assert result.error < 1e-10
        assert result.iterations == 3  # n(n+1)/2 = 3

    def test_3x3_spd(self):
        a = np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
        result = cholesky_decompose(a)
        assert np.allclose(result.value @ result.value.T, a, atol=1e-10)

    def test_not_symmetric_raises(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="symmetric"):
            cholesky_decompose(a)

    def test_not_positive_definite_raises(self):
        a = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="positive definite"):
            cholesky_decompose(a)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            cholesky_decompose(np.array([[1.0, 2.0, 3.0]]))


class TestCholeskySolve:
    def test_solve_known_system(self):
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 7.0])
        result = cholesky_solve(a, b)
        x = result.value
        assert np.allclose(a @ x, b, atol=1e-10)
        assert result.error < 1e-8
        assert "condition_number" in result.metadata

    def test_solve_identity(self):
        n = 5
        a = np.eye(n)
        b = np.arange(1.0, n + 1)
        result = cholesky_solve(a, b)
        assert np.allclose(result.value, b)

    def test_mismatch_vector_length(self):
        a = np.eye(2)
        with pytest.raises(ValueError, match="length"):
            cholesky_solve(a, [1.0, 2.0, 3.0])


class TestConditionNumber:
    def test_identity(self):
        a = np.eye(3)
        assert math.isclose(condition_number(a), 1.0, rel_tol=1e-10)

    def test_diagonal(self):
        a = np.diag([1.0, 10.0, 100.0])
        kappa = condition_number(a)
        assert math.isclose(kappa, 100.0, rel_tol=1e-10)

    def test_ill_conditioned(self):
        # Hilbert matrix is notoriously ill-conditioned
        n = 5
        a = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        kappa = condition_number(a)
        assert kappa > 1e4

    def test_singular_is_inf(self):
        a = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert condition_number(a) == float("inf")
