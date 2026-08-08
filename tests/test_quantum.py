"""Tests for 1-D Schrödinger equation solver."""

import numpy as np
import pytest

from black_diamond.physics.quantum import (
    HarmonicOscillator,
    InfiniteSquareWell,
    schrodinger_1d,
)


class TestSchrodinger1D:
    def test_harmonic_oscillator_energies(self):
        pot = HarmonicOscillator(k=1.0, m=1.0, hbar=1.0)
        result = schrodinger_1d(pot, x_range=(-6, 6), n_points=500, n_states=4)
        E = result.value["energies"]
        for n in range(4):
            exact = pot.energy_level(n)
            assert abs(E[n] - exact) / exact < 0.05

    def test_infinite_square_well(self):
        pot = InfiniteSquareWell(width=1.0)
        result = schrodinger_1d(pot, x_range=(0, 1), n_points=300, n_states=3)
        E = result.value["energies"]
        for n in range(1, 4):
            exact = pot.energy_level(n)
            assert abs(E[n - 1] - exact) / exact < 0.05

    def test_wavefunction_normalization(self):
        pot = HarmonicOscillator(k=1.0)
        result = schrodinger_1d(pot, x_range=(-5, 5), n_points=400, n_states=2)
        x = result.value["x"]
        psi = result.value["wavefunctions"]
        dx = x[1] - x[0]
        for i in range(2):
            norm = np.sum(psi[:, i] ** 2) * dx
            assert abs(norm - 1.0) < 0.01

    def test_custom_potential(self):
        v_fn = lambda x: 0.5 * x**2
        result = schrodinger_1d(v_fn, x_range=(-5, 5), n_points=300, n_states=1)
        assert result.value["energies"][0] > 0
        assert result.error < 1.0

    def test_invalid_domain(self):
        with pytest.raises(ValueError, match="x_max"):
            schrodinger_1d(lambda x: x**2, x_range=(5, -5))

    def test_ground_state_residual(self):
        pot = HarmonicOscillator(k=1.0)
        result = schrodinger_1d(pot, x_range=(-8, 8), n_points=600, n_states=1)
        assert result.error < 0.1
