"""Tests for electromagnetism module."""

import numpy as np
import pytest

from black_diamond.physics.electromagnetism import (
    PointCharge,
    electric_field,
    electric_potential,
    maxwell_wave_1d,
)


class TestElectricField:
    def test_single_charge_field_direction(self):
        charges = [PointCharge(q=1.0, position=(0.0, 0.0))]
        result = electric_field(charges, (-1, 1), (-1, 1), grid=11)
        Ex = result.value["Ex"]
        Ey = result.value["Ey"]
        # At (1, 0) field should point in +x
        assert Ex[5, 10] > 0
        assert abs(Ey[5, 10]) < abs(Ex[5, 10])

    def test_dipole_symmetry(self):
        charges = [
            PointCharge(q=1.0, position=(-0.5, 0.0)),
            PointCharge(q=-1.0, position=(0.5, 0.0)),
        ]
        result = electric_field(charges, (-2, 2), (-2, 2), grid=21)
        Ex = result.value["Ex"]
        # Field at origin of dipole should be along +x (positive charge left)
        center_idx = 10
        assert Ex[center_idx, center_idx] > 0

    def test_invalid_grid(self):
        with pytest.raises(ValueError, match="Grid"):
            electric_field([], (0, 1), (0, 1), grid=1)


class TestElectricPotential:
    def test_potential_sign(self):
        charges = [PointCharge(q=1.0, position=(0.0, 0.0))]
        result = electric_potential(charges, (-1, 1), (-1, 1), grid=11)
        V = result.value["V"]
        assert np.all(V > 0)

    def test_superposition(self):
        c1 = [PointCharge(q=1.0, position=(0.0, 0.0))]
        c2 = [PointCharge(q=2.0, position=(0.0, 0.0))]
        V1 = electric_potential(c1, (-1, 1), (-1, 1), grid=11).value["V"]
        V2 = electric_potential(c2, (-1, 1), (-1, 1), grid=11).value["V"]
        assert np.allclose(V2, 2.0 * V1)


class TestMaxwellWave1D:
    def test_cfl_violation(self):
        with pytest.raises(ValueError, match="CFL"):
            maxwell_wave_1d(length=1.0, n_points=50, c=1.0, t_max=1.0, dt=0.1)

    def test_pulse_propagation(self):
        result = maxwell_wave_1d(
            length=20.0, n_points=200, c=1.0, t_max=0.5, dt=0.002,
            pulse_center=10.0, pulse_width=0.5,
        )
        E = result.value["E"]
        assert E.shape[0] > 1
        assert E.shape[1] == 200
        # Initial pulse amplitude ~1; should stay bounded before boundary reflections
        assert np.max(np.abs(E)) < 2.0
        assert np.max(np.abs(E[0])) <= 1.0

    def test_metadata_cfl(self):
        result = maxwell_wave_1d(length=5.0, n_points=50, c=2.0, t_max=1.0, dt=0.01)
        assert result.metadata["cfl"] <= 1.0
