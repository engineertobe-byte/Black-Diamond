"""Tests for classical mechanics models."""

import math

import numpy as np
import pytest

from black_diamond.physics.mechanics import HarmonicOscillator, Pendulum, Projectile


class TestPendulum:
    def test_small_angle_period(self):
        p = Pendulum(L=1.0, g=9.81, theta0=0.01, omega0=0.0)
        result = p.solve(t_max=1.0, dt=0.001)
        assert "theta" in result.value
        assert result.value["theta"].shape == result.value["t"].shape
        assert result.error > 0
        assert abs(p.small_angle_period() - 2 * math.pi * math.sqrt(1.0 / 9.81)) < 1e-10

    def test_energy_conservation_small_angle(self):
        p = Pendulum(L=2.0, g=9.81, theta0=0.05, omega0=0.0)
        result = p.solve(t_max=5.0, dt=0.001)
        theta = result.value["theta"]
        omega = result.value["omega"]
        E = 0.5 * p.L**2 * omega**2 + p.g * p.L * (1 - np.cos(theta))
        assert np.std(E) / np.mean(E) < 0.01

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length"):
            Pendulum(L=-1.0).solve(t_max=1.0)


class TestHarmonicOscillator:
    def test_matches_analytical(self):
        osc = HarmonicOscillator(k=4.0, m=1.0, x0=1.0, v0=0.0)
        result = osc.solve(t_max=2.0, dt=0.001)
        t = result.value["t"]
        x_num = result.value["x"]
        x_exact = osc.analytical(t)
        assert np.max(np.abs(x_num - x_exact)) < 1e-4

    def test_period(self):
        osc = HarmonicOscillator(k=1.0, m=1.0)
        assert abs(osc.omega - 1.0) < 1e-12

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            HarmonicOscillator(k=-1.0).solve(t_max=1.0)


class TestProjectile:
    def test_range(self):
        proj = Projectile(v0=10.0, angle=45.0, g=9.81)
        _, vy0 = proj.launch_velocity
        flight_time = 2.0 * vy0 / proj.g
        result = proj.solve(t_max=flight_time, dt=0.001)
        x_final = result.value["x"][-1]
        assert abs(x_final - proj.range()) < 0.1

    def test_max_height(self):
        proj = Projectile(v0=20.0, angle=90.0, g=9.81)
        result = proj.solve(t_max=5.0, dt=0.001)
        y_max = np.max(result.value["y"])
        assert abs(y_max - proj.max_height()) < 0.1

    def test_trajectory_shape(self):
        proj = Projectile(v0=15.0, angle=30.0)
        result = proj.solve(t_max=3.0, dt=0.01)
        assert len(result.value["x"]) == len(result.value["y"])
