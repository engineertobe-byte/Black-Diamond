"""
Classical mechanics: pendulum, harmonic oscillator, projectile motion.

All models are integrated via RK4 (O(h⁴) error, O(n) cost per step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from black_diamond.core.result import SolverResult
from black_diamond.physics.ode import rk4

FloatArray = NDArray[np.floating]


@dataclass
class Pendulum:
    """
    Simple pendulum: θ'' + (g/L) sin(θ) = 0.

    Parameters
    ----------
    L : float
        Pendulum length (m).
    g : float
        Gravitational acceleration (m/s²).
    theta0 : float
        Initial angle (rad).
    omega0 : float
        Initial angular velocity (rad/s).
    """

    L: float
    g: float = 9.81
    theta0: float = 0.1
    omega0: float = 0.0

    def _derivatives(self, t: float, state: FloatArray) -> FloatArray:
        theta, omega = state
        dtheta = omega
        domega = -(self.g / self.L) * np.sin(theta)
        return np.array([dtheta, domega], dtype=np.float64)

    def solve(
        self,
        t_max: float,
        dt: float = 0.01,
    ) -> SolverResult[dict[str, FloatArray]]:
        """
        Integrate pendulum equations of motion.

        Returns
        -------
        SolverResult
            value : dict with keys ``t``, ``theta``, ``omega``.
            error : float — RK4 local error estimate.
            iterations : int — number of time steps.

        Complexity: O(n)
        Error order: O(h⁴)
        """
        if self.L <= 0:
            raise ValueError("Pendulum length L must be positive.")
        y0 = np.array([self.theta0, self.omega0], dtype=np.float64)
        result = rk4(self._derivatives, y0, (0.0, t_max), dt)
        t, y = result.value
        return SolverResult(
            value={"t": t, "theta": y[:, 0], "omega": y[:, 1]},
            error=result.error,
            iterations=result.iterations,
            metadata={"model": "pendulum", "L": self.L, "g": self.g, **(result.metadata or {})},
        )

    def small_angle_period(self) -> float:
        """Analytical period for small angles: T = 2π√(L/g)."""
        return 2.0 * np.pi * np.sqrt(self.L / self.g)


@dataclass
class HarmonicOscillator:
    """
    Harmonic oscillator: x'' + (k/m) x = 0.

    Parameters
    ----------
    k : float
        Spring constant (N/m).
    m : float
        Mass (kg).
    x0 : float
        Initial displacement (m).
    v0 : float
        Initial velocity (m/s).
    """

    k: float
    m: float = 1.0
    x0: float = 1.0
    v0: float = 0.0

    @property
    def omega(self) -> float:
        """Angular frequency ω = √(k/m)."""
        return float(np.sqrt(self.k / self.m))

    def _derivatives(self, t: float, state: FloatArray) -> FloatArray:
        x, v = state
        return np.array([v, -(self.k / self.m) * x], dtype=np.float64)

    def solve(
        self,
        t_max: float,
        dt: float = 0.01,
    ) -> SolverResult[dict[str, FloatArray]]:
        """
        Integrate harmonic oscillator motion.

        Complexity: O(n)
        Error order: O(h⁴)
        """
        if self.k <= 0 or self.m <= 0:
            raise ValueError("Spring constant k and mass m must be positive.")
        y0 = np.array([self.x0, self.v0], dtype=np.float64)
        result = rk4(self._derivatives, y0, (0.0, t_max), dt)
        t, y = result.value
        return SolverResult(
            value={"t": t, "x": y[:, 0], "v": y[:, 1]},
            error=result.error,
            iterations=result.iterations,
            metadata={"model": "harmonic_oscillator", "omega": self.omega, **(result.metadata or {})},
        )

    def analytical(self, t: FloatArray) -> FloatArray:
        """Exact solution x(t) = x₀ cos(ωt) + (v₀/ω) sin(ωt)."""
        t = np.asarray(t, dtype=np.float64)
        return self.x0 * np.cos(self.omega * t) + (self.v0 / self.omega) * np.sin(self.omega * t)


@dataclass
class Projectile:
    """
    Projectile motion with gravity: x'' = 0, y'' = -g.

    Parameters
    ----------
    v0 : float
        Initial speed (m/s).
    angle : float
        Launch angle (degrees).
    g : float
        Gravitational acceleration (m/s²).
    """

    v0: float
    angle: float
    g: float = 9.81

    def _derivatives(self, t: float, state: FloatArray) -> FloatArray:
        x, y, vx, vy = state
        return np.array([vx, vy, 0.0, -self.g], dtype=np.float64)

    @property
    def launch_velocity(self) -> tuple[float, float]:
        """Initial velocity components (vx₀, vy₀)."""
        rad = np.radians(self.angle)
        return self.v0 * np.cos(rad), self.v0 * np.sin(rad)

    def solve(
        self,
        t_max: float | None = None,
        dt: float = 0.01,
    ) -> SolverResult[dict[str, FloatArray]]:
        """
        Integrate projectile trajectory until landing or t_max.

        If t_max is None, integrates until the projectile returns to y = 0.

        Complexity: O(n)
        Error order: O(h⁴)
        """
        vx0, vy0 = self.launch_velocity
        y0 = np.array([0.0, 0.0, vx0, vy0], dtype=np.float64)

        if t_max is None:
            # Analytical flight time: 2 vy₀ / g
            t_max = max(2.0 * vy0 / self.g, dt)

        result = rk4(self._derivatives, y0, (0.0, t_max), dt)
        t, y = result.value
        return SolverResult(
            value={"t": t, "x": y[:, 0], "y": y[:, 1], "vx": y[:, 2], "vy": y[:, 3]},
            error=result.error,
            iterations=result.iterations,
            metadata={"model": "projectile", "angle_deg": self.angle, **(result.metadata or {})},
        )

    def range(self) -> float:
        """Analytical horizontal range: R = v₀² sin(2θ) / g."""
        rad = np.radians(self.angle)
        return self.v0**2 * np.sin(2.0 * rad) / self.g

    def max_height(self) -> float:
        """Analytical maximum height: h = v₀² sin²(θ) / (2g)."""
        rad = np.radians(self.angle)
        return (self.v0 * np.sin(rad)) ** 2 / (2.0 * self.g)
