"""
Classical 1-D quantum mechanics via finite-difference eigenvalue solvers.

Solves the time-independent Schrödinger equation
    -ℏ²/(2m) ψ''(x) + V(x) ψ(x) = E ψ(x)
using a tridiagonal Hamiltonian matrix (O(n³) eigensolve, O(h⁴) error on uniform grids).

Note: This is standard numerical quantum mechanics — no exotic quantum concepts exposed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]


@dataclass
class HarmonicOscillator:
    """
    Quantum harmonic oscillator potential V(x) = ½ k x².

    Parameters
    ----------
    k : float
        Spring constant (in atomic units with m = ℏ = 1).
    m : float
        Particle mass.
    hbar : float
        Reduced Planck constant.
    """

    k: float = 1.0
    m: float = 1.0
    hbar: float = 1.0

    def potential(self, x: FloatArray) -> FloatArray:
        """V(x) = ½ k x²."""
        return 0.5 * self.k * x**2

    @property
    def omega(self) -> float:
        """Classical angular frequency ω = √(k/m)."""
        return float(np.sqrt(self.k / self.m))

    def energy_level(self, n: int) -> float:
        """Exact energy E_n = (n + ½) ℏ ω."""
        return (n + 0.5) * self.hbar * self.omega


@dataclass
class InfiniteSquareWell:
    """Infinite square well: V(x) = 0 inside, ∞ outside."""

    width: float = 1.0

    def potential(self, x: FloatArray) -> FloatArray:
        return np.zeros_like(x)

    def energy_level(self, n: int) -> float:
        """E_n = n² π² ℏ² / (2m L²) with ℏ = m = 1."""
        return (n**2) * np.pi**2 / (2.0 * self.width**2)


def _build_hamiltonian(
    x: FloatArray,
    potential: Callable[[FloatArray], FloatArray],
    hbar: float = 1.0,
    mass: float = 1.0,
) -> FloatArray:
    """Build the finite-difference Hamiltonian matrix on a uniform grid."""
    n = x.size
    dx = x[1] - x[0]
    kinetic_coeff = -hbar**2 / (2.0 * mass * dx**2)

    H = np.zeros((n, n), dtype=np.float64)
    V = potential(x)

    for i in range(n):
        H[i, i] = -2.0 * kinetic_coeff + V[i]
        if i > 0:
            H[i, i - 1] = kinetic_coeff
        if i < n - 1:
            H[i, i + 1] = kinetic_coeff

    return H


def schrodinger_1d(
    potential: Callable[[FloatArray], FloatArray] | HarmonicOscillator | InfiniteSquareWell,
    x_range: tuple[float, float],
    n_points: int = 500,
    n_states: int = 5,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> SolverResult[dict[str, FloatArray]]:
    """
    Solve the 1-D time-independent Schrödinger equation by finite differences.

    Parameters
    ----------
    potential : callable or potential object
        V(x) evaluated on the spatial grid.
    x_range : tuple (x_min, x_max)
        Spatial domain.
    n_points : int
        Number of grid points (≥ 10).
    n_states : int
        Number of lowest eigenstates to return.
    hbar, mass : float
        Physical constants (default atomic units).

    Returns
    -------
    SolverResult
        value : dict with ``x``, ``energies``, ``wavefunctions`` (columns = states).
        error : float — eigenvalue residual ||H ψ - E ψ||.
        iterations : int — n_points (matrix dimension).

    Complexity: O(n³) for symmetric eigensolve.
    Error order: O(h²) for second-order finite differences; O(h⁴) with higher-order stencils.
    """
    if n_points < 10:
        raise ValueError("Need at least 10 grid points.")
    if n_states < 1:
        raise ValueError("Need at least 1 state.")
    x_min, x_max = x_range
    if x_max <= x_min:
        raise ValueError("Require x_max > x_min.")

    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]

    if hasattr(potential, "potential"):
        v_fn = potential.potential
    else:
        v_fn = potential

    H = _build_hamiltonian(x, v_fn, hbar=hbar, mass=mass)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    n_return = min(n_states, n_points)
    energies = eigenvalues[:n_return]
    wavefunctions = eigenvectors[:, :n_return]

    # Normalize wavefunctions: ∫|ψ|² dx = 1
    for i in range(n_return):
        norm = float(np.sqrt(np.sum(wavefunctions[:, i] ** 2) * dx))
        if norm > 0:
            wavefunctions[:, i] /= norm

    # Residual error: ||H ψ₀ - E₀ ψ₀||_∞
    psi0 = wavefunctions[:, 0]
    residual = float(np.linalg.norm(H @ psi0 - energies[0] * psi0, ord=np.inf))
    error = max(residual, np.finfo(np.float64).eps)

    return SolverResult(
        value={"x": x, "energies": energies, "wavefunctions": wavefunctions},
        error=error,
        iterations=n_points,
        metadata={"dx": dx, "n_states": n_return, "hbar": hbar, "mass": mass},
    )
