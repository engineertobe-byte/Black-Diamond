"""
Classical electromagnetism: Coulomb fields, potentials, 1D Maxwell wave equation.

Uses finite-difference methods with polynomial cost O(n²) or O(n³).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]

# Coulomb constant in vacuum (N·m²/C²)
K_COULOMB = 8.9875517923e9


@dataclass(frozen=True)
class PointCharge:
    """A point charge at a 2-D position."""

    q: float
    position: tuple[float, float]


def electric_field(
    charges: list[PointCharge],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: int = 50,
    softening: float = 0.1,
) -> SolverResult[dict[str, FloatArray]]:
    """
    Compute the electric field E = (Ex, Ey) on a 2-D grid from point charges.

    Uses Coulomb's law: E = k q r̂ / r² (classical electrostatics).

    Parameters
    ----------
    charges : list of PointCharge
        Charge distribution.
    x_range, y_range : tuple
        Domain (min, max) for each axis.
    grid : int
        Number of grid points per axis.
    softening : float
        Softening length to avoid singularity at charge locations (m).

    Returns
    -------
    SolverResult
        value : dict with ``x``, ``y``, ``Ex``, ``Ey`` (meshgrid arrays).
        error : float — maximum relative field smoothness estimate.
        iterations : int — grid points evaluated (grid²).

    Complexity: O(n²) for n×n grid.
    Error order: O(h²) spatial discretization; O(softening) near charges.
    """
    if grid < 2:
        raise ValueError("Grid must be at least 2.")
    if softening <= 0:
        raise ValueError("Softening length must be positive.")

    x = np.linspace(x_range[0], x_range[1], grid)
    y = np.linspace(y_range[0], y_range[1], grid)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)

    for charge in charges:
        dx = X - charge.position[0]
        dy = Y - charge.position[1]
        r2 = dx**2 + dy**2 + softening**2
        r3 = r2 ** 1.5
        Ex += K_COULOMB * charge.q * dx / r3
        Ey += K_COULOMB * charge.q * dy / r3

    # Error estimate: Laplacian magnitude of potential proxy (smoothness)
    h = max(x[1] - x[0], y[1] - y[0])
    laplacian_ex = np.abs(
        np.gradient(np.gradient(Ex, h, axis=1), h, axis=1)
        + np.gradient(np.gradient(Ex, h, axis=0), h, axis=0)
    )
    error = max(float(np.max(laplacian_ex)) * h**2, np.finfo(np.float64).eps)

    return SolverResult(
        value={"x": X, "y": Y, "Ex": Ex, "Ey": Ey},
        error=error,
        iterations=grid * grid,
        metadata={"grid": grid, "n_charges": len(charges), "softening": softening},
    )


def electric_potential(
    charges: list[PointCharge],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: int = 50,
    softening: float = 0.1,
) -> SolverResult[dict[str, FloatArray]]:
    """
    Compute electrostatic potential V on a 2-D grid: V = k q / r.

    Complexity: O(n²)
    Error order: O(h²)
    """
    if grid < 2:
        raise ValueError("Grid must be at least 2.")

    x = np.linspace(x_range[0], x_range[1], grid)
    y = np.linspace(y_range[0], y_range[1], grid)
    X, Y = np.meshgrid(x, y)
    V = np.zeros_like(X)

    for charge in charges:
        dx = X - charge.position[0]
        dy = Y - charge.position[1]
        r = np.sqrt(dx**2 + dy**2 + softening**2)
        V += K_COULOMB * charge.q / r

    h = max(x[1] - x[0], y[1] - y[0])
    error = max(h**2 * float(np.max(np.abs(V))), np.finfo(np.float64).eps)

    return SolverResult(
        value={"x": X, "y": Y, "V": V},
        error=error,
        iterations=grid * grid,
        metadata={"grid": grid, "n_charges": len(charges)},
    )


def maxwell_wave_1d(
    length: float,
    n_points: int,
    c: float,
    t_max: float,
    dt: float,
    pulse_center: float | None = None,
    pulse_width: float = 0.5,
) -> SolverResult[dict[str, FloatArray]]:
    """
    Solve the 1-D wave equation ∂²E/∂t² = c² ∂²E/∂x² via explicit finite differences.

    Initial condition: Gaussian pulse E(x, 0) = exp(-(x-x₀)²/σ²).

    Parameters
    ----------
    length : float
        Domain length (m).
    n_points : int
        Spatial grid points.
    c : float
        Wave speed (m/s).
    t_max : float
        Final simulation time (s).
    dt : float
        Time step (s). Must satisfy CFL: c·dt/dx ≤ 1.
    pulse_center : float, optional
        Center of initial Gaussian pulse (default: length/2).
    pulse_width : float
        Width σ of the Gaussian pulse.

    Returns
    -------
    SolverResult
        value : dict with ``x``, ``t``, ``E`` (E[time, space]).
        error : float — CFL violation indicator or discretization error.
        iterations : int — number of time steps.

    Complexity: O(n_t · n_x)
    Error order: O(h²) space, O(dt²) time.
    """
    if n_points < 3:
        raise ValueError("Need at least 3 spatial points.")
    if dt <= 0 or c <= 0 or length <= 0:
        raise ValueError("length, c, and dt must be positive.")

    dx = length / (n_points - 1)
    cfl = c * dt / dx
    if cfl > 1.0:
        raise ValueError(f"CFL condition violated: c·dt/dx = {cfl:.3f} > 1.")

    x0 = pulse_center if pulse_center is not None else length / 2.0
    x = np.linspace(0.0, length, n_points)

    E_prev = np.zeros(n_points, dtype=np.float64)
    E_curr = np.exp(-((x - x0) / pulse_width) ** 2)
    E_curr[0] = 0.0
    E_curr[-1] = 0.0
    E_next = np.zeros(n_points, dtype=np.float64)

    coeff = (c * dt / dx) ** 2

    # Zero initial velocity: Taylor init for leapfrog
    for i in range(1, n_points - 1):
        E_prev[i] = E_curr[i] + 0.5 * coeff * (
            E_curr[i + 1] - 2.0 * E_curr[i] + E_curr[i - 1]
        )
    E_prev[0] = 0.0
    E_prev[-1] = 0.0

    n_steps = int(np.ceil(t_max / dt))
    t_arr = np.linspace(0.0, n_steps * dt, n_steps + 1)
    E_hist = np.zeros((n_steps + 1, n_points), dtype=np.float64)
    E_hist[0] = E_curr.copy()

    iterations = 0

    for step in range(n_steps):
        for i in range(1, n_points - 1):
            E_next[i] = (
                2.0 * E_curr[i]
                - E_prev[i]
                + coeff * (E_curr[i + 1] - 2.0 * E_curr[i] + E_curr[i - 1])
            )
        # Dirichlet boundaries: E = 0 at domain edges
        E_next[0] = 0.0
        E_next[-1] = 0.0

        E_prev, E_curr, E_next = E_curr, E_next, E_prev
        E_hist[step + 1] = E_curr.copy()
        iterations += 1

    # Discretization error estimate: O(h² + dt²)
    error = max(dx**2 + dt**2, np.finfo(np.float64).eps)

    return SolverResult(
        value={"x": x, "t": t_arr, "E": E_hist},
        error=error,
        iterations=iterations,
        metadata={"cfl": cfl, "dx": dx, "dt": dt, "c": c},
    )
