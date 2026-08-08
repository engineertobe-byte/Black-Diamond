"""
Classical Runge-Kutta ODE integrators.

Complexity: O(n · s) where n is time steps and s is state dimension.
Error order: O(h⁴) global error for RK4 on smooth problems.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from black_diamond.core.result import SolverResult

FloatArray = NDArray[np.floating]
DerivativeFn = Callable[[float, FloatArray], FloatArray]


def rk4(
    f: DerivativeFn,
    y0: FloatArray,
    t_span: tuple[float, float],
    dt: float,
) -> SolverResult[tuple[FloatArray, FloatArray]]:
    """
    Integrate dy/dt = f(t, y) using the classical 4th-order Runge-Kutta method.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) returning dy/dt.
    y0 : array_like
        Initial state vector.
    t_span : tuple (t0, t1)
        Integration interval.
    dt : float
        Fixed time step (must be > 0).

    Returns
    -------
    SolverResult
        value : tuple (t, y) — time grid and state history, y.shape = (n_steps, dim).
        error : float — embedded RK4 step-difference estimate (Richardson).
        iterations : int — number of RK4 steps taken.

    Complexity: O(n · dim)
    Error order: O(h⁴)
    """
    t0, t1 = t_span
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if t1 <= t0:
        raise ValueError("Require t1 > t0.")

    y = np.asarray(y0, dtype=np.float64).copy()
    n_steps = int(np.ceil((t1 - t0) / dt))
    t_arr = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    y_hist = np.zeros((n_steps + 1, y.size), dtype=np.float64)
    y_hist[0] = y

    max_local_error = 0.0
    iterations = 0

    for i in range(n_steps):
        t = t_arr[i]
        h = t_arr[i + 1] - t

        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y_new = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Half-step estimate for error monitoring
        k1h = f(t, y)
        k2h = f(t + 0.5 * h, y + 0.5 * (h / 2.0) * k1h)
        k3h = f(t + 0.5 * h, y + 0.5 * (h / 2.0) * k2h)
        k4h = f(t + h, y + h * k3h)
        y_half = y + (h / 12.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)

        local_err = float(np.linalg.norm(y_new - y_half, ord=np.inf))
        max_local_error = max(max_local_error, local_err)

        y = y_new
        y_hist[i + 1] = y
        iterations += 1

    error = max(max_local_error, np.finfo(np.float64).eps)

    return SolverResult(
        value=(t_arr, y_hist),
        error=error,
        iterations=iterations,
        metadata={"dt": dt, "n_steps": n_steps, "state_dim": y.size},
    )
