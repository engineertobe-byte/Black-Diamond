"""
Newton-Raphson root-finding method.

Complexity: O(k · c) where k is iterations and c is cost of f and f'.
Error order: Quadratic convergence O(|e_{n+1}| ≈ C |e_n|²) near a simple root.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from black_diamond.core.result import SolverResult


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> SolverResult[float]:
    """
    Find a root of f(x) = 0 using the Newton-Raphson iteration
    x_{n+1} = x_n - f(x_n) / f'(x_n).

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    df : callable
        Derivative f'(x).
    x0 : float
        Initial guess.
    tol : float, optional
        Convergence tolerance on |f(x)| and step size (default 1e-12).
    max_iter : int, optional
        Maximum number of iterations (default 100).

    Returns
    -------
    SolverResult
        value : float
            Approximate root.
        error : float
            |f(x)| at convergence (residual-based error estimate).
        iterations : int
            Number of Newton iterations performed.
        metadata : dict
            Contains ``residual``, ``step_size``, and ``convergence_rate``.

    Complexity
    ----------
    O(k) iterations, each O(1) for scalar f and f'.

    Error order
    -----------
    Quadratic: |e_{n+1}| ≈ (|f''(ξ)| / (2|f'(x*)|)) |e_n|² near a simple root.
    """
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    x = float(x0)
    prev_error: float | None = None
    convergence_rate: float | None = None

    for iteration in range(1, max_iter + 1):
        fx = float(f(x))
        dfx = float(df(x))

        if abs(dfx) < np.finfo(np.float64).eps:
            raise ValueError(
                f"Derivative near zero at x = {x:.6g}; Newton-Raphson cannot proceed."
            )

        step = fx / dfx
        x_new = x - step
        residual = abs(float(f(x_new)))

        if prev_error is not None and prev_error > np.finfo(np.float64).eps:
            convergence_rate = residual / (prev_error**2)

        if residual < tol and abs(step) < tol:
            error = max(residual, np.finfo(np.float64).eps)
            return SolverResult(
                value=x_new,
                error=error,
                iterations=iteration,
                metadata={
                    "residual": residual,
                    "step_size": abs(step),
                    "convergence_rate": convergence_rate,
                },
            )

        prev_error = abs(x_new - x) if x != x_new else residual
        x = x_new

    raise RuntimeError(
        f"Newton-Raphson did not converge within {max_iter} iterations "
        f"(last residual = {abs(float(f(x))):.6e})."
    )
