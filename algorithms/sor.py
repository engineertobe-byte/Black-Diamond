"""SOR wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.iterative.sor import sor as _sor


def sor(a: Any, b: Any, x0: Any, omega: float = 1.25, tol: float = 1e-10, max_iter: int = 100) -> Any:
    """Solve A x = b with the SOR method."""
    return _sor(a, b, omega=omega, x0=x0, tol=tol, max_iter=max_iter)
