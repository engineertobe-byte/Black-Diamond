"""Standard return type for all Black Diamond solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SolverResult(Generic[T]):
    """
    Unified solver output.

    Attributes
    ----------
    value : Any
        Primary result (scalar, array, tuple, etc.).
    error : float
        Estimated absolute error (tends to 0, never exactly 0).
    iterations : int
        Number of algorithmic steps performed.
    metadata : dict, optional
        Extra diagnostics (residual, condition number, etc.).
    """

    value: T
    error: float
    iterations: int
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        meta = f", metadata={self.metadata}" if self.metadata else ""
        return (
            f"SolverResult(value={self.value!r}, error={self.error:.6e}, "
            f"iterations={self.iterations}{meta})"
        )
