"""
Black Diamond — Numerical solvers for classical mathematics, physics, and chemistry.

Every solver returns a structured result with value, error estimate, and iteration count.
All methods guarantee polynomial cost and controlled (non-exponential) error.
"""

__version__ = "0.1.0"
__author__ = "Black Quantum Diamond Ltd"

from black_diamond.core.result import SolverResult

__all__ = ["SolverResult", "__version__"]
