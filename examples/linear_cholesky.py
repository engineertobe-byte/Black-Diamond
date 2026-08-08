"""Example: solve a linear system with Cholesky."""

import numpy as np

from black_diamond.solvers.linear import cholesky_solve, condition_number

A = np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
b = np.array([12.0, 11.0, 7.0])

result = cholesky_solve(A, b)
print(f"Solution: {result.value}")
print(f"Error estimate: {result.error:.2e}")
print(f"Iterations: {result.iterations}")
print(f"Condition number κ(A): {condition_number(A):.2f}")
