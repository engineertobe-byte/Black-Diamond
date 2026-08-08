"""Example: Solving a linear system with Black Diamond."""

import numpy as np

from black_diamond.algorithms import cholesky_solve, gauss_elimination
from black_diamond.applications.math import LinearSolver


def main():
    # Define a symmetric positive-definite matrix
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])

    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    # Using the LinearSolver application
    result = LinearSolver.solve(A, b, method="cholesky")
    print(f"\nSolution (Cholesky): {result.value}")
    print(f"Error estimate: {result.error}")
    print(f"Iterations: {result.iterations}")

    # Using the algorithm directly
    result2 = cholesky_solve(A, b)
    print(f"\nDirect Cholesky solve: {result2.value}")

    # Using Gaussian elimination
    result3 = gauss_elimination(A, b)
    print(f"Gaussian elimination: {result3.value}")

    # Verify the solution
    print(f"\nVerification A @ x - b: {A @ result.value - b}")


if __name__ == "__main__":
    main()