"""Example: Numerical integration with Black Diamond."""

import math

import numpy as np

from black_diamond.algorithms import simpson_integrate, trapezoidal
from black_diamond.applications.math import Integrator


def main():
    # Define the function to integrate: exp(-x^2)
    def f(x):
        return math.exp(-x**2)

    a, b = 0.0, 1.0
    n = 100

    print(f"Integrating exp(-x^2) from {a} to {b}")
    print(f"Number of subintervals: {n}")

    # Using the Integrator application
    result = Integrator.integrate(f, a, b, method="simpson", n=n)
    print(f"\nSimpson result: {result.value}")
    print(f"Error estimate: {result.error}")
    print(f"Function evaluations: {result.iterations}")

    # Using Simpson directly
    result2 = simpson_integrate(f, a, b, n)
    print(f"\nDirect Simpson: {result2.value}")

    # Using trapezoidal rule
    result3 = trapezoidal(f, a, b, n)
    print(f"Trapezoidal: {result3.value}")

    # Compare with known value (approximate)
    # ∫₀¹ exp(-x²) dx ≈ 0.746824
    exact = 0.7468241328124271
    print(f"\nExact value (approx): {exact}")
    print(f"Simpson error: {abs(result.value - exact)}")
    print(f"Trapezoidal error: {abs(result3.value - exact)}")


if __name__ == "__main__":
    main()