"""Example: Root finding with Black Diamond."""

import math

import numpy as np

from black_diamond.algorithms import newton_raphson, bisection
from black_diamond.applications.math import RootFinder


def main():
    # Define the function: cos(x) - x = 0
    def f(x):
        return math.cos(x) - x

    def df(x):
        return -math.sin(x) - 1

    print("Finding root of cos(x) - x = 0")
    print("Expected root: ~0.739085")

    # Using the RootFinder application
    result = RootFinder.solve(f, method="newton", df=df, x0=0.5)
    print(f"\nNewton-Raphson (via RootFinder):")
    print(f"  Root: {result.value}")
    print(f"  Error: {result.error}")
    print(f"  Iterations: {result.iterations}")

    # Using Newton-Raphson directly
    result2 = newton_raphson(f, df, 0.5)
    print(f"\nDirect Newton-Raphson:")
    print(f"  Root: {result2.value}")
    print(f"  Error: {result2.error}")
    print(f"  Iterations: {result2.iterations}")

    # Using bisection
    result3 = bisection(f, 0.0, 1.0)
    print(f"\nBisection:")
    print(f"  Root: {result3.value}")
    print(f"  Error: {result3.error}")
    print(f"  Iterations: {result3.iterations}")

    # Verify
    print(f"\nVerification f(root): {f(result.value)}")
    print(f"Verification f(root): {f(result3.value)}")


if __name__ == "__main__":
    main()