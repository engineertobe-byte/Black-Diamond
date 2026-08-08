"""Example: find √2 with Newton-Raphson."""

import math

from black_diamond.solvers.roots import newton_raphson

f = lambda x: x**2 - 2.0
df = lambda x: 2.0 * x

result = newton_raphson(f, df, x0=1.5)
print(f"√2 ≈ {result.value:.15f}")
print(f"math.sqrt(2) = {math.sqrt(2):.15f}")
print(f"Residual error: {result.error:.2e}")
print(f"Converged in {result.iterations} iterations")
