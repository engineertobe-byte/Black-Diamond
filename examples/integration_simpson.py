"""Example: integrate sin(x) with Simpson's rule."""

import math

from black_diamond.solvers.integration import simpson_integrate

result = simpson_integrate(math.sin, 0.0, math.pi, n=100)
print(f"∫₀^π sin(x) dx ≈ {result.value:.10f}")
print(f"Exact value: 2.0")
print(f"Error estimate: {result.error:.2e}")
print(f"Function evaluations: {result.iterations}")
