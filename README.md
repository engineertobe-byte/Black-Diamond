# Black Diamond

**Numerical solvers for classical mathematics, physics, and chemistry.**

> *You give Black Diamond a classical problem — it returns a solution with controlled error and polynomial cost.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Promise

Black Diamond is an open-source Python library built for scientists and engineers who need **reliable numerical methods** — not black boxes.

Every solver guarantees:

| Property | Guarantee |
|----------|-----------|
| **Error** | Tends to 0 (never exactly 0); quadratic O(h²), cubic O(h³), or higher |
| **Cost** | Polynomial complexity O(n), O(n²), O(n³) — never exponential |
| **Output** | `SolverResult(value, error, iterations)` on every call |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Standalone Algorithms API (Recommended)

```python
import numpy as np
from black_diamond.algorithms import (
    cholesky_solve, simpson_integrate, newton_raphson,
    gauss_elimination, lu_decompose, householder_qr,
    jacobi, gauss_seidel, sor, bisection, fixed_point,
    lagrange, newton_interp, cubic_splines,
    forward_difference, central_difference, second_derivative,
    trapezoidal, condition_number, error_bound
)

# Linear system: A x = b
A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([8.0, 7.0])
result = cholesky_solve(A, b)
print(result)  # SolverResult(value=[1.25 1.5], error=..., iterations=...)

# Gaussian elimination
result = gauss_elimination(A, b)

# LU decomposition
L, U, P = lu_decompose(A).value

# QR decomposition
Q, R = householder_qr(A).value

# Iterative methods
x0 = np.zeros(2)
result = jacobi(A, b, x0, max_iter=1000)
result = gauss_seidel(A, b, x0, max_iter=1000)
result = sor(A, b, x0, omega=1.25, max_iter=1000)

# Root finding
root = newton_raphson(lambda x: x**2 - 2, lambda x: 2*x, x0=1.5)
root = bisection(lambda x: x**3 - x - 2, 1.0, 2.0)
root = fixed_point(lambda x: math.cos(x), 0.5)

# Integration
import math
integral = simpson_integrate(math.sin, 0.0, math.pi, n=100)
integral = trapezoidal(lambda x: x**2, 0.0, 1.0, n=100)

# Interpolation
xp = [0.0, 1.0, 2.0]
yp = [1.0, 2.0, 5.0]
result = lagrange(xp, yp, 1.5)
result = newton_interp(xp, yp, 1.5)
result = cubic_splines(xp, yp, 1.5)

# Differentiation
result = forward_difference(lambda x: x**3, 1.0)
result = central_difference(lambda x: x**3, 1.0)
result = second_derivative(lambda x: x**3, 1.0)

# Condition number and error bounds
kappa = condition_number(A)
bound = error_bound(A, b, np.array([0.1, 0.1]), result.value)
```

### Applications API

```python
from black_diamond.applications.math import LinearSolver, RootFinder, Integrator
from black_diamond.applications.physics import Mechanics, Quantum
from black_diamond.applications.chemistry import Kinetics, Thermodynamics

# Linear solver
result = LinearSolver.solve(A, b, method="cholesky")

# Root finder
result = RootFinder.solve(lambda x: x**2 - 2, method="newton", df=lambda x: 2*x, x0=1.5)

# Integration
result = Integrator.integrate(math.sin, 0.0, math.pi, method="simpson", n=100)

# Physics
t, theta = Mechanics.pendulum(L=1.0, g=9.81, theta0=0.1, t_max=10.0, dt=0.01)
eigenvalues, eigenvectors, x = Quantum.schrodinger_1d(lambda x: 0.5*x**2, (-6, 6), n_points=500)

# Chemistry
t, A = Kinetics.reaction_order1(k=0.1, A0=1.0, t_max=10.0, dt=0.01)
result = Thermodynamics.interpolate_data([300, 400, 500], [1, 2, 3], 350, method="spline")
```

## Architecture

```
black_diamond/
├── algorithms/          # Standalone algorithms API (NEW)
│   ├── cholesky.py      # Cholesky decomposition & solve
│   ├── gauss.py         # Gaussian elimination
│   ├── lu.py            # LU decomposition
│   ├── householder.py   # QR decomposition
│   ├── condition.py     # Condition number & error bounds
│   ├── jacobi.py        # Jacobi iterative method
│   ├── gauss_seidel.py  # Gauss-Seidel iterative method
│   ├── sor.py           # SOR iterative method
│   ├── bisection.py     # Bisection root finding
│   ├── newton.py        # Newton-Raphson root finding
│   ├── fixed_point.py   # Fixed-point iteration
│   ├── lagrange.py      # Lagrange interpolation
│   ├── newton_interp.py # Newton divided differences
│   ├── splines.py       # Cubic splines
│   ├── derivatives.py   # Finite differences
│   ├── trapezoidal.py   # Trapezoidal integration
│   └── simpson.py       # Simpson integration
├── solvers/             # Core solver implementations
│   ├── linear/          # Cholesky, Gauss, LU, QR
│   ├── iterative/       # Jacobi, Gauss-Seidel, SOR
│   ├── roots/           # Bisection, Newton-Raphson, Fixed-point
│   ├── interpolation/   # Lagrange, Newton, Cubic splines
│   ├── differentiation/ # Forward, backward, central differences
│   └── integration/     # Trapezoidal, Simpson
├── applications/        # Classical applications (NEW)
│   ├── math.py          # LinearSolver, RootFinder, Integrator, Interpolator
│   ├── physics.py       # Mechanics, Quantum, Electromagnetism
│   └── chemistry.py     # Kinetics, Thermodynamics
├── physics/             # Physics-specific solvers
├── chemistry/           # Chemistry-specific solvers
├── diamonds/            # Sector-specific applications
├── tests/
└── examples/
```

## Solvers Reference

### Linear Systems — O(n³)

| Function | Method | Error |
|----------|--------|-------|
| `cholesky_decompose` / `cholesky_solve` | Cholesky | O(ε · cond(A)) |
| `gauss_elimination` | Gaussian elimination | O(ε · cond(A)) |
| `lu_decompose` | LU with pivoting | O(ε · cond(A)) |
| `householder_qr` | QR (Gram-Schmidt) | O(ε · cond(A)) |
| `condition_number` | κ(A) = ‖A‖ · ‖A⁻¹‖ | — |
| `error_bound` | A posteriori error bound | — |

### Iterative Methods — O(k · n²)

| Function | Method | Convergence |
|----------|--------|-------------|
| `jacobi` | Jacobi | Linear O(ρᵏ) |
| `gauss_seidel` | Gauss-Seidel | Linear O(ρᵏ) |
| `sor` | SOR | Linear O(ρ_ωᵏ) |

### Root Finding

| Function | Method | Convergence |
|----------|--------|-------------|
| `bisection` | Interval halving | Linear O(2⁻ᵏ) |
| `newton_raphson` | Newton-Raphson | **Quadratic** O(e²) |
| `fixed_point` | Fixed-point | Linear O(ρᵏ) |

### Integration

| Function | Method | Error |
|----------|--------|-------|
| `trapezoidal` | Composite trapezoid | O(h²) |
| `simpson_integrate` | Composite Simpson | **O(h⁴)** |

### Interpolation & Differentiation

| Function | Method | Error |
|----------|--------|-------|
| `lagrange` | Lagrange | O(hⁿ⁺¹) |
| `newton_interp` | Newton divided differences | O(hⁿ⁺¹) |
| `cubic_splines` | Natural cubic spline | O(h⁴) |
| `forward_difference` | Forward difference | O(h) |
| `central_difference` | Central difference | **O(h²)** |
| `second_derivative` | Second derivative | O(h²) |

## Physics Module

```python
from black_diamond.applications.physics import Mechanics, Quantum, Electromagnetism

# Pendulum (finite differences, O(h²))
t, theta = Mechanics.pendulum(L=1.0, g=9.81, theta0=0.1, t_max=10.0, dt=0.01)

# Harmonic oscillator
t, x, v = Mechanics.harmonic_oscillator(k=1.0, m=1.0, x0=1.0, v0=0.0, t_max=10.0, dt=0.01)

# Schrödinger 1D — harmonic oscillator (finite differences, O(n³))
eigenvalues, eigenvectors, x = Quantum.schrodinger_1d(
    lambda x: 0.5 * x**2, (-6, 6), n_points=500
)

# Poisson 2D (placeholder)
V, x, y = Electromagnetism.poisson_2d(lambda x, y: 0, (-1, 1), (-1, 1), grid=50)
```

## Chemistry Module

```python
from black_diamond.applications.chemistry import Kinetics, Thermodynamics

# First-order reaction: A → products
t, A = Kinetics.reaction_order1(k=0.1, A0=1.0, t_max=10.0, dt=0.01)

# Second-order reaction: A + B → products
t, A, B = Kinetics.reaction_order2(k=0.1, A0=1.0, B0=1.0, t_max=10.0, dt=0.01)

# Thermodynamic data interpolation
result = Thermodynamics.interpolate_data(
    [300, 400, 500], [1, 2, 3], 350, method="spline"
)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_algorithms_full.py -v

# Check coverage
pytest --cov=black_diamond tests/
```

## Examples

```bash
# Linear system example
python examples/linear_system.py

# Integration example
python examples/integration.py

# Root finding example
python examples/root_finding.py
```

## Docker

Build the container image:

```bash
docker build -t black-diamond:latest .
```

Run the container in the supported modes:

```bash
docker run black-diamond:latest test
docker run black-diamond:latest example green
docker run -p 3000:3000 black-diamond:latest start --all
docker run -it black-diamond:latest shell
docker run black-diamond:latest version
```

Use `docker-compose.yml` for an optional multi-service stack. The frontend service is provided as an architectural placeholder and requires a `frontend/` project directory to be populated.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

**Black Quantum Diamond Ltd** — founded by Boutaina.
