"""Example: electric field from a dipole."""

from black_diamond.physics.electromagnetism import PointCharge, electric_field

charges = [
    PointCharge(q=1e-9, position=(-0.5, 0.0)),
    PointCharge(q=-1e-9, position=(0.5, 0.0)),
]

result = electric_field(charges, x_range=(-2, 2), y_range=(-2, 2), grid=50)
Ex = result.value["Ex"]
Ey = result.value["Ey"]
print(f"Electric field computed on {result.iterations} grid points")
print(f"Max |E|: {(Ex**2 + Ey**2).max()**0.5:.2e} N/C")
print(f"Discretization error estimate: {result.error:.2e}")
