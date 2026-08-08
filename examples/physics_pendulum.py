"""Example: simulate a simple pendulum."""

from black_diamond.physics.mechanics import Pendulum

pendulum = Pendulum(L=1.0, g=9.81, theta0=0.2, omega0=0.0)
result = pendulum.solve(t_max=10.0, dt=0.01)

t = result.value["t"]
theta = result.value["theta"]
print(f"Pendulum simulated over {t[-1]:.1f} s in {result.iterations} steps")
print(f"Initial angle: {theta[0]:.4f} rad")
print(f"Small-angle period (analytical): {pendulum.small_angle_period():.4f} s")
print(f"RK4 error estimate: {result.error:.2e}")
