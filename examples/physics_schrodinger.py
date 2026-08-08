"""Example: solve 1-D Schrödinger equation for a harmonic oscillator."""

from black_diamond.physics.quantum import HarmonicOscillator, schrodinger_1d

potential = HarmonicOscillator(k=1.0, m=1.0, hbar=1.0)
result = schrodinger_1d(potential, x_range=(-6, 6), n_points=500, n_states=4)

energies = result.value["energies"]
print("Harmonic oscillator energy levels:")
for n, E in enumerate(energies):
    exact = potential.energy_level(n)
    rel_err = abs(E - exact) / exact * 100
    print(f"  E_{n} = {E:.6f}  (exact: {exact:.6f}, error: {rel_err:.2f}%)")
print(f"Eigenvalue residual: {result.error:.2e}")
