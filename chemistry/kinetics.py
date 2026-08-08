"""Chemical kinetics and reaction system integration.

This module provides mass-action reaction modeling using classical ODE solvers.
The default integrator is RK4 with O(h^4) error. An improved Euler method is
available for lower-order benchmarking.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from black_diamond.core.result import SolverResult
from black_diamond.physics.ode import rk4

FloatArray = NDArray[np.floating]
DerivativeFn = Callable[[float, FloatArray], FloatArray]

_COEFFICIENT_PATTERN = re.compile(r"^\s*(?P<coef>\d+)?\s*(?P<species>[A-Za-z][A-Za-z0-9_]*)\s*$")


def _parse_side(side: str) -> dict[str, int]:
    """Parse one side of a reaction equation into stoichiometric coefficients."""
    if not side.strip():
        return {}

    terms = [term.strip() for term in side.split("+")]
    species_counts: dict[str, int] = {}
    for term in terms:
        match = _COEFFICIENT_PATTERN.match(term)
        if not match:
            raise ValueError(f"Invalid reaction term: '{term}'")
        coef = int(match.group("coef")) if match.group("coef") else 1
        species = match.group("species")
        species_counts[species] = species_counts.get(species, 0) + coef
    return species_counts


def _parse_equation(equation: str) -> tuple[dict[str, int], dict[str, int]]:
    """Parse a reaction equation string into reactants and products."""
    if "->" not in equation:
        raise ValueError("Reaction equation must contain '->'.")
    left, right = equation.split("->", 1)
    reactants = _parse_side(left)
    products = _parse_side(right)
    if not reactants and not products:
        raise ValueError("Empty reaction equation.")
    return reactants, products


def _build_state_vector(
    species_order: Sequence[str],
    concentrations: dict[str, float],
) -> FloatArray:
    return np.asarray([float(concentrations.get(species, 0.0)) for species in species_order], dtype=np.float64)


def _euler_improved(
    f: DerivativeFn,
    y0: FloatArray,
    t_span: tuple[float, float],
    dt: float,
) -> SolverResult[tuple[FloatArray, FloatArray]]:
    """Integrate with Heun's method (improved Euler)."""
    t0, t1 = t_span
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if t1 <= t0:
        raise ValueError("Require t1 > t0.")

    y = y0.copy()
    n_steps = int(np.ceil((t1 - t0) / dt))
    t_arr = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    y_hist = np.zeros((n_steps + 1, y.size), dtype=np.float64)
    y_hist[0] = y

    max_local_error = 0.0
    iterations = 0

    for i in range(n_steps):
        t = t_arr[i]
        h = t_arr[i + 1] - t
        k1 = f(t, y)
        y_predict = y + h * k1
        k2 = f(t + h, y_predict)
        y_new = y + 0.5 * h * (k1 + k2)
        max_local_error = max(max_local_error, float(np.linalg.norm(y_new - y_predict, ord=np.inf)))
        y = y_new
        y_hist[i + 1] = y
        iterations += 1

    error = max(max_local_error, np.finfo(np.float64).eps)
    return SolverResult(
        value=(t_arr, y_hist),
        error=error,
        iterations=iterations,
        metadata={"dt": dt, "n_steps": n_steps, "state_dim": y.size, "method": "euler"},
    )


@dataclass(frozen=True)
class Reaction:
    """Mass-action reaction model.

    Parameters
    ----------
    equation : str
        Reaction equation in the form 'A + B -> C'.
    k : float
        Forward rate constant.
    """

    equation: str
    k: float

    def __post_init__(self) -> None:
        reactants, products = _parse_equation(self.equation)
        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)
        species = set(reactants) | set(products)
        object.__setattr__(self, "species", tuple(sorted(species)))
        object.__setattr__(self, "stoichiometry", {
            species: products.get(species, 0) - reactants.get(species, 0)
            for species in self.species
        })
        if self.k < 0.0:
            raise ValueError("Rate constant k must be non-negative.")

    def _rate(self, concentrations: FloatArray, index: dict[str, int]) -> float:
        rate = float(self.k)
        for species, coefficient in self.reactants.items():
            rate *= concentrations[index[species]] ** coefficient
        return rate

    def _derivatives(self, _: float, concentrations: FloatArray, index: dict[str, int]) -> FloatArray:
        derivatives = np.zeros_like(concentrations)
        rate = self._rate(concentrations, index)
        for species, stoich in self.stoichiometry.items():
            derivatives[index[species]] = stoich * rate
        return derivatives

    def solve(
        self,
        initial: dict[str, float],
        t_max: float,
        dt: float = 0.1,
        method: str = "rk4",
    ) -> SolverResult[dict[str, FloatArray]]:
        """Solve the reaction kinetics for given initial concentrations."""
        if t_max <= 0.0:
            raise ValueError("t_max must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        state_species = sorted(set(initial) | set(self.species))
        state_index = {species: idx for idx, species in enumerate(state_species)}
        y0 = _build_state_vector(state_species, initial)

        def f(t: float, y: FloatArray) -> FloatArray:
            return self._derivatives(t, y, state_index)

        if method == "rk4":
            raw_result = rk4(f, y0, (0.0, t_max), dt)
        elif method == "euler":
            raw_result = _euler_improved(f, y0, (0.0, t_max), dt)
        else:
            raise ValueError("Unknown method. Supported methods: 'rk4', 'euler'.")

        t_values, y_history = raw_result.value
        time_series = {"t": t_values}
        time_series.update({species: y_history[:, idx] for species, idx in state_index.items()})

        return SolverResult(
            value=time_series,
            error=raw_result.error,
            iterations=raw_result.iterations,
            metadata={
                "equation": self.equation,
                "method": method,
                "k": self.k,
                "dt": dt,
                "n_species": y0.size,
                **(raw_result.metadata or {}),
            },
        )


class ReactionSystem:
    """Collection of coupled mass-action reactions."""

    def __init__(self, *reactions: Sequence[object], **rate_constants: float) -> None:
        if len(reactions) == 1 and isinstance(reactions[0], list):
            reactions = tuple(reactions[0])

        parsed_reactions: list[Reaction] = []
        auto_index = 1

        for item in reactions:
            if isinstance(item, Reaction):
                parsed_reactions.append(item)
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                parsed_reactions.append(Reaction(item[0], float(item[1])))
                continue
            if isinstance(item, str):
                key = f"k{auto_index}"
                if key not in rate_constants:
                    raise ValueError(
                        "Missing rate constant for reaction string. Use k1, k2, ... or provide Reaction objects."
                    )
                parsed_reactions.append(Reaction(item, float(rate_constants.pop(key))))
                auto_index += 1
                continue
            raise ValueError("ReactionSystem accepts Reaction, (equation, k) tuple, or equation string.")

        if not parsed_reactions:
            raise ValueError("ReactionSystem requires at least one reaction.")
        if rate_constants:
            raise ValueError(f"Unused rate constants: {', '.join(rate_constants)}")

        self.reactions = tuple(parsed_reactions)
        self.species = tuple(sorted({species for reaction in self.reactions for species in reaction.species}))

    def solve(
        self,
        initial: dict[str, float],
        t_max: float,
        dt: float = 0.1,
        method: str = "rk4",
    ) -> SolverResult[dict[str, FloatArray]]:
        """Solve the coupled reaction system over time."""
        if t_max <= 0.0:
            raise ValueError("t_max must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        state_species = sorted(set(initial) | set(self.species))
        state_index = {species: idx for idx, species in enumerate(state_species)}
        y0 = _build_state_vector(state_species, initial)

        def f(t: float, y: FloatArray) -> FloatArray:
            derivatives = np.zeros_like(y)
            for reaction in self.reactions:
                derivatives += reaction._derivatives(t, y, state_index)
            return derivatives

        if method == "rk4":
            raw_result = rk4(f, y0, (0.0, t_max), dt)
        elif method == "euler":
            raw_result = _euler_improved(f, y0, (0.0, t_max), dt)
        else:
            raise ValueError("Unknown method. Supported methods: 'rk4', 'euler'.")

        t_values, y_history = raw_result.value
        time_series = {"t": t_values}
        time_series.update({species: y_history[:, idx] for species, idx in state_index.items()})

        return SolverResult(
            value=time_series,
            error=raw_result.error,
            iterations=raw_result.iterations,
            metadata={
                "reactions": [reaction.equation for reaction in self.reactions],
                "method": method,
                "dt": dt,
                "n_species": y0.size,
                "n_reactions": len(self.reactions),
                **(raw_result.metadata or {}),
            },
        )


__all__ = ["Reaction", "ReactionSystem"]
