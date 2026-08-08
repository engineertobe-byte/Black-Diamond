"""Unit tests for chemical kinetics models."""

import math

import numpy as np

from black_diamond.chemistry.kinetics import Reaction, ReactionSystem


def test_first_order_reaction_rk4():
    reaction = Reaction("A -> B", k=0.1)
    result = reaction.solve(initial={"A": 1.0, "B": 0.0}, t_max=10.0, dt=0.1)
    assert "A" in result.value and "B" in result.value
    a = result.value["A"]
    b = result.value["B"]
    assert len(a) == len(b)
    analytical = np.exp(-0.1 * result.value["t"])
    assert np.allclose(a, analytical, atol=1e-3, rtol=1e-4)
    assert result.error < 1e-2


def test_second_order_reaction_rk4():
    reaction = Reaction("A + A -> B", k=0.2)
    result = reaction.solve(initial={"A": 1.0, "B": 0.0}, t_max=5.0, dt=0.05)
    a = result.value["A"]
    t = result.value["t"]
    # Analytical solution for d[A]/dt = -2k[A]^2 with [A](0)=1
    expected = 1.0 / (1.0 + 2.0 * 0.2 * t)
    assert np.allclose(a, expected, atol=4e-3, rtol=1e-3)


def test_reaction_system_competitive():
    system = ReactionSystem([
        ("A + B -> C", 0.05),
        ("C -> D + E", 0.02),
    ])
    result = system.solve(initial={"A": 1.0, "B": 2.0, "C": 0.0, "D": 0.0, "E": 0.0}, t_max=10.0, dt=0.1)
    assert result.value["C"][-1] > 0.0
    assert result.value["A"][-1] < 1.0
    assert result.error < 1e-2


def test_reaction_with_missing_initial_species():
    reaction = Reaction("A + B -> C", k=0.1)
    result = reaction.solve(initial={"A": 1.0}, t_max=1.0, dt=0.1)
    assert result.value["B"][0] == 0.0
    assert result.value["C"][0] == 0.0
    assert all(val >= 0.0 for val in result.value["A"])


def test_invalid_equation_raises():
    try:
        Reaction("A B -> C", k=0.1)
    except ValueError as exc:
        assert "Invalid reaction term" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid equation")


def test_negative_rate_constant_raises():
    try:
        Reaction("A -> B", k=-0.1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative k")


def test_reaction_system_requires_rate_constants():
    try:
        ReactionSystem("A -> B")
    except ValueError as exc:
        assert "Missing rate constant" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing rate constant")


def test_reaction_system_unused_rate_constant_raises():
    try:
        ReactionSystem("A -> B", k1=0.1, k99=0.2)
    except ValueError as exc:
        assert "Unused rate constants" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unused rate constant")


def test_euler_method_matches_rk4_trend():
    reaction = Reaction("A -> B", k=0.1)
    rk4_result = reaction.solve(initial={"A": 1.0, "B": 0.0}, t_max=5.0, dt=0.2, method="rk4")
    euler_result = reaction.solve(initial={"A": 1.0, "B": 0.0}, t_max=5.0, dt=0.2, method="euler")
    assert euler_result.value["A"][0] == rk4_result.value["A"][0]
    assert euler_result.value["A"][-1] < rk4_result.value["A"][-1] + 0.1


def test_metadata_includes_equation_and_method():
    reaction = Reaction("A -> B", k=0.1)
    result = reaction.solve(initial={"A": 1.0, "B": 0.0}, t_max=1.0, dt=0.1)
    assert result.metadata["equation"] == "A -> B"
    assert result.metadata["method"] == "rk4"
