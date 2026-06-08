"""
Tests for Shapley value and interaction index computation.

Verifies:
1. Correctness on known game-theoretic examples
2. All four Shapley axioms (efficiency, symmetry, null player, linearity)
3. Interaction index properties
4. Edge cases (single player, all null players, etc.)
"""

import pytest
from math import isclose
from contextshapley.shapley import (
    shapley_value,
    all_shapley_values,
    shapley_interaction,
    interaction_matrix,
    verify_efficiency_axiom,
    verify_symmetry_axiom,
    verify_null_player_axiom,
    _powerset,
)


# ---------------------------------------------------------------------------
# Fixture: known game-theoretic examples
# ---------------------------------------------------------------------------

def _make_value_fn(components, fn):
    """Build a coalition->value dict from a function."""
    vf = {}
    for subset in _powerset(components):
        vf[subset] = fn(subset)
    return vf


class TestPowerSet:
    def test_powerset_empty(self):
        result = list(_powerset([]))
        assert result == [frozenset()]

    def test_powerset_single(self):
        result = list(_powerset(["A"]))
        assert len(result) == 2
        assert frozenset() in result
        assert frozenset({"A"}) in result

    def test_powerset_three(self):
        result = list(_powerset(["A", "B", "C"]))
        assert len(result) == 8  # 2^3

    def test_powerset_five(self):
        """Our actual use case: 5 components -> 32 subsets."""
        result = list(_powerset(["S", "E", "D", "H", "T"]))
        assert len(result) == 32


class TestShapleyValueBasic:
    """Test on the classic 'glove game': 2 left gloves, 1 right glove."""

    @pytest.fixture
    def glove_game(self):
        """
        Players: L1, L2 (left gloves), R (right glove).
        v(S) = min(left gloves in S, right gloves in S).
        Fair split: R gets 2/3, each L gets 1/6.
        """
        components = ["L1", "L2", "R"]

        def value(coalition):
            left = sum(1 for c in coalition if c.startswith("L"))
            right = sum(1 for c in coalition if c == "R")
            return min(left, right)

        return components, _make_value_fn(components, value)

    def test_glove_game_shapley_values(self, glove_game):
        components, vf = glove_game
        sv = all_shapley_values(vf, components)
        assert isclose(sv["L1"], 1 / 6, abs_tol=1e-10)
        assert isclose(sv["L2"], 1 / 6, abs_tol=1e-10)
        assert isclose(sv["R"], 2 / 3, abs_tol=1e-10)

    def test_glove_game_efficiency(self, glove_game):
        components, vf = glove_game
        passes, total, expected = verify_efficiency_axiom(vf, components)
        assert passes
        assert isclose(total, 1.0)  # v({L1,L2,R}) - v({}) = 1 - 0

    def test_glove_game_symmetry(self, glove_game):
        """L1 and L2 are symmetric players."""
        components, vf = glove_game
        assert verify_symmetry_axiom("L1", "L2", vf, components)


class TestShapleyValueAdditive:
    """Test on an additive game where each player has independent value."""

    @pytest.fixture
    def additive_game(self):
        """
        v(S) = sum of individual values: A=10, B=20, C=30, D=5, E=15.
        In additive games, Shapley value = individual value.
        """
        components = ["A", "B", "C", "D", "E"]
        values = {"A": 10, "B": 20, "C": 30, "D": 5, "E": 15}

        def value(coalition):
            return sum(values.get(c, 0) for c in coalition)

        return components, _make_value_fn(components, value), values

    def test_additive_shapley_equals_individual(self, additive_game):
        components, vf, individual = additive_game
        sv = all_shapley_values(vf, components)
        for c in components:
            assert isclose(sv[c], individual[c], abs_tol=1e-10)

    def test_additive_efficiency(self, additive_game):
        components, vf, _ = additive_game
        passes, total, expected = verify_efficiency_axiom(vf, components)
        assert passes
        assert isclose(expected, 80.0)  # 10+20+30+5+15

    def test_additive_zero_interactions(self, additive_game):
        """In additive games, all interaction indices should be zero."""
        components, vf, _ = additive_game
        im = interaction_matrix(vf, components)
        for pair, val in im.items():
            assert isclose(val, 0.0, abs_tol=1e-10), \
                f"Interaction {pair} should be 0, got {val}"


class TestShapleyValueFiveComponents:
    """Test on our actual 5-component scenario (S, E, D, H, T)."""

    @pytest.fixture
    def context_game(self):
        """
        Simulated context game where:
        - S (system) contributes 20 baseline
        - D (docs) contributes 30 for knowledge
        - E (examples) contributes 15 for format
        - S+D together get a bonus of 5 (synergy)
        - E+T together get a penalty of -3 (interference)
        - H contributes 0 (null player)
        """
        components = ["S", "E", "D", "H", "T"]

        def value(coalition):
            score = 0
            if "S" in coalition:
                score += 20
            if "D" in coalition:
                score += 30
            if "E" in coalition:
                score += 15
            if "T" in coalition:
                score += 10
            # H is a null player
            # Synergy: S + D together
            if "S" in coalition and "D" in coalition:
                score += 5
            # Interference: E + T together
            if "E" in coalition and "T" in coalition:
                score -= 3
            return score

        return components, _make_value_fn(components, value)

    def test_five_component_efficiency(self, context_game):
        components, vf = context_game
        passes, total, expected = verify_efficiency_axiom(vf, components)
        assert passes
        # Grand coalition: 20+30+15+10+5-3 = 77, empty = 0
        assert isclose(expected, 77.0)

    def test_null_player_h(self, context_game):
        """H should be a null player with Shapley value 0."""
        components, vf = context_game
        axiom_holds, is_null, sv = verify_null_player_axiom("H", vf, components)
        assert axiom_holds
        assert is_null
        assert isclose(sv, 0.0, abs_tol=1e-10)

    def test_non_null_players(self, context_game):
        """S, E, D, T should NOT be null players."""
        components, vf = context_game
        for c in ["S", "E", "D", "T"]:
            _, is_null, sv = verify_null_player_axiom(c, vf, components)
            assert not is_null
            assert sv != 0.0

    def test_sd_synergy(self, context_game):
        """S and D should have positive interaction (synergy)."""
        components, vf = context_game
        interaction = shapley_interaction("S", "D", vf, components)
        assert interaction > 0, f"S-D interaction should be positive, got {interaction}"

    def test_et_interference(self, context_game):
        """E and T should have negative interaction (interference)."""
        components, vf = context_game
        interaction = shapley_interaction("E", "T", vf, components)
        assert interaction < 0, f"E-T interaction should be negative, got {interaction}"

    def test_independent_pairs_zero_interaction(self, context_game):
        """Pairs without designed interaction should have ~zero interaction."""
        components, vf = context_game
        # S and E have no designed interaction
        interaction = shapley_interaction("S", "E", vf, components)
        assert isclose(interaction, 0.0, abs_tol=1e-10)

    def test_interaction_matrix_symmetric(self, context_game):
        components, vf = context_game
        im = interaction_matrix(vf, components)
        for i, ci in enumerate(components):
            for j, cj in enumerate(components):
                if i < j:
                    assert isclose(im[(ci, cj)], im[(cj, ci)], abs_tol=1e-10)

    def test_d_has_highest_shapley(self, context_game):
        """D contributes 30 base + part of synergy, should be highest."""
        components, vf = context_game
        sv = all_shapley_values(vf, components)
        assert sv["D"] > sv["S"] > sv["E"]


class TestShapleyEdgeCases:
    def test_single_player(self):
        """One-player game: Shapley value = v({A}) - v({})."""
        components = ["A"]
        vf = {frozenset(): 0.0, frozenset({"A"}): 42.0}
        sv = all_shapley_values(vf, components)
        assert isclose(sv["A"], 42.0)

    def test_all_zero_game(self):
        """All coalitions have value 0."""
        components = ["A", "B", "C"]
        vf = _make_value_fn(components, lambda s: 0.0)
        sv = all_shapley_values(vf, components)
        for c in components:
            assert isclose(sv[c], 0.0)

    def test_unanimity_game(self):
        """
        Unanimity game: v(S) = 1 if all players present, else 0.
        Shapley value = 1/N for each player.
        """
        components = ["A", "B", "C", "D"]
        n = len(components)

        def value(coalition):
            return 1.0 if len(coalition) == n else 0.0

        vf = _make_value_fn(components, value)
        sv = all_shapley_values(vf, components)
        for c in components:
            assert isclose(sv[c], 1.0 / n, abs_tol=1e-10)

    def test_dictator_game(self):
        """
        One player (dictator) determines all value.
        Dictator gets full Shapley value, others get 0.
        """
        components = ["D", "A", "B"]

        def value(coalition):
            return 100.0 if "D" in coalition else 0.0

        vf = _make_value_fn(components, value)
        sv = all_shapley_values(vf, components)
        assert isclose(sv["D"], 100.0)
        assert isclose(sv["A"], 0.0)
        assert isclose(sv["B"], 0.0)


class TestInteractionIndex:
    def test_positive_interaction(self):
        """Two players that create surplus together."""
        components = ["A", "B"]
        vf = {
            frozenset(): 0.0,
            frozenset({"A"}): 5.0,
            frozenset({"B"}): 5.0,
            frozenset({"A", "B"}): 15.0,  # 5 surplus beyond individual sums
        }
        interaction = shapley_interaction("A", "B", vf, components)
        assert interaction > 0

    def test_negative_interaction(self):
        """Two players that destroy value together."""
        components = ["A", "B"]
        vf = {
            frozenset(): 0.0,
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 10.0,
            frozenset({"A", "B"}): 12.0,  # less than sum of individuals
        }
        interaction = shapley_interaction("A", "B", vf, components)
        assert interaction < 0

    def test_no_interaction(self):
        """Two independent players."""
        components = ["A", "B"]
        vf = {
            frozenset(): 0.0,
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 10.0,
            frozenset({"A", "B"}): 20.0,
        }
        interaction = shapley_interaction("A", "B", vf, components)
        assert isclose(interaction, 0.0, abs_tol=1e-10)
