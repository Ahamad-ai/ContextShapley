"""
Exact Shapley value and Shapley interaction index computation.

For N=5 components, exact computation requires 2^5=32 coalition evaluations
per task instance -- fully tractable without approximation.
"""

from itertools import combinations
from math import factorial
from typing import Callable


def _powerset(iterable):
    """Generate all subsets of an iterable."""
    items = list(iterable)
    for r in range(len(items) + 1):
        for combo in combinations(items, r):
            yield frozenset(combo)


def shapley_value(
    component: str,
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
) -> float:
    """
    Compute the exact Shapley value for a single component.

    Args:
        component: The component ID (e.g., "S", "E", "D", "H", "T").
        value_fn: Mapping from coalition (frozenset of component IDs) to score.
                  Must contain entries for all 2^N subsets.
        all_components: List of all component IDs.

    Returns:
        The Shapley value phi(component).
    """
    n = len(all_components)
    others = [c for c in all_components if c != component]
    phi = 0.0

    for subset in _powerset(others):
        s = len(subset)
        with_component = value_fn[subset | {component}]
        without_component = value_fn[subset]
        marginal = with_component - without_component
        weight = factorial(s) * factorial(n - s - 1) / factorial(n)
        phi += weight * marginal

    return phi


def all_shapley_values(
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
) -> dict[str, float]:
    """
    Compute Shapley values for all components.

    Returns:
        Dict mapping component ID -> Shapley value.
    """
    return {
        c: shapley_value(c, value_fn, all_components)
        for c in all_components
    }


def shapley_interaction(
    ci: str,
    cj: str,
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
) -> float:
    """
    Compute the Shapley interaction index between two components.

    Positive = synergy (together they contribute more than sum of individuals).
    Negative = interference (together they contribute less).
    Zero = independent.

    Args:
        ci, cj: The two component IDs.
        value_fn: Coalition -> score mapping.
        all_components: All component IDs.

    Returns:
        The Shapley interaction index I(ci, cj).
    """
    n = len(all_components)
    others = [c for c in all_components if c not in {ci, cj}]
    interaction = 0.0

    for subset in _powerset(others):
        s = len(subset)
        both = value_fn[subset | {ci, cj}]
        only_i = value_fn[subset | {ci}]
        only_j = value_fn[subset | {cj}]
        neither = value_fn[subset]

        marginal_interaction = (both - only_i) - (only_j - neither)
        weight = factorial(s) * factorial(n - s - 2) / factorial(n - 1)
        interaction += weight * marginal_interaction

    return interaction


def interaction_matrix(
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
) -> dict[tuple[str, str], float]:
    """
    Compute the full pairwise Shapley interaction matrix.

    Returns:
        Dict mapping (ci, cj) -> interaction index for all pairs.
    """
    pairs = {}
    for i, ci in enumerate(all_components):
        for j, cj in enumerate(all_components):
            if i < j:
                val = shapley_interaction(ci, cj, value_fn, all_components)
                pairs[(ci, cj)] = val
                pairs[(cj, ci)] = val
    return pairs


def verify_efficiency_axiom(
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
    tol: float = 1e-9,
) -> tuple[bool, float, float]:
    """
    Verify the efficiency axiom: sum of all Shapley values equals
    v(grand coalition) - v(empty set).

    Returns:
        (passes, sum_of_shapley, expected_value)
    """
    sv = all_shapley_values(value_fn, all_components)
    total_shapley = sum(sv.values())
    grand = value_fn[frozenset(all_components)]
    empty = value_fn[frozenset()]
    expected = grand - empty
    return abs(total_shapley - expected) < tol, total_shapley, expected


def verify_symmetry_axiom(
    ci: str,
    cj: str,
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
    tol: float = 1e-9,
) -> bool:
    """
    Verify symmetry: if ci and cj are interchangeable in all coalitions,
    their Shapley values must be equal.

    Checks if they ARE symmetric first, then verifies equal values.
    Returns True if either (a) they are not symmetric, or
    (b) they are symmetric AND have equal Shapley values.
    """
    others = [c for c in all_components if c not in {ci, cj}]
    is_symmetric = True
    for subset in _powerset(others):
        val_with_i = value_fn[subset | {ci}] - value_fn[subset]
        val_with_j = value_fn[subset | {cj}] - value_fn[subset]
        if abs(val_with_i - val_with_j) > tol:
            is_symmetric = False
            break

    if not is_symmetric:
        return True  # axiom is vacuously true when players aren't symmetric

    sv = all_shapley_values(value_fn, all_components)
    return abs(sv[ci] - sv[cj]) < tol


def verify_null_player_axiom(
    component: str,
    value_fn: dict[frozenset[str], float],
    all_components: list[str],
    tol: float = 1e-9,
) -> tuple[bool, bool, float]:
    """
    Verify null player axiom: if a component adds zero marginal contribution
    to every coalition, its Shapley value must be zero.

    Returns:
        (axiom_holds, is_null_player, shapley_value)
    """
    others = [c for c in all_components if c != component]
    is_null = True
    for subset in _powerset(others):
        marginal = value_fn[subset | {component}] - value_fn[subset]
        if abs(marginal) > tol:
            is_null = False
            break

    sv = shapley_value(component, value_fn, all_components)

    if is_null:
        return abs(sv) < tol, True, sv
    else:
        return True, False, sv  # axiom is vacuously true
