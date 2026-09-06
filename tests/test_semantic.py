"""Tests for step-level semantic divergence.

The measurement these protect is the one the original spec weighted most
heavily and the pipeline could not test, so the clustering has to be right
before any number from it is believed. The failure mode that matters is
notational: a model that writes the same arithmetic three ways would look
maximally uncertain under string equality, and that divergence would correlate
with verbosity rather than with doubt.
"""

import pytest

from car.uncertainty.semantic import (
    cluster_by_equivalence,
    exact_match_equivalence,
    normalised_semantic_divergence,
    numeric_equivalence,
    semantic_entropy,
)

BS = chr(92)


# ---- numeric equivalence ---------------------------------------------


def test_same_value_different_notation_is_one_meaning():
    a = "He sold 48/2 = <<48/2=24>>24 clips."
    b = BS + "( 48 / 2 = 24 " + BS + ")"
    c = "48 / 2 = 24"
    assert numeric_equivalence(a, b)
    assert numeric_equivalence(b, c)
    assert not exact_match_equivalence(a, b)


def test_different_values_are_different_meanings():
    assert not numeric_equivalence("48/2 = <<48/2=24>>24", "48/2 = <<48/2=25>>25")


def test_falls_back_to_string_equality_without_a_number():
    assert numeric_equivalence("First we find the total.", "First we find the total.")
    assert not numeric_equivalence("First we find the total.", "Next, subtract.")


def test_a_step_with_a_number_never_matches_one_without():
    """Falling back to string equality here would be worse than no match."""
    assert not numeric_equivalence("2 + 2 = 4", "First we add them up.")


def test_the_last_asserted_value_is_the_one_compared():
    """Chained working ends on its conclusion; that is what the step asserts."""
    a = BS + "[ " + BS + "text{total} = 3 " + BS + "times 8 = 24 " + BS + "]"
    assert numeric_equivalence(a, "the total is 3*8 = 24")


# ---- clustering and divergence ---------------------------------------


def test_unanimous_samples_have_zero_divergence():
    ids = cluster_by_equivalence(["1+1 = 2", "1 + 1 = 2", "1+1=2"], numeric_equivalence)
    assert len(set(ids)) == 1
    assert normalised_semantic_divergence(ids) == 0.0


def test_total_disagreement_is_maximal_divergence():
    ids = cluster_by_equivalence(
        ["1+1 = 2", "1+1 = 3", "1+1 = 4", "1+1 = 5"], numeric_equivalence
    )
    assert len(set(ids)) == 4
    assert normalised_semantic_divergence(ids) == pytest.approx(1.0)


def test_notation_variety_does_not_manufacture_divergence():
    """The reason numeric equivalence exists, stated as a test.

    Three phrasings of one computation. Exact match reports near-maximal
    disagreement; numeric equivalence reports none.
    """
    samples = [
        "48/2 = <<48/2=24>>24 clips",
        BS + "( 48 / 2 = 24 " + BS + ") clips",
        "That gives 48 / 2 = 24 clips.",
    ]
    numeric = normalised_semantic_divergence(
        cluster_by_equivalence(samples, numeric_equivalence)
    )
    exact = normalised_semantic_divergence(
        cluster_by_equivalence(samples, exact_match_equivalence)
    )
    assert numeric == 0.0
    assert exact == pytest.approx(1.0)


def test_divergence_is_normalised_across_sample_counts():
    """Unnormalised entropy would let sample count dominate the composite."""
    five = cluster_by_equivalence([f"1+1 = {i}" for i in range(5)], numeric_equivalence)
    ten = cluster_by_equivalence([f"1+1 = {i}" for i in range(10)], numeric_equivalence)
    assert normalised_semantic_divergence(five) == pytest.approx(1.0)
    assert normalised_semantic_divergence(ten) == pytest.approx(1.0)
    assert semantic_entropy(ten) > semantic_entropy(five)


def test_single_sample_carries_no_divergence():
    assert normalised_semantic_divergence([0]) == 0.0
    assert normalised_semantic_divergence([]) == 0.0


def test_majority_agreement_sits_between_the_extremes():
    ids = cluster_by_equivalence(
        ["1+1 = 2", "1+1 = 2", "1+1 = 2", "1+1 = 3"], numeric_equivalence
    )
    d = normalised_semantic_divergence(ids)
    assert 0.0 < d < 1.0
