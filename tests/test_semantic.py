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
    EntailmentEquivalence,
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


# ---- bidirectional entailment ----------------------------------------
# The NLI model is a 1.6GB download and far too slow for a unit test, so these
# stub `_entails_batch` and exercise the logic around it: the bidirectionality,
# the cache, the batching hook and the question conditioning. What the model
# itself decides is measured in docs/FINDINGS-ENTAILMENT.md, not asserted here.


class FakeNLI(EntailmentEquivalence):
    """Entails iff the directed pair is in `truth`. Counts forward passes."""

    def __init__(self, truth, **kw):
        super().__init__(**kw)
        self.truth = set(truth)
        self.batches = 0

    def _entails_batch(self, keys):
        if keys:
            self.batches += 1
        self.calls += len(keys)
        # Keys are (context, premise, hypothesis); the fake ignores context so
        # the tests can state their expectations as plain text pairs.
        return [(a, b) in self.truth for _, a, b in keys]


def test_entailment_must_hold_in_both_directions():
    """One-way entailment is not equivalence.

    "he sold 24 clips" entails "he sold clips" and not the reverse, and calling
    that a shared meaning is what collapses distinct answers into one cluster.
    """
    eq = FakeNLI({("a", "b")})
    assert not eq("a", "b")
    assert not eq("b", "a")

    eq = FakeNLI({("a", "b"), ("b", "a")})
    assert eq("a", "b")
    assert eq("b", "a")


def test_identical_texts_never_reach_the_model():
    eq = FakeNLI(set())
    assert eq("same", "same")
    assert eq.calls == 0


def test_prime_batches_every_pair_at_once():
    """Without priming this is one forward pass per question, which on CPU is
    the difference between half an hour and half a day."""
    eq = FakeNLI({("a", "b"), ("b", "a")})
    eq.prime(["a", "b", "c"])
    assert eq.batches == 1
    assert eq.calls == 6  # 3 texts, ordered pairs, self-pairs skipped

    before = eq.calls
    assert eq("a", "b")
    assert not eq("a", "c")
    assert eq.calls == before  # answered from cache


def test_prime_does_not_repeat_work_across_steps():
    eq = FakeNLI({("a", "b"), ("b", "a")})
    eq.prime(["a", "b"])
    first = eq.calls
    eq.prime(["a", "b"])
    assert eq.calls == first


def test_clustering_through_the_relation_merges_only_mutual_pairs():
    eq = FakeNLI({("x", "y"), ("y", "x")})
    eq.prime(["x", "y", "z"])
    ids = cluster_by_equivalence(["x", "y", "z"], eq)
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_context_is_prepended_to_both_sides():
    """Kuhn et al. condition entailment on the question; two step fragments can
    be mutually entailing as bare strings and disagree as answers to it."""
    eq = FakeNLI(set())
    eq.set_context("Q: how many clips?")
    assert eq._pair("a", "b") == ("Q: how many clips? a", "Q: how many clips? b")
    eq.set_context("")
    assert eq._pair("a", "b") == ("a", "b")


def test_prime_many_fills_one_batch_across_steps():
    """The reason the corpus is primed before clustering rather than per step.

    A single step has ~11 distinct ordered pairs, nowhere near a batch; priming
    per step measured under 0.2 steps/s on 16 cores.
    """
    eq = FakeNLI(set(), batch_size=64)
    eq.prime_many([("q1", ["a", "b"]), ("q2", ["c", "d"]), ("q3", ["e", "f"])])
    assert eq.batches == 1
    assert eq.calls == 6


def test_cache_is_keyed_on_context_too():
    """Two identical fragments can entail under one question and not another.

    A context-blind cache reuses the first verdict for the second question,
    which is wrong rather than merely approximate -- and silent.
    """
    eq = FakeNLI({("a", "b"), ("b", "a")})
    eq.set_context("Q1")
    eq.prime(["a", "b"])
    assert eq("a", "b")

    before = eq.calls
    eq.set_context("Q2")
    eq.prime(["a", "b"])
    assert eq.calls > before, "second question must not reuse the first's cache"


def test_prime_many_reports_progress():
    """A half-hour CPU run with no output is indistinguishable from a hung one."""
    seen = []
    eq = FakeNLI(set(), batch_size=2)
    eq.prime_many([("q", ["a", "b", "c"])], progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]


def test_verdicts_survive_a_restart(tmp_path):
    """Resumability. The NLI pass is ~28k pairs at ~2/s on CPU -- nearly four
    hours -- and losing it to an interruption is the failure mode that has
    already cost this project one long run."""
    path = tmp_path / "cache.jsonl"
    eq = FakeNLI({("a", "b"), ("b", "a")}, cache_path=path)
    eq.set_context("Q")
    eq.prime(["a", "b"])
    first = eq.calls
    eq.close()
    assert first > 0

    resumed = FakeNLI(set(), cache_path=path)  # empty truth: any call is a miss
    resumed.set_context("Q")
    assert resumed.cached_hits == first
    assert resumed("a", "b")          # answered from the persisted verdicts
    assert resumed.calls == 0


def test_a_truncated_cache_line_is_skipped_not_fatal(tmp_path):
    """A hard kill mid-write leaves a partial final line."""
    path = tmp_path / "cache.jsonl"
    path.write_text('{"c":"Q","a":"a","b":"b","e":true}\n{"c":"Q","a":"a","b"',
                    encoding="utf-8")
    eq = FakeNLI(set(), cache_path=path)
    assert eq.cached_hits == 1
