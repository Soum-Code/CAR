"""Tests for verifier reach as a lookback window.

The measured result these lock in: a step-local arithmetic verifier has scope
0.0000, reach saturates at 0.1999, and the ceiling exists because 80% of
inherited corruption has no upstream arithmetic error at all.
"""

import pytest

from car.data.math_shepherd import parse_solution
from car.verification.lookback import (
    LookbackVerifier,
    corruption_distance,
    inherited_steps,
)

# Step 2 is arithmetically wrong (13*8/13 = 8, not 6); steps 3 and 4 are
# arithmetically perfect and inherit its corruption.
CHAIN = (
    "Q "
    "Step 1: total ratio is 5+8 = <<5+8=13>>13. + "
    "Step 2: waiter ate 13 x 8 / 13 = <<13*8/13=6>>6 slices. - "
    "Step 3: Buzz ate 78 - 6 = <<78-6=72>>72 slices. - "
    "Step 4: waiter ate 72 - 20 = <<72-20=52>>52. -"
)


def test_step_local_verifier_is_blind_to_inheritance():
    """k=0 is the naive calculator: it checks the current step's arithmetic,
    which is valid, and passes. Measured scope on 35,535 real steps: 0.0000."""
    sol = parse_solution(CHAIN)
    v = LookbackVerifier(k=0)
    assert not v.verify_step(sol, 2).detected  # step 3, locally valid
    assert not v.verify_step(sol, 3).detected  # step 4, locally valid


def test_lookback_one_catches_the_immediate_premise():
    sol = parse_solution(CHAIN)
    v = LookbackVerifier(k=1)
    r = v.verify_step(sol, 2)
    assert r.detected and r.culprit_index == 1


def test_lookback_one_misses_distance_two():
    """Step 4 is two hops from the error; a k=1 window cannot reach it."""
    sol = parse_solution(CHAIN)
    assert not LookbackVerifier(k=1).verify_step(sol, 3).detected
    assert LookbackVerifier(k=2).verify_step(sol, 3).detected


def test_unbounded_lookback_reaches_any_distance():
    sol = parse_solution(CHAIN)
    v = LookbackVerifier(k=None)
    assert v.verify_step(sol, 3).detected
    assert v.verify_step(sol, 3).culprit_index == 1


def test_reach_costs_steps_examined():
    """Reach is not free -- window k re-examines up to k+1 steps."""
    sol = parse_solution(CHAIN)
    narrow, wide = LookbackVerifier(k=0), LookbackVerifier(k=None)
    narrow.verify_step(sol, 3)
    wide.verify_step(sol, 3)
    assert wide.cost_per_call() > narrow.cost_per_call()
    assert narrow.cost_per_call() == 1.0


def test_window_uses_step_labels_not_list_positions():
    """Regression: `index` comes from the "Step N:" label and a few solutions
    number non-contiguously, so indexing the list by it raised IndexError."""
    sol = parse_solution(
        "Q Step 1: a = <<1+1=2>>2. + Step 3: b = <<2+2=4>>4. +"
    )
    v = LookbackVerifier(k=None)
    r = v.verify_step(sol, 2)  # label 3 -> index 2, but list has only 2 entries
    assert r.window == [0, 2]


def test_corruption_distance():
    sol = parse_solution(CHAIN)
    assert corruption_distance(sol, 2) == 1
    assert corruption_distance(sol, 3) == 2
    assert corruption_distance(sol, 1) is None  # nothing wrong upstream yet


def test_inherited_steps_are_locally_valid_and_globally_wrong():
    sol = parse_solution(CHAIN)
    inh = inherited_steps(sol)
    assert [s.index for s in inh] == [2, 3]
    assert all(s.local_ok is True and not s.global_ok for s in inh)


def test_no_upstream_arithmetic_error_is_undetectable_at_any_window():
    """The ceiling, in miniature. Every step's arithmetic is correct, yet the
    solution is globally wrong -- the mistake is in the SETUP, not the
    calculation. 80% of real inherited corruption looks like this, and no
    window size helps."""
    sol = parse_solution(
        "Q Step 1: x = <<2+2=4>>4. - Step 2: y = <<4+1=5>>5. -"
    )
    assert all(s.local_ok is True for s in sol.steps)
    assert not LookbackVerifier(k=None).verify_step(sol, 1).detected


def test_rejects_negative_window():
    with pytest.raises(ValueError):
        LookbackVerifier(k=-1)
