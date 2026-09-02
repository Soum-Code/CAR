"""Tests for Math-Shepherd parsing and the local/global step distinction.

Parsing tests run everywhere; corpus tests skip without the download.

The corpus tests also guard the sampling trap found here: the Math-Shepherd
file is sorted into blocks by label and task, so a contiguous prefix is
entirely one class.
"""

import json
from pathlib import Path

import pytest

from car.data.math_shepherd import (
    check_arithmetic,
    class_balance,
    load_solutions,
    parse_solution,
)

DATA = Path("data/raw/mathshepherd/strided.jsonl")
needs_data = pytest.mark.skipif(not DATA.exists(), reason="Math-Shepherd not downloaded")


# ---- arithmetic checking ---------------------------------------------


def test_check_arithmetic_accepts_correct():
    assert check_arithmetic("48/2", "24") is True


def test_check_arithmetic_rejects_wrong():
    assert check_arithmetic("20*40", "8000") is False  # = 800


def test_check_arithmetic_returns_none_when_unparseable():
    assert check_arithmetic("x+y", "3") is None
    assert check_arithmetic("1/0", "0") is None


def test_check_arithmetic_refuses_code():
    """Model output is untrusted; this path must not reach the interpreter."""
    assert check_arithmetic("__import__('os').system('x')", "0") is None


# ---- the local/global distinction ------------------------------------


LABELLED = (
    "If Buzz bought a pizza with 78 slices... "
    "Step 1: The total ratio is 5+8 = <<5+8=13>>13. + "
    "Step 2: The waiter ate 13 x 8 / 13 = <<13*8/13=6>>6 slices. - "
    "Step 3: Buzz ate 78 - 6 = <<78-6=72>>72 slices. - "
    "Step 4: The waiter ate 72 - 20 = <<72-20=52>>52. -"
)


def test_parses_steps_and_labels():
    sol = parse_solution(LABELLED)
    assert len(sol.steps) == 4
    assert [s.global_ok for s in sol.steps] == [True, False, False, False]


def test_locally_wrong_step_is_detected():
    """Step 2 claims 13*8/13 = 6, which is 8. A calculator catches it."""
    sol = parse_solution(LABELLED)
    assert sol.steps[1].local_ok is False


def test_inherited_corruption_is_the_gap():
    """THE measured phenomenon: steps 3 and 4 are arithmetically perfect and
    still globally wrong, because step 2 poisoned the premise.

    A local verifier cannot see these. That is why controlling local selective
    risk bounds nothing about the answer.
    """
    sol = parse_solution(LABELLED)
    assert sol.steps[2].local_ok is True and sol.steps[2].global_ok is False
    assert sol.steps[2].inherited_corruption
    assert sol.steps[3].inherited_corruption
    assert not sol.steps[1].inherited_corruption  # locally wrong, not inherited


def test_first_error_indices():
    sol = parse_solution(LABELLED)
    assert sol.first_bad_index == 1
    assert sol.first_local_error_index == 1
    assert not sol.final_correct


def test_step_without_arithmetic_is_uncheckable_not_assumed_correct():
    sol = parse_solution("Q Step 1: Some prose with no calculation. +")
    assert sol.steps[0].local_ok is None
    assert not sol.steps[0].checkable


def test_parse_returns_none_without_steps():
    assert parse_solution("just a question, no steps") is None


# ---- corpus ----------------------------------------------------------


@needs_data
def test_sample_is_not_single_class():
    """Guards the sampling trap: the file is sorted into blocks by label, so a
    contiguous read gives 0% or 100% accuracy. The strided sample must contain
    both classes."""
    strided = load_solutions(DATA, limit=20000, stride=3)
    assert 0.05 < class_balance(strided) < 0.95, (
        f"strided sample looks single-class: {class_balance(strided):.1%}"
    )
    # And the trap itself: a prefix read lands inside one block.
    prefix = load_solutions(DATA, limit=20000)
    assert class_balance(prefix) < class_balance(strided), (
        "prefix read should be more class-skewed than a strided one"
    )


@needs_data
def test_most_bad_steps_are_inherited_not_locally_wrong():
    """The headline measurement. Among steps in wrong-answer solutions, far
    more are locally valid-but-poisoned than locally invalid."""
    sols = [s for s in load_solutions(DATA, limit=20000, stride=3) if not s.final_correct]
    steps = [s for sol in sols for s in sol.steps]
    bad = [s for s in steps if not s.global_ok]
    inherited = [s for s in bad if s.local_ok is True]
    assert len(inherited) / len(bad) > 0.5


@needs_data
def test_later_steps_are_harder():
    """Closes the last escape for front-loaded verification: if EARLY steps
    were harder, front-loading would gain. They are not."""
    import numpy as np

    sols = load_solutions(DATA, limit=20000, stride=3)
    tot, bad = {}, {}
    for sol in sols:
        for s in sol.steps:
            if s.checkable:
                tot[s.index] = tot.get(s.index, 0) + 1
                bad[s.index] = bad.get(s.index, 0) + (1 if s.local_ok is False else 0)
    xs = [i for i in sorted(tot) if tot[i] >= 300]
    ys = [bad[i] / tot[i] for i in xs]
    assert float(np.corrcoef(xs, ys)[0, 1]) > 0.5


@needs_data
def test_corruption_is_near_absorbing():
    sols = load_solutions(DATA, limit=20000, stride=3)
    tot = still_bad = 0
    for sol in sols:
        fb = sol.first_bad_index
        if fb is None:
            continue
        tail = [s for s in sol.steps if s.index > fb]
        tot += len(tail)
        still_bad += sum(1 for s in tail if not s.global_ok)
    assert still_bad / max(1, tot) > 0.9
