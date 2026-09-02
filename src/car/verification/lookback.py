"""Verifier reach as a controllable lookback window.

`scope` -- the probability a verifier detects inherited corruption -- has been a
free parameter in every propagation result so far. This module makes it
measurable, and in doing so says what scope actually is.

The reframing
-------------
Scope is not a fixed property of a verifier type. It is a property of **how much
context the verifier re-examines**.

    Step 2:  13 x 8 / 13 = <<13*8/13=6>>6      WRONG, should be 8
    Step 3:  78 - 6 = <<78-6=72>>72            arithmetically perfect

A calculator that checks only step 3 sees `78 - 6 = 72` and passes it. It has no
way to know the 6 was fabricated. The SAME calculator re-checking steps 2-3
catches the error, and so rejects step 3's premise.

So reach is a window, not a capability:

    k = 0    check the current step only          scope ~ 0
    k = 1    also re-check the immediate premise  catches d = 1
    k = inf  re-check the whole prefix            catches everything checkable

This makes scope a design dial with a measurable cost, rather than an
unmeasured constant.

The ceiling
-----------
Reach does not reach 1 even at k = inf. Roughly 12% of GSM8K steps carry no
calculator annotation, so an arithmetic verifier cannot check them at all. An
error that originates in, or propagates through, an unverifiable step is
invisible at any window size. That ceiling is the interesting quantity and it
is what `scripts/exp_verifier_scope.py` measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from car.data.math_shepherd import ShepherdSolution, ShepherdStep


@dataclass
class ScopeResult:
    detected: bool
    # Index of the step whose arithmetic actually failed, when found. Useful for
    # checking the verifier localises the error rather than merely flagging.
    culprit_index: int | None
    # Steps in the window that carried no arithmetic, so could not be checked.
    n_unverifiable: int
    window: list[int]


class LookbackVerifier:
    """Re-checks the arithmetic of the last `k` steps plus the current one.

    Parameters
    ----------
    k:
        Lookback window. 0 checks only the current step (the naive calculator);
        None re-checks the entire prefix.

    Cost model: a call with window k re-examines up to k+1 steps, so cost grows
    linearly in reach. That is the trade-off the experiment quantifies -- reach
    is not free, and an unbounded-lookback verifier costs O(chain length) per
    call.
    """

    def __init__(self, k: int | None = 0) -> None:
        if k is not None and k < 0:
            raise ValueError("k must be >= 0 or None")
        self.k = k
        self.calls = 0
        self.steps_examined = 0

    @property
    def name(self) -> str:
        return f"lookback(k={'inf' if self.k is None else self.k})"

    def window_for(self, steps: list[ShepherdStep], t: int) -> list[ShepherdStep]:
        """Steps in the lookback window ending at the step labelled `t`.

        Selects by the step's own `index` attribute rather than list position:
        `index` comes from the "Step N:" label in the source text, and a small
        number of solutions number their steps non-contiguously, so the two are
        not interchangeable.
        """
        lo = None if self.k is None else t - self.k
        return [s for s in steps if s.index <= t and (lo is None or s.index >= lo)]

    def verify_step(self, sol: ShepherdSolution, t: int) -> ScopeResult:
        """Check step `t` and its lookback window.

        Returns detected=True when ANY step in the window has provably wrong
        arithmetic. That is the realistic semantics: the verifier reports that
        the reasoning supporting step t is unsound, without needing to know
        which downstream conclusion it invalidates.
        """
        self.calls += 1
        window = self.window_for(sol.steps, t)
        self.steps_examined += len(window)

        culprit = None
        unverifiable = 0
        for s in window:
            if s.local_ok is None:
                unverifiable += 1
            elif s.local_ok is False and culprit is None:
                culprit = s.index

        return ScopeResult(
            detected=culprit is not None,
            culprit_index=culprit,
            n_unverifiable=unverifiable,
            window=[s.index for s in window],
        )

    def cost_per_call(self) -> float:
        return self.steps_examined / max(1, self.calls)


def corruption_distance(sol: ShepherdSolution, t: int) -> int | None:
    """Hops from the nearest upstream local error to step `t`.

    None when no local error precedes t -- either the step is clean, or its
    corruption came from something an arithmetic checker cannot see, which is
    itself part of the ceiling being measured.
    """
    origin = None
    for s in sol.steps:
        if s.index >= t:
            break
        if s.local_ok is False:
            origin = s.index
    return None if origin is None else t - origin


def inherited_steps(sol: ShepherdSolution) -> list[ShepherdStep]:
    """Steps that are locally valid but globally wrong.

    The population whose detectability defines scope: arithmetically perfect,
    and still wrong because a premise was.
    """
    return [s for s in sol.steps if s.inherited_corruption]
