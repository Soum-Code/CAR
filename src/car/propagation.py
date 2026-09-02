"""Error propagation model: local validity vs global correctness.

The distinction this module makes precise, which the original spec collapses,
and which turns out to decide whether CAR has a thesis.

A reasoning step can fail in two different ways:

  local invalidity     the step does not follow from its premises
                       (47 * 3 = 131 -- wrong regardless of context)

  inherited corruption the step follows perfectly from its premises, but a
                       premise is itself globally wrong
                       (Aristotle died in 1850, so he could have used a laptop
                       -- impeccable logic, false conclusion)

    global_correct(t) = local_valid(t) AND NOT premise_corrupt(t)

Why it matters: **verifiers differ in which failure they can see.**

A calculator confirms 47 * 3 = 141 given those inputs. It cannot tell you the
141 came from a hallucinated premise three steps back. A retrieval verifier
checking a claim against evidence can. The spec's verifier table lists
calculator / retrieval / sandbox side by side as if interchangeable. They are
not, and `scope` models the difference.

Three modelling axes, each of which changes the conclusion:

  scope         can the verifier see inherited corruption at all?
  scope_decay   does that ability fade as the error travels? Singh & Pawar
                (arXiv:2608.14588) measured exactly this decay empirically --
                escape probabilities of 24.6% / 48.3% / 89.3% across successive
                pipeline boundaries. Constant scope is the unrealistic case.
  outcome       does the final answer depend only on the LAST step, or on every
                step having been right when committed?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

OutcomeModel = Literal["terminal", "conjunctive"]


@dataclass
class StepOutcome:
    step_id: int
    local_valid: bool
    premise_corrupt: bool
    verified: bool = False
    repaired: bool = False
    caught: bool = False

    @property
    def global_correct(self) -> bool:
        return self.local_valid and not self.premise_corrupt


@dataclass
class ChainResult:
    steps: list[StepOutcome] = field(default_factory=list)
    verification_calls: int = 0
    ever_committed_error: bool = False
    # Steps that had to be regenerated because a repair invalidated work built
    # on top of the corrupted premise. This is the cost channel the accuracy
    # metrics miss entirely: catching an error at step 1 costs one revision,
    # catching the same error at step 7 means steps 2-7 were built on sand.
    regenerated_steps: int = 0

    def total_cost(self, verify_cost: float = 1.0, gen_cost: float = 1.0) -> float:
        """Total compute: generation + verification + regeneration."""
        return (
            len(self.steps) * gen_cost
            + self.verification_calls * verify_cost
            + self.regenerated_steps * gen_cost
        )

    def final_correct(self, outcome: OutcomeModel = "terminal") -> bool:
        """Whether the final answer is right.

        terminal     the answer reflects the last step only. Generous to late
                     verification: one global check at the end can rescue the
                     whole chain.
        conjunctive  the answer is wrong if any step was ever committed while
                     globally incorrect. Harsh: repair cannot undo a conclusion
                     already built on a bad premise.

        Reality sits between these. Reporting both bounds the conclusion
        instead of smuggling in an assumption.
        """
        if not self.steps:
            return False
        if outcome == "terminal":
            return self.steps[-1].global_correct
        return not self.ever_committed_error

    @property
    def local_selective_risk(self) -> float:
        """Error rate among ACCEPTED steps, judged LOCALLY.

        What a verifier reports, and therefore what conformal machinery can
        actually control.
        """
        accepted = [s for s in self.steps if not s.verified]
        if not accepted:
            return 0.0
        return sum(not s.local_valid for s in accepted) / len(accepted)

    @property
    def global_selective_risk(self) -> float:
        """Error rate among accepted steps, judged GLOBALLY.

        What we care about, and not directly observable: deciding it for a
        downstream step needs to know what that step would have been under an
        uncorrupted premise.
        """
        accepted = [s for s in self.steps if not s.verified]
        if not accepted:
            return 0.0
        return sum(not s.global_correct for s in accepted) / len(accepted)

    @property
    def first_error_position(self) -> int | None:
        for s in self.steps:
            if not s.global_correct:
                return s.step_id
        return None


class PropagationChain:
    """Simulates a reasoning chain with local errors and inherited corruption.

    Parameters
    ----------
    local_error_rate:
        P(a step is locally invalid), independent per step.
    verifier_scope:
        P(the verifier detects inherited corruption) at the step immediately
        after it was introduced. 0.0 = purely local verifier, 1.0 = global.
    scope_decay:
        Multiplicative decay of that detection probability per step of
        propagation distance. Effective scope at age k is
        `verifier_scope * scope_decay ** (k - 1)`. decay = 1.0 means
        detectability never fades, which the empirical evidence contradicts.
    verifier_error_rate:
        P(the verifier is wrong about something it can see).
    repair_success:
        P(a caught error is successfully revised rather than merely flagged).
    """

    def __init__(
        self,
        *,
        length: int = 8,
        local_error_rate: float = 0.15,
        verifier_scope: float = 0.0,
        scope_decay: float = 1.0,
        verifier_error_rate: float = 0.0,
        repair_success: float = 1.0,
    ) -> None:
        self.length = length
        self.local_error_rate = local_error_rate
        self.verifier_scope = verifier_scope
        self.scope_decay = scope_decay
        self.verifier_error_rate = verifier_error_rate
        self.repair_success = repair_success

    def effective_scope(self, age: int) -> float:
        """Detection probability for corruption that is `age` steps old."""
        if age < 1:
            return 0.0
        return self.verifier_scope * (self.scope_decay ** (age - 1))

    def run(self, policy, rng: np.random.Generator, budget: int | None = None) -> ChainResult:
        """Execute one chain under a verification `policy`.

        `policy(step_id, length, remaining_budget, rng) -> bool` gets no access
        to ground truth.
        """
        result = ChainResult()
        remaining = budget if budget is not None else self.length
        premise_corrupt = False
        corruption_age = 0

        for t in range(self.length):
            local_valid = rng.random() >= self.local_error_rate
            step = StepOutcome(
                step_id=t, local_valid=local_valid, premise_corrupt=premise_corrupt
            )

            if policy(t, self.length, remaining, rng) and remaining > 0:
                step.verified = True
                remaining -= 1
                result.verification_calls += 1

                sees_local = not local_valid
                sees_inherited = premise_corrupt and rng.random() < self.effective_scope(
                    corruption_age
                )

                if (sees_local or sees_inherited) and rng.random() >= self.verifier_error_rate:
                    step.caught = True
                    if rng.random() < self.repair_success:
                        step.repaired = True
                        step.local_valid = True
                        # A repair clears inherited corruption only if the
                        # verifier could see it. A calculator cannot
                        # un-corrupt a bad premise.
                        if sees_inherited:
                            step.premise_corrupt = False
                            premise_corrupt = False
                            # Every step generated since the corruption was
                            # introduced rested on it and must be redone.
                            result.regenerated_steps += corruption_age
                            corruption_age = 0

            result.steps.append(step)

            if not step.global_correct:
                result.ever_committed_error = True
                if not premise_corrupt:
                    premise_corrupt = True
                    corruption_age = 0

            if premise_corrupt:
                corruption_age += 1

        return result


# ---- verification allocation policies --------------------------------


def never(step_id, length, remaining, rng) -> bool:
    return False


def always(step_id, length, remaining, rng) -> bool:
    return True


def uniform_random(rate: float):
    def policy(step_id, length, remaining, rng):
        return rng.random() < rate

    return policy


def first_k(k: int):
    def policy(step_id, length, remaining, rng):
        return step_id < k

    return policy


def schedule_policy(schedule):
    """Per-step verification probabilities."""

    def policy(step_id, length, remaining, rng):
        return rng.random() < schedule[step_id]

    return policy


# ---- closed form -----------------------------------------------------


def final_error_probability(
    length: int,
    verify_rate: float | list[float],
    scope: float,
    local_error_rate: float,
    *,
    scope_decay: float = 1.0,
    outcome: OutcomeModel = "terminal",
) -> float:
    """Exact P(final answer wrong), by age-indexed Markov chain.

    States: CLEAN, plus CORRUPT_k for corruption of age k = 1..length. Age must
    be tracked because detectability decays with it; a two-state chain is only
    correct when scope_decay == 1.

    Transitions per step, with v_t the verification probability:

        CLEAN -> CORRUPT_1   e * (1 - v_t)
            local invalidity is visible to ANY verifier, so scope is absent

        CORRUPT_k -> CLEAN   v_t * scope * decay^(k-1)
            a successful repair restores both global correctness and local
            validity -- finding the claim wrong and revising it correctly
            fixes both at once, which is what the simulator does

    Note the asymmetry that drives every result here: entering corruption does
    not depend on scope, escaping it does.

    Under `outcome="conjunctive"` the answer is wrong if corruption was ever
    entered, so repairs cannot help and this reduces to
    `1 - prod_t (1 - e(1 - v_t))`.
    """
    v = (
        [float(verify_rate)] * length
        if isinstance(verify_rate, int | float)
        else list(verify_rate)
    )
    if len(v) != length:
        raise ValueError(f"verify_rate has length {len(v)}, expected {length}")
    e = local_error_rate

    if outcome == "conjunctive":
        p_clean = 1.0
        for t in range(length):
            p_clean *= 1.0 - e * (1.0 - v[t])
        return float(1.0 - p_clean)

    # Age-indexed: index 0 = CLEAN, index k = CORRUPT of age k.
    state = np.zeros(length + 2)
    state[0] = 1.0
    for t in range(length):
        nxt = np.zeros_like(state)
        to_corrupt = e * (1.0 - v[t])
        nxt[0] += state[0] * (1.0 - to_corrupt)
        nxt[1] += state[0] * to_corrupt
        for k in range(1, length + 1):
            if state[k] <= 0:
                continue
            recover = v[t] * scope * (scope_decay ** (k - 1))
            nxt[0] += state[k] * recover
            nxt[min(k + 1, length + 1)] += state[k] * (1.0 - recover)
        state = nxt
    return float(1.0 - state[0])


def optimal_schedule(
    length: int,
    total_budget: float,
    scope: float,
    local_error_rate: float,
    *,
    scope_decay: float = 1.0,
    outcome: OutcomeModel = "terminal",
    n_rounds: int = 400,
    seed: int = 0,
) -> list[float]:
    """Optimise the per-step schedule under a total-budget constraint.

    Uses PAIRWISE EXCHANGE, not coordinate descent. Adjusting one coordinate at
    a time cannot work here: the budget constraint is tight, so any increase is
    rejected and any decrease raises the error, leaving the initial point
    unchanged. Moving mass BETWEEN steps keeps the budget exact and actually
    explores.
    """
    rng = np.random.default_rng(seed)
    v = np.full(length, total_budget / length, dtype=float)

    def err(x):
        return final_error_probability(
            length, list(x), scope, local_error_rate,
            scope_decay=scope_decay, outcome=outcome,
        )

    best = err(v)
    for step_size in (0.20, 0.10, 0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for _ in range(n_rounds):
                i, j = rng.integers(0, length, size=2)
                if i == j:
                    continue
                delta = min(step_size, v[i], 1.0 - v[j])
                if delta <= 1e-9:
                    continue
                trial = v.copy()
                trial[i] -= delta
                trial[j] += delta
                e_trial = err(trial)
                if e_trial < best - 1e-12:
                    v, best = trial, e_trial
                    improved = True
    return list(v)
