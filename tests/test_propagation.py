"""Tests for the error-propagation model.

These lock in the findings from scripts/exp_propagation.py and
exp_allocation.py, including the ones that went against the project's own
hypotheses. A negative result that silently stops reproducing is worse than no
result at all.
"""

import numpy as np
import pytest

from car.propagation import (
    PropagationChain,
    always,
    final_error_probability,
    never,
    optimal_schedule,
    schedule_policy,
    uniform_random,
)

LENGTH = 8
LOCAL_ERR = 0.15


def _evaluate(chain, policy, n=8000, seed=1, outcome="terminal"):
    rng = np.random.default_rng(seed)
    err, local, calls = [], [], []
    for _ in range(n):
        r = chain.run(policy, rng, budget=chain.length)
        err.append(not r.final_correct(outcome))
        local.append(r.local_selective_risk)
        calls.append(r.verification_calls)
    return float(np.mean(err)), float(np.mean(local)), float(np.mean(calls))


def _shapes(length, budget):
    f = np.array([length - i - 1 for i in range(length)], dtype=float)
    b = np.array([i + 1 for i in range(length)], dtype=float)
    return {
        "front": list(np.clip(f / f.sum() * budget, 0, 1)),
        "uniform": [budget / length] * length,
        "back": list(np.clip(b / b.sum() * budget, 0, 1)),
    }


# ---- the core structural claim ---------------------------------------


def test_local_risk_does_not_track_final_error():
    """THE finding: controlling local selective risk bounds nothing globally.

    With a purely local verifier, local risk stays pinned near the local error
    rate across budgets while final-answer error moves enormously. Any method
    that certifies local step correctness is therefore certifying the wrong
    quantity.
    """
    chain = PropagationChain(
        length=LENGTH, local_error_rate=LOCAL_ERR, verifier_scope=0.0
    )
    locals_, finals = [], []
    for rate in (0.0, 0.25, 0.5, 0.75):
        pol = never if rate == 0.0 else uniform_random(rate)
        fin, loc, _ = _evaluate(chain, pol)
        locals_.append(loc)
        finals.append(fin)

    # Local risk barely moves...
    assert max(locals_) - min(locals_) < 0.05, f"local risk moved: {locals_}"
    # ...while final error moves by a lot.
    assert max(finals) - min(finals) > 0.35, f"final error flat: {finals}"


def test_corrupt_state_is_absorbing_for_a_local_verifier():
    """scope=0 means inherited corruption can never be detected.

    Once committed, no amount of verification recovers it -- the only lever a
    local verifier has is preventing the error in the first place.
    """
    chain = PropagationChain(
        length=6, local_error_rate=1.0, verifier_scope=0.0
    )
    rng = np.random.default_rng(0)
    # Every step is locally invalid, and we verify nothing at step 0, so
    # corruption enters immediately and must persist.
    sched = [0.0] + [1.0] * 5
    r = chain.run(schedule_policy(sched), rng, budget=6)
    assert not r.final_correct("terminal")
    assert all(s.premise_corrupt for s in r.steps[1:])


def test_global_verifier_can_escape_corruption():
    chain = PropagationChain(
        length=6, local_error_rate=1.0, verifier_scope=1.0, scope_decay=1.0
    )
    rng = np.random.default_rng(0)
    fin, _, _ = _evaluate(chain, always, n=2000)
    assert fin < 0.5, "a global verifier at every step should recover"


# ---- closed form vs simulation ---------------------------------------


@pytest.mark.parametrize("scope", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("decay", [1.0, 0.377])
@pytest.mark.parametrize("v", [0.25, 0.5])
def test_closed_form_matches_simulation(scope, decay, v):
    chain = PropagationChain(
        length=LENGTH,
        local_error_rate=LOCAL_ERR,
        verifier_scope=scope,
        scope_decay=decay,
    )
    cf = final_error_probability(
        LENGTH, v, scope, LOCAL_ERR, scope_decay=decay, outcome="terminal"
    )
    sim, _, _ = _evaluate(chain, uniform_random(v), n=20000)
    assert abs(cf - sim) < 0.015, f"closed form {cf:.4f} vs sim {sim:.4f}"


def test_conjunctive_outcome_ignores_scope():
    """Under the conjunctive model a repair cannot undo a committed error, so
    the verifier's reach is irrelevant and only prevention matters."""
    a = final_error_probability(LENGTH, 0.4, 0.0, LOCAL_ERR, outcome="conjunctive")
    b = final_error_probability(LENGTH, 0.4, 1.0, LOCAL_ERR, outcome="conjunctive")
    assert a == pytest.approx(b)


def test_decay_makes_late_verification_less_effective():
    fast = final_error_probability(LENGTH, 0.4, 1.0, LOCAL_ERR, scope_decay=0.377)
    none = final_error_probability(LENGTH, 0.4, 1.0, LOCAL_ERR, scope_decay=1.0)
    assert fast > none


# ---- the negative result ---------------------------------------------


@pytest.mark.parametrize("length", [4, 8, 16])
def test_front_loading_loses_to_back_loading(length):
    """Spec hypothesis H3 ('verify early') is NOT supported.

    Front-loaded allocation loses to back-loaded at matched budget, and the gap
    widens with chain length. Recorded as a test because it contradicts the
    project's stated hypothesis and our own earlier proposal, so it must keep
    reproducing rather than quietly drifting.
    """
    s = _shapes(length, 0.375 * length)
    front = final_error_probability(length, s["front"], 1.0, LOCAL_ERR)
    back = final_error_probability(length, s["back"], 1.0, LOCAL_ERR)
    assert back < front


@pytest.mark.parametrize("repair", [1.0, 0.5, 0.05])
def test_back_loading_wins_across_repair_rates(repair):
    """Robustness: the result is not an artifact of assuming perfect repair."""
    s = _shapes(LENGTH, 3.0)
    chain_kwargs = dict(
        length=LENGTH,
        local_error_rate=LOCAL_ERR,
        verifier_scope=1.0,
        repair_success=repair,
    )
    front, _, _ = _evaluate(
        PropagationChain(**chain_kwargs), schedule_policy(s["front"]), n=8000
    )
    back, _, _ = _evaluate(
        PropagationChain(**chain_kwargs), schedule_policy(s["back"]), n=8000
    )
    assert back < front


def test_regeneration_cost_favours_early_but_not_enough():
    """Early verification does save regeneration work -- just not enough.

    Front-loading wastes less thrown-away generation, which is the one honest
    argument in its favour. The effect is real but an order of magnitude
    smaller than the accuracy gap it costs.
    """
    s = _shapes(LENGTH, 3.0)
    chain = PropagationChain(
        length=LENGTH, local_error_rate=LOCAL_ERR, verifier_scope=1.0
    )

    def regen_and_err(sched):
        rng = np.random.default_rng(1)
        rs, es = [], []
        for _ in range(8000):
            r = chain.run(schedule_policy(sched), rng, budget=LENGTH)
            rs.append(r.regenerated_steps)
            es.append(not r.final_correct("terminal"))
        return float(np.mean(rs)), float(np.mean(es))

    front_regen, front_err = regen_and_err(s["front"])
    back_regen, back_err = regen_and_err(s["back"])

    assert front_regen < back_regen, "front-loading should waste less work"
    # But the accuracy gap dwarfs the cost saving.
    assert (front_err - back_err) > (back_regen - front_regen) / LENGTH


# ---- optimiser -------------------------------------------------------


def test_optimiser_beats_the_naive_shapes():
    """Regression: an earlier coordinate-descent version could not move off its
    starting point under the budget constraint and always returned uniform."""
    s = _shapes(LENGTH, 3.0)
    best_naive = min(
        final_error_probability(LENGTH, v, 1.0, LOCAL_ERR) for v in s.values()
    )
    opt = optimal_schedule(LENGTH, 3.0, 1.0, LOCAL_ERR)
    assert final_error_probability(LENGTH, opt, 1.0, LOCAL_ERR) <= best_naive + 1e-9
    assert sum(opt) <= 3.0 + 1e-6, "optimiser must respect the budget"


def test_optimiser_respects_budget_exactly():
    opt = optimal_schedule(6, 2.0, 0.5, LOCAL_ERR)
    assert sum(opt) <= 2.0 + 1e-6
    assert all(0.0 <= x <= 1.0 for x in opt)
