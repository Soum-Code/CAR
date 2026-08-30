"""Tests for the gate, budget accounting, and influence weighting."""

import pytest

from car.control import (
    Budget,
    BudgetPolicy,
    ControlGate,
    InfluenceWeighting,
    build_adjacency,
    descendant_counts,
)
from car.types import Decision, ReasoningStep


# ---- gate ------------------------------------------------------------


def test_gate_continues_below_threshold():
    gate = ControlGate()
    out = gate.decide(score=0.2, threshold=0.5)
    assert out.decision == Decision.CONTINUE


def test_gate_verifies_above_threshold():
    gate = ControlGate()
    out = gate.decide(score=0.9, threshold=0.5)
    assert out.decision == Decision.VERIFY
    assert not out.forced_exploration


def test_forced_exploration_overrides_a_confident_step():
    """Exploration must fire even when the gate would happily continue --
    otherwise it samples the same region the gate already observes and provides
    no new information."""
    gate = ControlGate()
    out = gate.decide(score=0.1, threshold=0.5, force_explore=True)
    assert out.decision == Decision.VERIFY
    assert out.forced_exploration
    # Still an accept-region step: that is what makes its label valuable.
    assert ControlGate.was_accepted(out)


def test_was_accepted_distinguishes_gate_opinion_from_action():
    gate = ControlGate()
    explored = gate.decide(score=0.1, threshold=0.5, force_explore=True)
    fired = gate.decide(score=0.9, threshold=0.5)

    assert explored.decision == fired.decision == Decision.VERIFY
    # Same action, opposite provenance. Conflating these breaks the calibrator.
    assert ControlGate.was_accepted(explored)
    assert not ControlGate.was_accepted(fired)


def test_budget_exhaustion_forces_continue_and_is_recorded():
    gate = ControlGate()
    budget = Budget(0)
    out = gate.decide(score=0.9, threshold=0.5, budget=budget)
    assert out.decision == Decision.CONTINUE
    assert out.budget_blocked, "a forced continue must be distinguishable"


def test_abstain_above_abstain_threshold():
    gate = ControlGate(abstain_threshold=2.0)
    out = gate.decide(score=3.0, threshold=0.5)
    assert out.decision == Decision.ABSTAIN


def test_influence_raises_gate_value_for_load_bearing_steps():
    """Same uncertainty, more dependents -> more likely to be verified."""
    gate = ControlGate(influence_weighting=InfluenceWeighting(mode="sqrt"))
    isolated = gate.decide(score=0.4, threshold=0.5, n_descendants=0)
    load_bearing = gate.decide(score=0.4, threshold=0.5, n_descendants=9)

    assert isolated.decision == Decision.CONTINUE
    assert load_bearing.decision == Decision.VERIFY
    assert load_bearing.gate_value > isolated.gate_value


def test_influence_mode_none_is_a_pure_uncertainty_gate():
    gate = ControlGate(influence_weighting=InfluenceWeighting(mode="none"))
    a = gate.decide(score=0.4, threshold=0.5, n_descendants=0)
    b = gate.decide(score=0.4, threshold=0.5, n_descendants=20)
    assert a.gate_value == b.gate_value


# ---- influence -------------------------------------------------------


def _chain(n: int) -> list[ReasoningStep]:
    return [
        ReasoningStep(step_id=i, claim=f"c{i}", dependency_ids=[i - 1] if i else [])
        for i in range(n)
    ]


def test_descendant_counts_on_a_chain():
    """In a 5-chain, step 0 supports 4 later steps and the last supports none."""
    counts = descendant_counts(_chain(5))
    assert counts == {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}


def test_descendant_counts_on_a_diamond():
    #   0 -> 1, 0 -> 2, (1,2) -> 3
    steps = [
        ReasoningStep(step_id=0, claim="root"),
        ReasoningStep(step_id=1, claim="left", dependency_ids=[0]),
        ReasoningStep(step_id=2, claim="right", dependency_ids=[0]),
        ReasoningStep(step_id=3, claim="join", dependency_ids=[1, 2]),
    ]
    counts = descendant_counts(steps)
    assert counts[0] == 3  # transitive, and 3 is not double counted
    assert counts[1] == 1
    assert counts[3] == 0


def test_early_steps_carry_more_influence():
    """Why 'verify early' falls out of the framework instead of being a
    separate empirical claim: earlier steps have more descendants."""
    counts = descendant_counts(_chain(8))
    assert counts[0] > counts[4] > counts[7]


def test_build_adjacency_maps_children():
    children = build_adjacency(_chain(3))
    assert children == {0: [1], 1: [2], 2: []}


def test_influence_weighting_is_capped():
    w = InfluenceWeighting(mode="linear", cap=4.0)
    assert w(100) == 4.0


def test_influence_rejects_unknown_mode():
    with pytest.raises(ValueError):
        InfluenceWeighting(mode="quadratic")


# ---- schema ----------------------------------------------------------


def test_step_rejects_self_dependency():
    with pytest.raises(ValueError, match="itself"):
        ReasoningStep(step_id=2, claim="x", dependency_ids=[2])


def test_step_rejects_forward_dependency():
    """A step may only rest on earlier steps; otherwise the graph is not a
    DAG in generation order and influence is not computable online."""
    with pytest.raises(ValueError, match="later step"):
        ReasoningStep(step_id=1, claim="x", dependency_ids=[5])


# ---- budget ----------------------------------------------------------


def test_budget_tracks_exploration_separately():
    """Exploration calls are a cost of the method and must stay visible in any
    accuracy-vs-cost comparison."""
    b = Budget(5)
    b.spend(1, exploration=True)
    b.spend(1, exploration=False)
    s = b.summary()
    assert s["spent"] == 2
    assert s["spent_on_exploration"] == 1
    assert s["spent_on_gate"] == 1


def test_budget_refuses_overspend():
    b = Budget(1)
    assert b.spend(1)
    assert not b.spend(1)
    assert b.exhausted


def test_budget_policy_tightens_as_budget_drains():
    b = Budget(10)
    p = BudgetPolicy(mode="linear", strength=0.5)
    fresh = p.scale(b)
    for _ in range(9):
        b.spend()
    drained = p.scale(b)
    assert drained > fresh
