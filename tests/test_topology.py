"""Tests for DAG topologies and topology-aware propagation.

The headline test is `test_influence_equals_front_on_a_chain`: it documents why
the chain-only experiment in FINDINGS-PROPAGATION.md could not evaluate
influence weighting at all, which is what motivated this whole module.

Also guards the two artifacts found and fixed here:
  * decay being silently inert (distance measured to the nearest wrong
    ancestor is always 1 once corruption spreads)
  * the terminal node being verifiable, which makes "check the answer" a
    trivially winning oracle policy
"""

import numpy as np
import pytest

from car.topology import (
    SCHEDULES,
    DAGPropagation,
    bushy,
    chain,
    converging_tree,
    diamond,
    influence_schedule,
    parallel_chains,
    uniform_schedule,
)

LOCAL_ERR = 0.15
TOPOS = [
    chain(10),
    parallel_chains(3, 3),
    converging_tree(3),
    diamond(3, 3),
    bushy(13, seed=0),
]


def _err(dag, sched, scope=1.0, decay=1.0, n=15000, seed=1, repair=1.0):
    sim = DAGPropagation(
        dag,
        local_error_rate=LOCAL_ERR,
        verifier_scope=scope,
        scope_decay=decay,
        repair_success=repair,
    )
    rng = np.random.default_rng(seed)
    return float(
        np.mean([not sim.run(sched, rng, budget=dag.n).terminal_correct for _ in range(n)])
    )


# ---- graph structure -------------------------------------------------


def test_chain_influence_is_position():
    dag = chain(10)
    assert dag.position_influence_correlation() == pytest.approx(-1.0)
    assert [dag.influence(v) for v in range(10)] == list(range(9, -1, -1))


def test_non_chain_topologies_decouple_influence_from_position():
    """The premise of this whole module: on these graphs the two signals are
    distinguishable, so the comparison is meaningful."""
    for dag in (parallel_chains(3, 3), bushy(13, seed=0)):
        assert abs(dag.position_influence_correlation()) < 0.8, dag.name


def test_parents_must_be_topological():
    from car.topology import DAG

    with pytest.raises(ValueError, match="non-topological"):
        DAG(n=3, parents=((), (2,), ()), terminal=2)


def test_descendants_and_ancestors_are_transitive():
    dag = chain(5)
    assert dag.descendants(0) == frozenset({1, 2, 3, 4})
    assert dag.ancestors(4) == frozenset({0, 1, 2, 3})
    assert dag.descendants(4) == frozenset()


def test_diamond_does_not_double_count_shared_descendants():
    dag = diamond(1, 2)  # 0 -> {1,2} -> 3
    assert dag.influence(0) == 3


def test_all_topologies_reach_their_terminal():
    for dag in TOPOS:
        for v in range(dag.n):
            assert dag.reaches_terminal(v), f"{dag.name}: node {v} is orphaned"


# ---- the confound ----------------------------------------------------


def test_influence_equals_front_on_a_chain():
    """THE reason the earlier rejection of influence weighting was invalid.

    On a chain, descendant count decreases monotonically with position, so the
    two schedules are numerically identical and no experiment on chains can
    tell them apart.
    """
    dag = chain(10)
    b = 0.375 * dag.n
    infl = influence_schedule(dag, b)
    front = SCHEDULES["front"](dag, b)
    assert infl == pytest.approx(front)


def test_influence_differs_from_front_off_chain():
    dag = parallel_chains(3, 3)
    b = 0.375 * dag.n
    assert influence_schedule(dag, b) != pytest.approx(SCHEDULES["front"](dag, b))


# ---- artifact guards -------------------------------------------------


def test_terminal_is_not_verifiable_by_default():
    """A verifier that can check the final answer is an oracle on the task.

    Allowing it drove measured error to exactly 0.0000 for any schedule that
    put mass on the last node -- an artifact that made three policies look
    perfect.
    """
    dag = parallel_chains(3, 3)
    sim = DAGPropagation(dag)
    assert dag.terminal not in sim.verifiable

    for name in SCHEDULES:
        sched = SCHEDULES[name](dag, 0.375 * dag.n)
        assert sched[dag.terminal] == 0.0, f"{name} spends budget on the terminal"


def test_schedules_spend_the_same_budget():
    """Otherwise any comparison is just 'verify more'."""
    for dag in TOPOS:
        b = 0.375 * dag.n
        totals = [sum(SCHEDULES[n](dag, b)) for n in SCHEDULES]
        assert max(totals) - min(totals) < 1e-6, dag.name
        assert totals[0] == pytest.approx(b)


def test_decay_actually_bites():
    """Regression: measuring distance to the NEAREST wrong ancestor always
    gives 1 once corruption spreads, silently disabling decay."""
    dag = parallel_chains(3, 3)
    sched = uniform_schedule(dag, 0.375 * dag.n)
    strong = _err(dag, sched, decay=1.0)
    weak = _err(dag, sched, decay=0.05)
    assert weak > strong + 0.01, f"decay had no effect: {strong:.4f} vs {weak:.4f}"


def test_origin_distance_grows_along_a_chain():
    dag = chain(6)
    sim = DAGPropagation(dag, local_error_rate=1.0, verifier_scope=0.0)
    rng = np.random.default_rng(0)
    r = sim.run([0.0] * dag.n, rng, budget=0)
    # Every node is locally invalid, so each starts its own corruption.
    assert all(d == 0 for d in r.origin_dist.values())


def test_repair_cuts_only_paths_through_the_repaired_node():
    """The property a chain cannot express, and the reason trees were worth
    testing: parallel routes from the same bad ancestor survive a repair."""
    dag = parallel_chains(2, 2)  # branches {0,1} and {2,3}, merge at 4
    sim = DAGPropagation(dag, local_error_rate=0.0, verifier_scope=1.0)
    rng = np.random.default_rng(0)
    r = sim.run([0.0] * dag.n, rng, budget=0)
    assert r.terminal_correct  # no errors at all -> clean


# ---- the result ------------------------------------------------------


@pytest.mark.parametrize("dag", TOPOS, ids=lambda d: d.name)
def test_influence_weighting_never_beats_uniform(dag):
    """The finding, now established on topologies that can actually see it.

    Influence weighting is slightly better than front-loading off-chain, but it
    still loses to plain uniform on every topology tested. Recorded because it
    contradicts POSITIONING.md 4.2 and our own proposal.
    """
    b = 0.375 * dag.n
    infl = _err(dag, influence_schedule(dag, b))
    unif = _err(dag, uniform_schedule(dag, b))
    assert infl > unif, f"{dag.name}: influence {infl:.4f} vs uniform {unif:.4f}"


@pytest.mark.parametrize(
    "dag", [parallel_chains(3, 3), converging_tree(3), diamond(3, 3)],
    ids=lambda d: d.name,
)
def test_ancestor_count_beats_descendant_count(dag):
    """The replacement finding: on DAGs the useful structural signal is how
    much upstream reasoning a check SCREENS, not how much downstream damage a
    node could cause."""
    b = 0.375 * dag.n
    depth = _err(dag, SCHEDULES["depth"](dag, b))
    infl = _err(dag, influence_schedule(dag, b))
    assert depth < infl, f"{dag.name}: depth {depth:.4f} vs influence {infl:.4f}"


def test_more_budget_helps_monotonically():
    dag = parallel_chains(3, 3)
    errs = [_err(dag, uniform_schedule(dag, f * dag.n), n=8000)
            for f in (0.15, 0.375, 0.65)]
    assert errs[0] > errs[1] > errs[2]
