"""Tests for StrategyQA dependency-graph extraction.

Parsing tests run everywhere. Corpus-level tests skip when the dataset has not
been downloaded, so a fresh clone still gets a green suite.
"""

from pathlib import Path

import pytest

from car.data.strategyqa import (
    classify_shape,
    corpus_summary,
    dag_stats,
    decomposition_to_dag,
    load_dags,
    load_examples,
    parse_references,
)

DATA = Path("data/raw/strategyqa/strategyqa_train.json")
needs_data = pytest.mark.skipif(
    not DATA.exists(), reason="StrategyQA not downloaded"
)


# ---- reference parsing -----------------------------------------------


def test_parse_references_is_zero_based():
    assert parse_references("Is #2 greater than #1?") == [1, 0]


def test_parse_references_handles_none():
    assert parse_references("How many kids did Julius Caesar have?") == []


def test_parse_references_handles_multi_digit():
    assert parse_references("combine #10 and #2") == [9, 1]


# ---- DAG construction ------------------------------------------------


def test_canonical_converging_decomposition():
    """The single most common StrategyQA structure (980 of 2272 questions)."""
    dec = [
        "How many kids did Julius Caesar have?",
        "How many kids did Genghis Khan have?",
        "Is #2 greater than #1?",
    ]
    dag = decomposition_to_dag(dec)
    assert dag.parents == ((), (), (0, 1))
    assert dag.terminal == 2
    assert classify_shape(dag) == "converging"
    assert dag.influence(0) == 1 and dag.influence(2) == 0


def test_chain_decomposition():
    dec = ["Who can perform lawful arrests?", "Are members of The Police also #1?"]
    dag = decomposition_to_dag(dec)
    assert dag.parents == ((), (0,))
    assert classify_shape(dag) == "chain"


def test_forward_references_are_dropped_not_raised():
    """A few annotations reference later steps. Dropping the edge biases the
    topology statistics less than discarding the whole question would."""
    dag = decomposition_to_dag(["refers to #2 somehow", "second"])
    assert dag.parents == ((), ())  # forward ref removed, no exception


def test_self_reference_is_dropped():
    dag = decomposition_to_dag(["first", "self #2"])
    assert dag.parents[1] == ()


def test_every_step_index_is_topological():
    dec = ["a", "b using #1", "c using #1 and #2", "d using #3"]
    dag = decomposition_to_dag(dec)
    for v, ps in enumerate(dag.parents):
        assert all(p < v for p in ps)


# ---- statistics ------------------------------------------------------


def test_longest_path_measures_depth_not_node_count():
    """The distinction that matters for propagation headroom: three steps in a
    converging shape are only two hops deep."""
    converging = decomposition_to_dag(["a", "b", "c from #1 and #2"])
    chain = decomposition_to_dag(["a", "b from #1", "c from #2"])
    assert dag_stats(converging)["n"] == dag_stats(chain)["n"] == 3
    assert dag_stats(converging)["longest_path"] == 2
    assert dag_stats(chain)["longest_path"] == 3


def test_dead_end_detection():
    # step 2 feeds nothing and is not the answer
    dag = decomposition_to_dag(["a", "b", "c from #1"])
    assert dag_stats(dag)["n_dead_ends"] == 1


# ---- corpus ----------------------------------------------------------


@needs_data
def test_corpus_loads_and_is_shallow():
    """The headline empirical finding: real StrategyQA reasoning has almost no
    propagation depth, so the benchmark cannot exercise the snowball effect."""
    dags = load_dags(DATA)
    s = corpus_summary(dags)
    assert s["n_questions"] > 2000
    assert s["steps"]["mean"] < 3.5
    assert s["longest_path"]["mean"] < 2.5


@needs_data
def test_most_steps_have_no_intermediate_descendant():
    """A step can only corrupt downstream reasoning if downstream reasoning
    exists. For most StrategyQA steps the only descendant is the answer."""
    dags = load_dags(DATA)
    total = intermediate = 0
    for d in dags:
        for v in range(d.n):
            total += 1
            if d.descendants(v) - {d.terminal}:
                intermediate += 1
    assert intermediate / total < 0.2


@needs_data
def test_converging_is_the_dominant_real_shape():
    dags = load_dags(DATA)
    shapes = corpus_summary(dags)["shapes"]
    assert shapes["converging"] > shapes["chain"]


@needs_data
def test_examples_load_with_evidence():
    exs = load_examples(DATA)
    assert len(exs) > 2000
    assert all(e.decomposition for e in exs)
    assert any(e.evidence for e in exs)
