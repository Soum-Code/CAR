"""Tests for GSM8K loading and derived dependency-graph extraction.

Parsing tests run everywhere; corpus tests skip without the download.

GSM8K is the primary benchmark, and unlike StrategyQA its dependency graph is
DERIVED rather than annotated, so the extraction logic carries more weight and
is tested harder.
"""

from pathlib import Path

import pytest

from car.data import PRIMARY, DATASETS
from car.data.gsm8k import (
    extraction_stats,
    final_answer,
    load_dags,
    load_examples,
    parse_calc_steps,
    question_numbers,
    solution_to_dag,
)
from car.data.strategyqa import classify_shape, dag_stats

DATA = Path("data/raw/gsm8k/train.jsonl")
needs_data = pytest.mark.skipif(not DATA.exists(), reason="GSM8K not downloaded")


# ---- registry --------------------------------------------------------


def test_gsm8k_is_the_primary_benchmark():
    assert PRIMARY == "gsm8k"
    assert "gsm8k" in DATASETS and "strategyqa" in DATASETS


# ---- calculator parsing ----------------------------------------------


def test_parse_calc_steps():
    ans = "Natalia sold 48/2 = <<48/2=24>>24 clips.\nThen 48+24 = <<48+24=72>>72.\n#### 72"
    assert parse_calc_steps(ans) == [("48/2", 24.0), ("48+24", 72.0)]


def test_final_answer():
    assert final_answer("blah\n#### 72") == "72"


def test_question_numbers_extracts_givens():
    assert question_numbers("He buys 2 large and 16 small") == {2.0, 16.0}


# ---- dependency derivation -------------------------------------------


def test_chain_dependency_is_derived():
    """Line 2 consumes line 1's result, so it depends on it."""
    q = "Natalia sold 48 clips."
    a = "48/2 = <<48/2=24>>24\n48+24 = <<48+24=72>>72\n#### 72"
    dag, _ = solution_to_dag(q, a)
    assert dag.parents == ((), (0,))
    assert classify_shape(dag) == "chain"


def test_converging_dependency_is_derived():
    """The structure that makes GSM8K more than a deeper chain.

    Two independent subtotals computed from givens, then combined.
    """
    q = "He buys 2 large pizzas and 2 small. Large has 16 slices, small has 8."
    a = ("2 x 16 = <<2*16=32>>32\n"
         "2 x 8 = <<2*8=16>>16\n"
         "32 + 16 = <<32+16=48>>48\n#### 48")
    dag, _ = solution_to_dag(q, a)
    assert dag.parents == ((), (), (0, 1))
    assert classify_shape(dag) == "converging"
    assert dag_stats(dag)["longest_path"] == 2


def test_operands_matching_no_earlier_result_are_roots():
    q = "There are 10 yellow."
    a = "80/100 * 10 = <<80/100*10=8>>8\n#### 8"
    dag, stats = solution_to_dag(q, a + "\n5+5 = <<5+5=10>>10")
    # First step's operands (80, 100, 10) are all givens -> no parents.
    assert dag.parents[0] == ()


def test_most_recent_producer_wins():
    """When two earlier lines produce the same value, link to the nearer one."""
    q = "start"
    a = ("1+1 = <<1+1=5>>5\n"
         "2+3 = <<2+3=5>>5\n"
         "5+0 = <<5+0=9>>9\n#### 9")
    dag, _ = solution_to_dag(q, a)
    assert dag.parents[2] == (1,)


def test_single_step_solutions_are_rejected():
    """No dependency structure to study; including them would only dilute the
    corpus statistics."""
    dag, stats = solution_to_dag("q", "2+2 = <<2+2=4>>4\n#### 4")
    assert dag is None
    assert stats["n_steps"] == 1


def test_derived_graph_is_topological():
    q = "q"
    a = ("1+1 = <<1+1=2>>2\n2+2 = <<2+2=4>>4\n4+2 = <<4+2=6>>6\n#### 6")
    dag, _ = solution_to_dag(q, a)
    for v, ps in enumerate(dag.parents):
        assert all(p < v for p in ps)


def test_ambiguous_link_is_counted_not_hidden():
    """An operand matching both an earlier result and a question number is
    ambiguous. It is still linked, but the rate must be visible."""
    q = "He has 16 slices"
    a = "2*8 = <<2*8=16>>16\n16+1 = <<16+1=17>>17\n#### 17"
    dag, stats = solution_to_dag(q, a)
    assert dag.parents[1] == (0,)
    assert stats["n_ambiguous"] >= 1


# ---- corpus ----------------------------------------------------------


@needs_data
def test_corpus_has_more_headroom_than_strategyqa():
    """The reason for the switch. Fraction of steps with a non-terminal
    descendant: StrategyQA 11.2%, GSM8K 26.6%."""
    dags = load_dags(DATA)
    total = inter = 0
    for d in dags:
        for v in range(d.n):
            total += 1
            if d.descendants(v) - {d.terminal}:
                inter += 1
    assert inter / total > 0.20


@needs_data
def test_corpus_reaches_real_depth():
    dags = load_dags(DATA)
    depths = [dag_stats(d)["longest_path"] for d in dags]
    assert max(depths) >= 6, "GSM8K should have a genuine depth tail"
    assert sum(1 for d in depths if d >= 3) > 2500, "enough deep items to study"


@needs_data
def test_extraction_quality_is_acceptable():
    st = extraction_stats(DATA)
    assert st["usable"] / st["rows"] > 0.9
    # Ambiguity is real but bounded; recorded so a regression is visible.
    assert st["ambiguous_rate"] < 0.15


@needs_data
def test_examples_load_with_step_decomposition():
    exs = load_examples(DATA)
    assert len(exs) > 7000
    assert all(e.gold_answer for e in exs)
    assert any(len(e.decomposition) >= 4 for e in exs)


# ---- operand extraction (regression) ---------------------------------


def test_subtraction_operator_is_not_read_as_a_sign():
    r"""THE bug hand-validation found.

    The old regex `-?\d+` captured the subtraction operator as part of the
    operand, so "110-80" yielded [110, -80]. The -80 matched no earlier
    result, and every subtraction silently lost its dependency edge.
    """
    from car.data.gsm8k import expression_operands

    assert expression_operands("110-80") == [110.0, 80.0]
    assert expression_operands("10-9") == [10.0, 9.0]
    assert sorted(expression_operands("30-10-15")) == [10.0, 15.0, 30.0]


def test_leading_dot_decimals_survive():
    """The same regex dropped the leading dot, turning .8 into 8."""
    from car.data.gsm8k import expression_operands

    assert expression_operands("100*.8") == [100.0, 0.8]


def test_subtraction_edge_is_actually_derived():
    """End-to-end: the edge the old extractor lost must now exist."""
    q = "Natalia had 110 clips."
    a = "50+30 = <<50+30=80>>80\n110-80 = <<110-80=30>>30\n#### 30"
    dag, _ = solution_to_dag(q, a)
    assert dag.parents[1] == (0,), "L2 consumes L1's result via subtraction"


@needs_data
def test_fix_increases_link_rate_and_depth():
    """Corpus-level effect of the fix, so a regression is visible.

    Before: link rate 0.307, orphan steps 0.277, mean longest path 2.54.
    """
    from car.data.strategyqa import corpus_summary

    st = extraction_stats(DATA)
    assert st["link_rate"] > 0.33
    assert st["orphan_step_rate"] < 0.23
    assert corpus_summary(load_dags(DATA))["longest_path"]["mean"] > 2.7
