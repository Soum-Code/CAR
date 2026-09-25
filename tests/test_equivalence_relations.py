"""Pins the entailment comparison: the reference relation does not rescue it.

Chapter 7's semantic-divergence result has one obvious escape hatch -- that
`numeric_equivalence` is a cheap stand-in and the relation the literature uses
would find the signal it misses. The measurement closes that, and these tests
keep it closed: the numbers are in the thesis, and a change to the clustering,
the splits or the scorer that moves them should fail loudly.

The NLI pass is ~28k pairs and ~20 hours of CPU, so nothing here runs it. The
artifacts are committed and these read them.
"""

import json
from pathlib import Path

import pytest

RUN = Path("runs/equivalence_relations.json")
CACHE = Path("runs/entailment_cache.jsonl")

needs_run = pytest.mark.skipif(
    not RUN.exists(), reason="run scripts/exp_equivalence_relations.py first")
needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason="entailment NLI cache not present")


def by_name():
    return {r["relation"]: r for r in json.loads(RUN.read_text(encoding="utf-8"))}


@needs_run
def test_all_three_relations_land_near_chance():
    """C11. The negative result is not an artifact of the clustering."""
    rows = by_name()
    assert len(rows) == 3
    for name, r in rows.items():
        assert 0.45 < r["auroc_test"] < 0.60, f"{name} moved out of the near-chance band"


@needs_run
def test_the_reference_relation_does_not_rescue_the_signal():
    """The whole point of running it.

    Pins the values, NOT the ordering. A solution-clustered bootstrap over the
    182 test solutions puts the entailment-minus-numeric gap at 95% CI
    [-0.087, +0.059] with P(worse) = 0.63, so asserting `entail < numeric`
    would pin a coin flip and break on a harmless reshuffle. What is real is
    that entailment stays far from usable.
    """
    rows = by_name()
    numeric = rows["numeric equivalence"]["auroc_test"]
    entail = rows["bidirectional entailment"]["auroc_test"]
    assert numeric == pytest.approx(0.5740, abs=0.002)
    assert entail == pytest.approx(0.5625, abs=0.002)
    assert entail < 0.62, "the CI's optimistic end; above this the chapter changes"


@needs_run
def test_entailment_is_the_most_permissive_relation():
    """C11c, and the surprise: a two-sided test merges MORE than a one-sided one.

    If this ever inverts, the paragraph in 7.2 explaining why is wrong.
    """
    rows = by_name()
    assert (rows["bidirectional entailment"]["unanimous"]
            > rows["numeric equivalence"]["unanimous"]
            > rows["exact string match"]["unanimous"])
    assert (rows["bidirectional entailment"]["mean_divergence"]
            < rows["numeric equivalence"]["mean_divergence"]
            < rows["exact string match"]["mean_divergence"])


@needs_cache
def test_the_nli_model_judges_most_pairs_entailing():
    """86.6%. The evidence for calling this relation permissive rather than strict."""
    n = entailed = 0
    with CACHE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            entailed += bool(json.loads(line)["e"])
    assert n == 27_936
    assert entailed / n == pytest.approx(0.866, abs=0.01)


@needs_cache
def test_cache_keys_carry_the_question_context():
    """The bug that would have been silent: a context-blind cache reuses the
    verdict from a different question."""
    with CACHE.open(encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    assert set(rec) == {"c", "a", "b", "e"}
    assert rec["c"], "context must be stored, not dropped"


@needs_cache
def test_entailment_merges_half_the_numerically_conflicting_pairs():
    """C11c's sharp half: the reference relation is a poor fit for arithmetic.

    A pooled agreement rate hides this. Conditioned the other way -- on pairs
    whose asserted numbers actually conflict -- the NLI model calls 52% of them
    mutually entailing, erasing the disagreement the signal depends on.
    """
    import sys
    sys.path.insert(0, "src")
    from car.data.generated import arithmetic_claims
    from car.uncertainty.semantic import _asserted_value

    cache = {}
    with CACHE.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                cache[(r["c"], r["a"], r["b"])] = r["e"]

    agree_mutual = agree_n = dis_mutual = dis_n = 0
    seen = set()
    for (c, a, b) in cache:
        key = (c, a, b) if a < b else (c, b, a)
        if key in seen:
            continue
        seen.add(key)
        va = _asserted_value(key[1], arithmetic_claims)
        vb = _asserted_value(key[2], arithmetic_claims)
        if va is None or vb is None:
            continue
        mutual = cache.get((c, a, b), False) and cache.get((c, b, a), False)
        if abs(va - vb) < 1e-6:
            agree_n += 1
            agree_mutual += mutual
        else:
            dis_n += 1
            dis_mutual += mutual

    assert agree_mutual / agree_n == pytest.approx(0.930, abs=0.01)
    assert dis_mutual / dis_n == pytest.approx(0.520, abs=0.01)
    # The asymmetry is the finding; a pooled rate would read as 85.5%.
    assert agree_mutual / agree_n - dis_mutual / dis_n > 0.35
