"""Pins probe round two, including the claim it refuted.

C9c said 0.6968 was "a floor, not a ceiling", on the strength of a learning
curve that turned out to have been scored on the selection split. The
correction matters more than the original claim did, so it gets the test: if a
future change makes the held-out curve climb again, that is either a real
finding or a reintroduced leak, and either way it should fail loudly.
"""

import json
from pathlib import Path

import pytest

RUN = Path("runs/probe_variants.json")

needs_run = pytest.mark.skipif(
    not RUN.exists(), reason="run scripts/exp_probe_variants.py first")


def blob():
    return json.loads(RUN.read_text(encoding="utf-8"))


def by_variant():
    return {r["variant"][0]: r for r in blob()["variants"]}


@needs_run
def test_more_training_data_does_not_rescue_the_probe():
    """C9c refuted. Doubling the training set leaves the held-out AUROC alone."""
    v = by_variant()
    assert v["A"]["auroc_test"] == pytest.approx(0.6968, abs=0.002)
    assert v["B"]["n_train"] > 2 * v["A"]["n_train"] * 0.9   # roughly doubled
    assert v["B"]["auroc_test"] < v["A"]["auroc_test"] + 0.02, (
        "if more data now helps materially, C9c may be back -- check whether "
        "the curve is being scored on held-out data before believing it")


@needs_run
def test_the_held_out_learning_curve_is_flat():
    """The evidence behind the refutation, not just its conclusion."""
    curve = blob()["learning_curve_pooled"]
    assert len(curve) >= 4
    aurocs = [a for _, a in curve]
    assert max(aurocs) - min(aurocs) < 0.08, "a flat curve, not a climbing one"
    slope = ((curve[-1][1] - curve[-2][1])
             / max(1, curve[-1][0] - curve[-2][0])) * 1000
    assert slope < 0.005, "the last segment must not be climbing"


@needs_run
def test_training_on_first_bad_steps_doubles_first_bad_recall():
    """C12. The constructive half of 7.6, and the direction is the claim."""
    v = by_variant()
    assert v["C"]["first_bad_recall"] > 1.7 * v["B"]["first_bad_recall"]
    d = blob()["pooled_first_bad_recall_diff"]
    assert d["ci"][0] > 0, "the interval must exclude zero for this to be a result"
    assert d["p_better"] >= 0.95


@needs_run
def test_the_right_target_costs_auroc():
    """The trade is the point: ranked by AUROC, variant C is the worst of them,
    and it is the one that best does the job the system is paid for."""
    v = by_variant()
    assert v["C"]["auroc_test"] < v["B"]["auroc_test"] - 0.05
    assert v["C"]["auroc_test"] == min(r["auroc_test"] for r in blob()["variants"])
    assert v["C"]["first_bad_recall"] == max(
        r["first_bad_recall"] for r in blob()["variants"])


@needs_run
def test_residualising_position_reduces_the_positional_component():
    """Variant D isolates 7.4's mechanism rather than asserting it."""
    v = by_variant()
    assert abs(v["D"]["score_position_corr"]) < abs(v["B"]["score_position_corr"])
    assert v["D"]["first_bad_recall"] > v["B"]["first_bad_recall"]


@needs_run
def test_the_gate_safe_result_is_reported_as_underpowered():
    """C12c. The significant variant saw the calibration split; the deployable
    one trains on 27 positives and its interval spans zero. Both must stay in
    the artifact so neither can be quoted as the other."""
    b = blob()
    gs = b["gate_safe"]
    assert gs["firstbad"]["n_pos_train"] < 40
    assert gs["first_bad_recall_diff"]["ci"][0] < 0 < gs["first_bad_recall_diff"]["ci"][1]
    assert b["pooled_first_bad_recall_diff"]["ci"][0] > 0
    assert b["n_first_bad_test"] == 40


@needs_run
def test_the_first_bad_probe_does_not_make_the_gate_pay():
    """It moves PROJ accuracy the right way and still sits under the baseline.
    Reporting the gain without this would overstate the finding."""
    g = blob()["gate"]
    assert g["probe_firstbad"]["first_bad_recall"] > g["probe_global"]["first_bad_recall"]
    assert g["probe_firstbad"]["proj_acc"] > g["probe_global"]["proj_acc"]
    assert g["probe_firstbad"]["proj_acc"] < g["baseline_no_gate"]
    # and it costs selective risk, which a summary could quietly drop
    assert g["probe_firstbad"]["sel_risk"] > g["probe_global"]["sel_risk"]
