"""Pins the null, and the control that produced it.

The first draft of this experiment reported "+0.0250, the largest of three
levers" by comparing each arm's own best layer. That is instrument PLUS layer
re-selection -- the same confound this section had already corrected twice, and
the tests written alongside it pinned the confounded number. It never shipped.

So these tests guard the control rather than the result: a comparison that
claims to isolate the instrument must hold the layer fixed, and the reason the
answer is null must stay visible.
"""

import json
from pathlib import Path

import pytest

RUN = Path("runs/probe_nonlinear.json")

needs_run = pytest.mark.skipif(
    not RUN.exists(), reason="run scripts/exp_probe_nonlinear.py first")


def blob():
    return json.loads(RUN.read_text(encoding="utf-8"))


def matched():
    return [(tag, where, m) for tag, v in blob()["results"].items()
            for where, m in v.get("matched_layer", {}).items()]


@needs_run
def test_the_instrument_is_compared_at_a_fixed_layer():
    """The control. Without it the delta carries layer re-selection, which is
    how the first draft got a wrong headline."""
    ms = matched()
    assert len(ms) >= 4, "both arms' layers must be tested at both sizes"
    for tag, where, m in ms:
        assert {"layer", "auroc_linear", "auroc_nonlinear", "delta"} <= set(m)


@needs_run
def test_no_identifiable_instrument_effect():
    """C13. The sign depends on which layer is held fixed, so no claim about
    linear vs non-linear survives. If this ever becomes one-signed AND
    significant, that is a real finding and the chapter must change."""
    means = [m["delta"]["mean"] for _, _, m in matched()]
    assert min(means) < 0 < max(means), (
        "the sign no longer flips across layers -- re-read 7.6 before "
        "believing any instrument claim")
    sig = [m for _, _, m in matched() if m["delta"]["ci"][0] > 0 or m["delta"]["ci"][1] < 0]
    assert len(sig) <= 1, "more than one significant matched comparison"
    if sig:
        assert sig[0]["delta"]["ci"][1] < 0, "the significant one favours LINEAR"


@needs_run
def test_at_the_published_layer_the_instrument_is_worth_almost_nothing():
    """Layer 25 is the configuration ch. 7.6 actually reports."""
    m = next(m for tag, where, m in matched()
             if tag == "dev" and where == "linear_layer")
    assert m["layer"] == 25
    assert m["delta"]["mean"] == pytest.approx(0.0051, abs=0.002)
    assert m["delta"]["ci"][0] < 0 < m["delta"]["ci"][1]
    # smaller than the data lever it was once claimed to be 2.4x larger than
    assert m["delta"]["mean"] < 0.0105


@needs_run
def test_layer_choice_is_noisier_than_any_effect_measured():
    """C13b, and the reason the answer is null rather than merely negative."""
    worst = max((v["layer_resolution"] for v in blob()["results"].values()),
                key=lambda r: r["test_spread"])
    assert worst["test_spread"] > 3 * worst["selection_spread"]
    assert worst["test_spread"] > 0.05
    means = [abs(m["delta"]["mean"]) for _, _, m in matched()]
    assert worst["test_spread"] > max(means), (
        "an arbitrary layer choice must swamp the instrument effect; if it no "
        "longer does, the resolution argument in 7.6 is stale")


@needs_run
def test_the_confounded_comparison_is_kept_and_not_quoted_as_the_effect():
    """`delta_auroc` is each arm's argmax against the other's -- retained so the
    confound is auditable, and it must stay larger than the matched number so
    the difference between them is visible rather than argued."""
    for tag, v in blob()["results"].items():
        assert "delta_auroc" in v
        same = v["matched_layer"]["linear_layer"]["delta"]["mean"]
        assert v["delta_auroc"]["mean"] > same, tag


@needs_run
def test_the_linear_arm_paid_for_its_layer_choice():
    """The concrete cost: 0.0012 of selection AUROC bought 0.0749 of test."""
    rows = blob()["results"]["pooled"]["layer_resolution"]["rows"]
    rows = sorted(rows, key=lambda r: -r[1])
    top, second = rows[0], rows[1]
    assert top[1] - second[1] < 0.005, "the top two layers are near-tied on selection"
    assert second[2] - top[2] > 0.05, "and the rejected one is far better on test"


@needs_run
def test_no_lever_approaches_the_reprobe_prediction():
    """C13c. 0.9033 needs +0.21 from 0.6968; every measured lever is <= 0.011."""
    m = next(m for tag, where, m in matched()
             if tag == "dev" and where == "linear_layer")
    assert 0.9033 - 0.6968 > 20 * abs(m["delta"]["mean"])
