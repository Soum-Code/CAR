"""Tests for the score-quality dial and the crossing it measures.

Two separable things need protecting.

The dial itself must be exact: the whole experiment reads "at AUROC x the
verifier is worth y", and if `separation_for_auroc` were off by even a little
the x-axis would be mislabelled and the crossing would move. That is checkable
in closed form and does not need the corpus.

The crossing and the position-bias finding are pinned against the committed run
artifact, in the same spirit as every other refuted claim here: the numbers are
in the thesis, so a change to the scorer, the splits or the calibrator that
moves them should fail loudly rather than quietly rewrite chapter 7.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from car.eval.metrics import step_detection_auroc
from car.uncertainty.synthetic import SyntheticAUROCScorer, separation_for_auroc

RUN = Path("runs/score_quality_threshold.json")

needs_run = pytest.mark.skipif(
    not RUN.exists(),
    reason="run scripts/exp_score_quality_threshold.py first",
)


# ---- the dial -------------------------------------------------------------


@pytest.mark.parametrize("target", [0.55, 0.65, 0.75, 0.90, 0.99])
def test_separation_produces_the_requested_auroc(target):
    """Binormal inversion: AUROC = Phi(d / sqrt(2)), so d = sqrt(2) Phi^-1(a)."""
    d = separation_for_auroc(target)
    rng = np.random.default_rng(0)
    n = 200_000
    correct = rng.random(n) > 0.4
    scores = rng.normal(np.where(correct, 0.0, d), 1.0)
    assert step_detection_auroc(scores, correct) == pytest.approx(target, abs=0.005)


def test_auroc_half_means_no_separation():
    assert separation_for_auroc(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [0.0, 0.49, 1.0, 1.5])
def test_out_of_range_auroc_raises(bad):
    """1.0 needs infinite separation and <0.5 is a score pointing backwards.

    Both are call-site bugs. Clipping them would silently produce a sweep whose
    x-axis does not mean what the figure says it means.
    """
    with pytest.raises(ValueError):
        separation_for_auroc(bad)


def test_scorer_is_keyed_on_identity_not_value():
    """Two steps with equal features must still get independent scores.

    `ReplayStepGenerator` hands back the same feature object on each visit, and
    distinct steps can carry numerically identical features -- keying on value
    would collapse them onto one score and quietly correlate the draw with
    whatever made them identical.
    """

    class Step:
        def __init__(self, features, global_ok):
            self.features = features
            self.global_ok = global_ok

    class Example:
        def __init__(self, steps):
            self.steps = steps

    feats_a, feats_b = object(), object()
    corpus = [Example([Step(feats_a, False), Step(feats_b, False)])]
    scorer = SyntheticAUROCScorer(corpus, 0.75, seed=0)
    assert scorer.score(feats_a) != scorer.score(feats_b)


def test_unknown_feature_object_scores_zero():
    """A step outside the corpus must not raise mid-run."""
    scorer = SyntheticAUROCScorer([], 0.75, seed=0)
    assert scorer.score(object()) == 0.0


# ---- the measured result --------------------------------------------------


@needs_run
def test_crossing_is_below_the_probes_own_auroc():
    """C10. The draft guessed "well above 0.70"; the measurement says ~0.65.

    The direction is the claim: the score quality the verifier needs is already
    reached by the probe in section 7.6. If this ever flips back above 0.6968
    the chapter's conclusion changes.
    """
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    assert blob["crossing_measured_fa"] == pytest.approx(0.657, abs=0.02)
    assert blob["crossing_ci_last_negative"] == pytest.approx(0.625, abs=0.001)
    assert blob["crossing_ci_first_positive"] == pytest.approx(0.700, abs=0.001)
    assert blob["crossing_measured_fa"] < 0.6968


@needs_run
def test_the_threshold_is_made_of_false_alarms():
    """C10b. With no false alarms there is no crossing to find.

    Every point in sweep B must beat the no-gate baseline, including the 0.55
    one -- a score barely better than a coin.
    """
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    base = blob["baseline_projected_accuracy"]
    assert blob["crossing_no_fa"] is None
    worst = min(p["projected_accuracy_lo"] for p in blob["sweep_b"])
    assert worst > base


@needs_run
def test_no_score_quality_holds_a_binding_alpha():
    """C10c. The budget binds across the whole range, not just at the oracle."""
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    alpha = blob["binding_alpha"]
    best = min(p["selective_risk"] for p in blob["sweep_c"])
    assert best > alpha
    assert best / alpha > 1.5


@needs_run
def test_probe_underperforms_its_own_auroc_at_matched_budget():
    """C10d. AUROC is not a sufficient description of a score.

    Sweep D pins the verification rate, so the only thing differing between the
    probe and the synthetic score is which steps they rank highly.
    """
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    probe_auroc, probe_acc = blob["reference"]["probe, layer 25 (sec. 7.6)"]
    near = min(blob["sweep_d"],
               key=lambda p: abs(p["auroc_measured"] - probe_auroc))
    assert probe_acc < near["projected_accuracy_lo"]
    # and it is not that the probe caught fewer errors -- it caught more
    assert near["first_bad_recall"] > 0.3077


@needs_run
def test_first_bad_recall_rises_monotonically_with_score_quality():
    """The trend the position-bias claim actually rests on.

    Any single pair of rows differs by two or three trajectories out of ~39, so
    the monotone climb across the grid is the evidence, not one comparison.
    """
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    vals = [p["first_bad_recall"] for p in blob["sweep_a"]]
    assert vals == sorted(vals)
    assert vals[-1] - vals[0] > 0.4
