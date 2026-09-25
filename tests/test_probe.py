"""Tests for the internal-state probe.

What needs protecting here is not the fit -- sklearn does that -- but the
selection discipline. There are 29 candidate layers and five regularisation
strengths on ~600 training steps; choosing among them by test AUROC would
manufacture a result out of the search space, and it is the easiest possible
way to get a wrong answer from this experiment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# exp_probe_variants lives in scripts/, which is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from car.uncertainty.probe import (
    DEFAULT_CS,
    auroc,
    fit_one,
    learning_curve,
    select_and_fit,
)


def synthetic(n=400, hidden=8, signal_layer=2, n_layers=4, strength=1.5, seed=0):
    """One layer carries a real signal; the others are pure noise.

    A correct selection procedure must find `signal_layer` without being told.
    """
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    by_layer = {}
    for L in range(n_layers):
        X = rng.normal(size=(n, hidden))
        if L == signal_layer:
            X[:, 0] += np.where(y, strength, 0.0)
        by_layer[L] = X
    return by_layer, y


def split_idx(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    return p[: n // 2], p[n // 2 : 3 * n // 4], p[3 * n // 4 :]


def test_finds_the_layer_that_carries_the_signal():
    by_layer, y = synthetic()
    tr, sel, _ = split_idx(len(y))
    r = select_and_fit(by_layer, y, tr, sel)
    assert r.layer == 2
    assert r.auroc_select > 0.7


def test_reports_every_layer_it_tried():
    """The per-layer table is what tells a reader the result is not one lucky
    layer out of 29."""
    by_layer, y = synthetic()
    tr, sel, _ = split_idx(len(y))
    r = select_and_fit(by_layer, y, tr, sel)
    assert set(r.per_layer) == set(by_layer)
    assert r.per_layer[r.layer] == pytest.approx(r.auroc_select)


def test_noise_only_states_land_near_chance():
    """The null. If this ever returns a high AUROC the harness is leaking."""
    by_layer, y = synthetic(signal_layer=-1)   # no layer carries signal
    tr, sel, test = split_idx(len(y))
    r = select_and_fit(by_layer, y, tr, sel)
    # Selection over 4 layers x 5 Cs inflates the SELECTION score; the held-out
    # score is the one that must stay near chance, and that gap is exactly why
    # the layer is never chosen on test.
    assert auroc(r.model, r.scaler, by_layer[r.layer][test], y[test]) < 0.65


def test_selection_cannot_see_the_test_set():
    """Structural, not behavioural: test indices are not a parameter."""
    import inspect

    params = set(inspect.signature(select_and_fit).parameters)
    assert "test_idx" not in params
    assert params == {"states_by_layer", "y", "train_idx", "select_idx", "Cs"}


def test_score_is_oriented_like_every_other_score():
    """Higher must mean LESS trustworthy, matching token entropy and semantic
    divergence, or the composite scorer would silently invert it."""
    by_layer, y = synthetic()
    tr, sel, test = split_idx(len(y))
    r = select_and_fit(by_layer, y, tr, sel)
    s = r.score(by_layer[r.layer][test])
    assert s[y[test]].mean() > s[~y[test]].mean()
    assert ((s >= 0) & (s <= 1)).all()


def test_learning_curve_grows_with_data_when_signal_exists():
    by_layer, y = synthetic(n=800, strength=2.0)
    tr, sel, _ = split_idx(len(y))
    curve = learning_curve(by_layer[2], y, tr, sel, C=1.0)
    assert len(curve) >= 3
    assert curve[-1][0] > curve[0][0]
    assert curve[-1][1] > curve[0][1] - 0.15   # non-decreasing, modulo noise


def test_degenerate_labels_do_not_crash():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = np.zeros(50, dtype=bool)
    m, s = fit_one(X[:30], np.array([True] * 15 + [False] * 15), 1.0)
    assert np.isnan(auroc(m, s, X[30:], y[30:]))


def test_default_regularisation_spans_orders_of_magnitude():
    """3,584 features on ~600 examples: the useful C is not near 1."""
    assert min(DEFAULT_CS) <= 1e-4
    assert max(DEFAULT_CS) >= 1.0


# ---- what a probe should be trained on -------------------------------------
# Ch. 7.4 measured that two scores of equal AUROC are worth different amounts,
# because the projection only pays for the FIRST globally-wrong step. These
# cover the pieces that encode that, since getting them subtly wrong would
# produce a plausible number rather than a failure.


def test_first_bad_recall_counts_only_the_first_bad_step():
    from exp_probe_variants import first_bad_recall

    # Four steps, two wrong; only the earlier one is "first bad".
    first_bad = np.array([False, True, False, False])
    # A score that ranks the LAST step highest catches nothing that matters.
    assert first_bad_recall(np.array([0.1, 0.2, 0.3, 0.9]), first_bad, 0.25) == 0.0
    # One that ranks the first-bad step highest catches all of it.
    assert first_bad_recall(np.array([0.1, 0.9, 0.3, 0.2]), first_bad, 0.25) == 1.0


def test_first_bad_recall_is_measured_at_a_fixed_budget():
    """A score must not be able to win by flagging more steps."""
    from exp_probe_variants import first_bad_recall

    first_bad = np.array([False, True, False, False, False, False, False, False])
    scores = np.array([0.0, 0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3])
    tight = first_bad_recall(scores, first_bad, 0.125)   # top 1 of 8
    loose = first_bad_recall(scores, first_bad, 0.75)    # top 6 of 8
    assert tight == 0.0 and loose == 1.0, "the budget must bind"


def test_residualising_removes_a_planted_position_signal():
    """Variant D's mechanism. If position survives this, D measures nothing."""
    from exp_probe_variants import residualise

    rng = np.random.default_rng(0)
    pos = rng.random(400)
    X = rng.normal(0, 1, (400, 5))
    X[:, 0] += pos * 10.0                      # dimension 0 is pure position
    before = abs(np.corrcoef(X[:, 0], pos)[0, 1])
    after = abs(np.corrcoef(residualise(X, pos)[:, 0], pos)[0, 1])
    assert before > 0.9
    assert after < 1e-6


def test_residualising_keeps_signal_that_is_not_positional():
    from exp_probe_variants import residualise

    rng = np.random.default_rng(0)
    pos = rng.random(400)
    y = rng.random(400) < 0.3
    X = rng.normal(0, 1, (400, 3))
    X[:, 1] += y * 4.0                          # dimension 1 is the real signal
    kept = residualise(X, pos)
    assert abs(np.corrcoef(kept[:, 1], y.astype(float))[0, 1]) > 0.7


def test_select_by_never_takes_test_indices():
    """Same discipline as select_and_fit, restated for the variant selector.

    29 layers x 5 C values chosen on a criterion with ~14 positives in the
    selection split is exactly where a winner's curse comes from; the guard is
    that test indices cannot be passed in even by accident.
    """
    import inspect

    from exp_probe_variants import select_by

    params = set(inspect.signature(select_by).parameters)
    assert not {"test_idx", "test", "test_index"} & params
