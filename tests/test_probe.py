"""Tests for the internal-state probe.

What needs protecting here is not the fit -- sklearn does that -- but the
selection discipline. There are 29 candidate layers and five regularisation
strengths on ~600 training steps; choosing among them by test AUROC would
manufacture a result out of the search space, and it is the easiest possible
way to get a wrong answer from this experiment.
"""

import numpy as np
import pytest

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
