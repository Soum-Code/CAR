"""Tests for the calibration layer.

The important one is `test_coverage_guarantee_holds_empirically`. If split
conformal does not attain its target on clean exchangeable data, nothing built
on top of it can be trusted, and every later result would be measuring our bug
rather than the model.
"""

import numpy as np
import pytest

from car.conformal import (
    RiskControlCalibrator,
    SplitConformalCalibrator,
    conformal_quantile,
    crc_is_feasible,
    false_safe_loss,
    min_calibration_size,
)


def test_quantile_is_conservative_at_small_n():
    # With n small relative to alpha, the corrected level exceeds 1 and the
    # honest answer is the max score -- never a value that would silently
    # under-cover.
    scores = np.array([0.1, 0.2, 0.3])
    assert conformal_quantile(scores, alpha=0.01) == pytest.approx(0.3)


def test_min_calibration_size():
    assert min_calibration_size(0.1) == 9
    assert min_calibration_size(0.05) == 19


def test_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([1.0, 2.0]), alpha=0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([1.0, 2.0]), alpha=1.0)


def test_quantile_rejects_empty():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), alpha=0.1)


def test_coverage_guarantee_holds_empirically():
    """Marginal coverage should be at least 1-alpha, averaged over trials.

    This is the ground-truth check on our implementation of the guarantee.
    """
    rng = np.random.default_rng(0)
    alpha = 0.1
    n_cal, n_test, n_trials = 500, 500, 200

    coverages = []
    for _ in range(n_trials):
        cal = rng.normal(size=n_cal)
        test = rng.normal(size=n_test)
        q = conformal_quantile(cal, alpha)
        coverages.append(float((test <= q).mean()))

    mean_cov = float(np.mean(coverages))
    assert mean_cov >= 1 - alpha - 0.01, f"under-coverage: {mean_cov:.4f}"
    # Should not be wildly conservative either -- a threshold at +inf would
    # "pass" a one-sided check while being useless.
    assert mean_cov <= 1 - alpha + 0.05, f"over-coverage: {mean_cov:.4f}"


def test_split_calibrator_fits_on_correct_steps_only():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    labels = np.array([True] * 200 + [False] * 200)

    cal = SplitConformalCalibrator(alpha=0.1).fit(scores, labels)
    # Threshold is set by the correct-step distribution, so it must sit well
    # below the wrong-step cluster at ~5.
    assert cal.threshold < 3.0
    assert cal.n_calibration == 200


def test_split_calibrator_raises_before_fit():
    with pytest.raises(RuntimeError):
        _ = SplitConformalCalibrator().threshold


def test_crc_controls_false_safe_rate():
    """CRC threshold should keep the wrong-step acceptance rate under alpha."""
    rng = np.random.default_rng(2)
    n = 2000
    correct = rng.random(n) > 0.3
    # Separable-ish: wrong steps score higher.
    scores = np.where(correct, rng.normal(0, 1, n), rng.normal(2.5, 1, n))

    cal = RiskControlCalibrator(alpha=0.1).fit(scores, correct)
    accepted = scores <= cal.threshold
    realised = float((accepted & ~correct).mean())
    assert realised <= 0.1 + 0.02, f"risk not controlled: {realised:.4f}"


def test_crc_reports_infeasible_when_score_is_uninformative():
    """A score with no signal should be flagged, not silently accepted."""
    rng = np.random.default_rng(3)
    n = 400
    correct = rng.random(n) > 0.5  # 50% wrong, score carries nothing
    scores = rng.normal(size=n)

    lambdas = np.linspace(-5, 5, 200)
    loss = false_safe_loss(scores, correct)
    # Target below what any threshold can deliver on pure noise.
    assert not crc_is_feasible(loss, lambdas, alpha=0.001)


def test_crc_grid_must_be_sorted():
    scores = np.array([0.0, 1.0, 2.0])
    labels = np.array([True, True, False])
    from car.conformal import crc_lambda

    with pytest.raises(ValueError):
        crc_lambda(false_safe_loss(scores, labels), np.array([1.0, 0.0]), 0.1)
