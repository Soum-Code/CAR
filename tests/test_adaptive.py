"""Tests for adaptive calibration under censored feedback.

`test_naive_update_is_biased_and_ipw_is_not` is the central experiment of the
project reduced to a unit test on a case where the right answer is known in
closed form. If this stops passing, the research claim is broken.

The simulation
--------------
Scores are uniform on [0, 1] and P(wrong | score) = score. The gate accepts
when score <= q, so the accept-region risk is

    E[wrong | score <= q] = (1/q) * integral_0^q s ds = q / 2

Setting that equal to alpha gives the ideal threshold q* = 2 * alpha.
For alpha = 0.1, q* = 0.2.

Meanwhile the REJECT region has risk E[wrong | score > q] = (1 + q) / 2, which
at q* is 0.6 -- six times the target. A naive update that consumes whatever
labels arrive is therefore fed a systematically inflated error rate and drives
the threshold down without bound. That is the failure, and it is not subtle.
"""

import numpy as np
import pytest

from car.conformal import AdaptiveCalibrator

ALPHA = 0.1
Q_STAR = 2 * ALPHA  # 0.2, derived above


def _run_stream(update_mode: str, *, epsilon=0.2, gamma=0.005, T=40000, seed=0):
    """Drive a calibrator over a synthetic stream with known accept-region risk.

    Returns the calibrator and the full threshold trajectory. The trajectory
    matters: with a constant step size these updates do not converge to a
    point, they converge to a stationary DISTRIBUTION. The tail mean is the
    right estimator; the final sample is one noisy draw from that distribution.
    """
    rng = np.random.default_rng(seed)
    cal = AdaptiveCalibrator(
        alpha=ALPHA,
        gamma=gamma,
        epsilon=epsilon,
        update_mode=update_mode,
        clip=(0.01, 1.0),
        seed=seed,
    )
    # Start deliberately too permissive so convergence is visible.
    cal.fit(np.array([0.9]), None)
    cal._threshold = 0.9

    trajectory = np.empty(T)
    for t in range(T):
        s = float(rng.random())
        wrong = int(rng.random() < s)
        explore = cal.should_explore()
        accepted = s <= cal.threshold

        # A label exists only if the step was verified: gate fired, or the
        # exploration coin came up.
        verified = (not accepted) or explore
        if verified:
            cal.update(wrong, was_exploration=explore, was_accepted=accepted)
        trajectory[t] = cal.threshold

    return cal, trajectory


def _tail_mean(trajectory: np.ndarray, frac: float = 0.25) -> float:
    return float(trajectory[int((1 - frac) * len(trajectory)) :].mean())


def test_naive_update_collapses_under_censored_feedback():
    """The motivating failure, and it is not marginal.

    Naive ACI consumes whatever labels arrive. Under this gate almost all of
    them come from the REJECT region, whose error rate is (1+q)/2 -- about 0.6
    near q*, against a target of 0.1. The update therefore sees a permanent
    error surplus and drives the threshold to the floor, verifying everything.

    Note this is not slow drift that more data would fix: it pins to the clip
    bound at every horizon tested.
    """
    naive, traj = _run_stream("naive")
    assert _tail_mean(traj) < 0.5 * Q_STAR, (
        f"naive should collapse toward always-verify, got {_tail_mean(traj):.4f}"
    )
    # Pinned to the clip floor rather than wandering near it.
    assert naive.threshold < 0.05


def test_exploration_only_recovers_the_analytic_optimum():
    """Unbiased: uses only accept-region labels, so it targets the right risk."""
    _, traj = _run_stream("exploration_only")
    est = _tail_mean(traj)
    assert abs(est - Q_STAR) < 0.05, f"expected ~{Q_STAR}, got {est:.4f}"


def test_ipw_recovers_the_analytic_optimum():
    """Same estimand as exploration-only, reached by a different route."""
    _, traj = _run_stream("ipw")
    est = _tail_mean(traj)
    assert abs(est - Q_STAR) < 0.05, f"expected ~{Q_STAR}, got {est:.4f}"


def test_ipw_converges_faster_than_exploration_only():
    """The point of the propensity weighting.

    Both use the same labels. IPW scales each update by 1/epsilon, giving
    1/epsilon times the expected drift, so on a SHORT horizon it should be
    substantially further along. This is the practical argument for it: budget
    is limited, so convergence speed per verification call is what matters.
    """
    _, ipw_traj = _run_stream("ipw", T=5000)
    _, expl_traj = _run_stream("exploration_only", T=5000)

    ipw_gap = abs(_tail_mean(ipw_traj) - Q_STAR)
    expl_gap = abs(_tail_mean(expl_traj) - Q_STAR)
    assert ipw_gap < expl_gap, (
        f"ipw should lead early: ipw gap {ipw_gap:.4f} vs expl gap {expl_gap:.4f}"
    )


def test_ipw_pays_for_speed_with_variance():
    """The cost side of the same trade-off.

    Amplifying each update by 1/epsilon amplifies the noise too. Documented as
    a test because a run that reports only the final threshold will look
    erratic for this reason, and that is a property of the estimator rather
    than a bug to chase.
    """
    _, ipw_traj = _run_stream("ipw")
    _, expl_traj = _run_stream("exploration_only")

    tail = slice(int(0.75 * len(ipw_traj)), None)
    assert ipw_traj[tail].std() > expl_traj[tail].std()


def test_zero_exploration_leaves_accept_region_unobserved():
    """With epsilon = 0 there are no accept-region labels at all.

    IPW and exploration-only must then make no updates -- the honest outcome
    when the feedback channel is fully closed.
    """
    cal, _ = _run_stream("exploration_only", epsilon=0.0, T=2000)
    assert cal.n_updates == 0
    assert cal.n_exploration_calls == 0


def test_ipw_requires_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        AdaptiveCalibrator(update_mode="ipw", epsilon=0.0)


def test_update_direction_tightens_on_error():
    """Observing an error must lower the threshold, never raise it.

    This is the safety property of the whole gate. A sign error here would make
    the system verify LESS as it discovers more mistakes.
    """
    cal = AdaptiveCalibrator(alpha=0.1, gamma=0.1, epsilon=1.0, update_mode="ipw")
    cal.fit(np.array([0.0, 1.0]), None)
    before = cal.threshold

    cal.update(1, was_exploration=True, was_accepted=True)  # observed an error
    assert cal.threshold < before

    after_error = cal.threshold
    cal.update(0, was_exploration=True, was_accepted=True)  # observed success
    assert cal.threshold > after_error


def test_exploration_rate_matches_epsilon():
    """Propensity must be what we claim it is -- IPW divides by this number."""
    cal = AdaptiveCalibrator(alpha=0.1, epsilon=0.25, update_mode="ipw", seed=7)
    cal.fit(np.array([0.0, 1.0]), None)
    draws = [cal.should_explore() for _ in range(20000)]
    assert abs(np.mean(draws) - 0.25) < 0.02


def test_threshold_respects_clip():
    cal = AdaptiveCalibrator(
        alpha=0.1, gamma=1.0, epsilon=1.0, update_mode="naive", clip=(0.0, 1.0)
    )
    cal.fit(np.array([0.0, 1.0]), None)
    for _ in range(100):
        cal.update(1, was_exploration=True, was_accepted=True)
    assert 0.0 <= cal.threshold <= 1.0


def test_rejects_unknown_update_mode():
    with pytest.raises(ValueError, match="update_mode"):
        AdaptiveCalibrator(update_mode="magic")


def test_trace_records_provenance():
    """Analysis needs to know which updates came from exploration."""
    cal = AdaptiveCalibrator(alpha=0.1, epsilon=1.0, update_mode="ipw")
    cal.fit(np.array([0.0, 1.0]), None)
    cal.update(1, was_exploration=True, was_accepted=True)
    cal.update(0, was_exploration=True, was_accepted=True)

    arrays = cal.trace.as_arrays()
    assert arrays["explored"].all()
    assert len(arrays["error"]) == 2
