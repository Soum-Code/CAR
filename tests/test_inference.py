"""Tests for the bootstrap intervals the thesis's AUROC claims are read through.

The load-bearing ones are `test_auroc_matches_sklearn_including_ties` -- because
every interval is a percentile of this statistic and a tie-handling bug would
shift all of them the same way and stay invisible -- and
`test_clustered_variance_exceeds_iid_when_labels_follow_the_cluster`, which is
the assumption the whole module rests on, checked on data where the answer is
known rather than argued from the corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from car.eval.inference import (
    auroc,
    auroc_ci,
    auroc_resolution_ceiling,
    bootstrap_p_value,
    design_effect,
    intra_cluster_correlation,
    paired_auroc_delta_ci,
    proportion_ci,
    proportion_delta_ci,
    proportion_design_effect,
    wilson,
)
from car.eval.metrics import step_detection_auroc

BOOT = 400  # enough for the shape of the distribution; the scripts use 4,000


def _corpus(n_groups=60, per_group=5, seed=0):
    """Steps nested in solutions, with a score that ranks wrong steps high."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    correct = rng.random(groups.size) > 0.35
    scores = rng.normal(np.where(correct, 0.0, 0.8), 1.0)
    return scores, correct, groups


# -- the statistic itself ----------------------------------------------------

def test_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    for n_levels in (2, 7, 50, None):
        s = rng.normal(size=500)
        if n_levels:                      # force heavy ties, as divergence has
            s = np.round(s * n_levels) / n_levels
        y = rng.random(500) > 0.4
        assert auroc(s, y) == pytest.approx(step_detection_auroc(s, y), abs=1e-12)


def test_a_constant_score_is_exactly_chance():
    y = np.array([True, False, True, False])
    assert auroc(np.zeros(4), y) == pytest.approx(0.5)


def test_a_perfect_score_is_one_and_an_inverted_one_is_zero():
    y = np.array([True, True, False, False])
    assert auroc(np.array([0.0, 0.1, 0.9, 1.0]), y) == pytest.approx(1.0)
    assert auroc(np.array([1.0, 0.9, 0.1, 0.0]), y) == pytest.approx(0.0)


def test_auroc_is_nan_when_one_class_is_missing():
    assert np.isnan(auroc(np.arange(4.0), np.ones(4, dtype=bool)))


# -- the interval -----------------------------------------------------------

def test_interval_brackets_the_point_estimate():
    s, y, g = _corpus()
    ci = auroc_ci(s, y, g, n_boot=BOOT)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.n_units == 60
    assert ci.width > 0


def test_a_signal_this_strong_excludes_chance_and_noise_does_not():
    s, y, g = _corpus()
    assert auroc_ci(s, y, g, n_boot=BOOT).excludes(0.5)

    rng = np.random.default_rng(7)
    noise = rng.normal(size=y.size)
    assert not auroc_ci(noise, y, g, n_boot=BOOT).excludes(0.5)


def test_the_same_seed_gives_the_same_interval():
    s, y, g = _corpus()
    a = auroc_ci(s, y, g, n_boot=BOOT, seed=3)
    b = auroc_ci(s, y, g, n_boot=BOOT, seed=3)
    assert (a.lo, a.hi, a.se) == (b.lo, b.hi, b.se)


def test_a_different_seed_moves_the_interval_only_a_little():
    s, y, g = _corpus()
    a = auroc_ci(s, y, g, n_boot=BOOT, seed=1)
    b = auroc_ci(s, y, g, n_boot=BOOT, seed=2)
    assert abs(a.lo - b.lo) < 0.05 and abs(a.hi - b.hi) < 0.05


def test_a_degenerate_resample_is_counted_not_propagated():
    """One solution, all-correct: every resample has one class, nothing is silent."""
    s = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.ones(4, dtype=bool)
    ci = auroc_ci(s, y, np.zeros(4, dtype=int), n_boot=20)
    assert ci.n_degenerate == 20
    assert np.isnan(ci.lo) and np.isnan(ci.hi)


def test_an_unknown_unit_is_refused():
    s, y, g = _corpus()
    with pytest.raises(ValueError):
        auroc_ci(s, y, g, n_boot=10, unit="question")


# -- why the unit is the solution ------------------------------------------

def test_clustered_labels_alone_do_not_inflate_the_variance():
    """The plausible reason for clustering, and it is the wrong one.

    Every step in a solution shares one label -- the extreme of the premise
    propagation chapter 4 measures -- and the design effect still comes out at
    about 1. AUROC is a two-sample rank statistic: it barely notices that the
    class balance of a resample moves around. This is pinned because the module
    docstring makes the claim, and a reader who assumed the obvious argument
    would believe the opposite.
    """
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(60), 5)
    correct = (rng.random(60) > 0.4)[groups]
    scores = rng.normal(np.where(correct, 0.0, 0.7), 1.0)

    d = design_effect(scores, correct, groups, n_boot=BOOT)
    assert 0.8 < d["deff"] < 1.25
    assert (d["n_solutions"], d["n_steps"]) == (60, 300)


def test_a_per_solution_shift_in_the_score_is_what_inflates_it():
    """The mechanism that does matter: one solution, one offset, every step.

    A verbose question whose every step draws a high score and a short one whose
    every step draws a low one. Here the step-level bootstrap genuinely
    understates the spread, and with clustered labels on top the two multiply.
    """
    rng = np.random.default_rng(21)
    groups = np.repeat(np.arange(60), 5)
    correct = (rng.random(60) > 0.4)[groups]
    offset = rng.normal(0, 1.0, 60)[groups]
    scores = rng.normal(np.where(correct, 0.0, 0.7), 0.5) + offset

    d = design_effect(scores, correct, groups, n_boot=BOOT)
    assert d["deff"] > 2.0
    assert d["se_clustered"] > d["se_iid_steps"]
    assert d["se_inflation"] == pytest.approx(np.sqrt(d["deff"]))


def test_independent_steps_give_a_design_effect_near_one():
    """The converse: one step per solution, so the two bootstraps coincide."""
    rng = np.random.default_rng(12)
    y = rng.random(300) > 0.4
    s = rng.normal(np.where(y, 0.0, 0.7), 1.0)
    d = design_effect(s, y, np.arange(300), n_boot=BOOT)
    assert 0.6 < d["deff"] < 1.6


# -- paired differences ----------------------------------------------------

def test_a_score_against_itself_has_a_delta_of_exactly_zero():
    s, y, g = _corpus()
    d = paired_auroc_delta_ci(s, s, y, g, n_boot=BOOT)
    assert d.point == 0.0 and d.lo == 0.0 and d.hi == 0.0
    assert bootstrap_p_value(d) == pytest.approx(1.0)


def test_a_real_improvement_is_detected_and_signed():
    s, y, g = _corpus()
    rng = np.random.default_rng(5)
    better = np.where(y, -1.0, 1.0) + rng.normal(0, 0.3, size=y.size)
    d = paired_auroc_delta_ci(s, better, y, g, n_boot=BOOT)
    assert d.point > 0 and not d.excludes(d.point)
    assert d.lo > 0                      # the improvement is not a coin flip
    assert bootstrap_p_value(d) < 0.05


def test_pairing_is_tighter_than_two_independent_intervals():
    """Why the resamples are shared: two scores that differ by a little noise.

    Comparing their separate intervals would suggest no difference at all; the
    paired interval on the difference resolves it, because the corpus variability
    they share cancels.
    """
    s, y, g = _corpus()
    rng = np.random.default_rng(6)
    nudged = s + 0.12 * np.where(y, -1.0, 1.0) + rng.normal(0, 0.02, size=y.size)

    paired = paired_auroc_delta_ci(s, nudged, y, g, n_boot=BOOT)
    a, b = (auroc_ci(x, y, g, n_boot=BOOT) for x in (s, nudged))
    unpaired_se = float(np.hypot(a.se, b.se))
    assert paired.se < unpaired_se / 2


def test_delta_is_antisymmetric():
    s, y, g = _corpus()
    rng = np.random.default_rng(8)
    other = s + rng.normal(0, 0.5, size=y.size)
    fwd = paired_auroc_delta_ci(s, other, y, g, n_boot=BOOT, seed=4)
    rev = paired_auroc_delta_ci(other, s, y, g, n_boot=BOOT, seed=4)
    assert fwd.point == pytest.approx(-rev.point)
    assert fwd.lo == pytest.approx(-rev.hi)


def test_p_value_cannot_beat_the_bootstrap_resolution():
    s, y, g = _corpus()
    perfect = np.where(y, 0.0, 1.0)
    d = paired_auroc_delta_ci(s, perfect, y, g, n_boot=BOOT)
    assert bootstrap_p_value(d) == pytest.approx(1.0 / BOOT)


# -- resolution ------------------------------------------------------------

def test_coarsening_to_a_seven_level_grid_costs_auroc():
    """A continuous score forced onto divergence's seven levels loses ranking."""
    rng = np.random.default_rng(13)
    y = rng.random(800) > 0.4
    s = rng.normal(np.where(y, 0.0, 1.0), 1.0)
    grid = rng.choice([0.0, 0.43, 0.56, 0.68, 0.86, 0.93, 1.0], size=800)

    r = auroc_resolution_ceiling(s, y, grid)
    assert r["n_levels"] == 7
    assert r["auroc_on_grid"] < r["auroc_continuous"]
    assert r["resolution_cost"] > 0


def test_a_grid_as_fine_as_the_score_costs_nothing():
    rng = np.random.default_rng(14)
    y = rng.random(400) > 0.4
    s = rng.normal(np.where(y, 0.0, 1.0), 1.0)
    r = auroc_resolution_ceiling(s, y, np.linspace(0, 1, 400))
    assert r["resolution_cost"] == pytest.approx(0.0, abs=1e-12)


def test_the_grid_must_cover_every_step():
    y = np.array([True, False, True])
    with pytest.raises(ValueError):
        auroc_resolution_ceiling(np.arange(3.0), y, np.array([0.0, 1.0]))


# -- rates, where clustering acts through a different mechanism -------------

def test_a_rate_interval_brackets_the_rate_and_shrinks_with_n():
    rng = np.random.default_rng(31)
    wide = proportion_ci(rng.random(80) < 0.6, np.arange(80), n_boot=BOOT)
    narrow = proportion_ci(rng.random(2000) < 0.6, np.arange(2000), n_boot=BOOT)
    for ci in (wide, narrow):
        assert ci.lo <= ci.point <= ci.hi
    assert narrow.width < wide.width


def test_an_empty_denominator_is_nan_not_a_crash():
    ci = proportion_ci([], [], n_boot=BOOT)
    assert np.isnan(ci.point) and ci.n_boot == 0


def test_clustering_costs_a_rate_nothing_when_the_clusters_are_singletons():
    """The C1 case. Dependence can be near-total and still not widen anything.

    Every solution contributes ONE step to the denominator, so there is nothing
    for the within-solution correlation to act on. This is the shape the real
    C1 denominator has (mean 1.98 steps per wrong-answer solution) and the
    reason its published Wilson interval turned out to be very nearly right.
    """
    rng = np.random.default_rng(32)
    y = rng.random(300) < 0.9
    d = proportion_design_effect(y, np.arange(300), n_boot=BOOT)
    assert 0.7 < d["deff"] < 1.4
    assert d["mean_cluster_size"] == pytest.approx(1.0)


def test_clustering_costs_a_rate_a_great_deal_when_the_clusters_are_large():
    """Same dependence, bigger clusters: now the interval has to widen.

    Eight steps per solution all sharing one outcome -- the absorbing case. The
    denominator says 480 and the information says 60.
    """
    rng = np.random.default_rng(33)
    groups = np.repeat(np.arange(60), 8)
    y = (rng.random(60) < 0.9)[groups]
    d = proportion_design_effect(y, groups, n_boot=BOOT)
    assert d["deff"] > 4.0
    assert d["rho"] > 0.9
    assert d["mean_cluster_size"] == pytest.approx(8.0)


def test_the_predicted_design_effect_tracks_the_bootstrap_one():
    """1 + (m-1)*rho against the resampled answer, on equal-sized clusters.

    They agree where the formula's assumptions hold, which is what makes a
    DISagreement on real data informative rather than alarming.
    """
    rng = np.random.default_rng(34)
    groups = np.repeat(np.arange(120), 4)
    base = (rng.random(120) < 0.7)[groups]
    flip = rng.random(base.size) < 0.15
    y = np.where(flip, ~base, base)

    d = proportion_design_effect(y, groups, n_boot=2000)
    assert d["deff"] == pytest.approx(d["deff_predicted"], rel=0.35)


def test_rho_is_one_for_perfectly_absorbing_clusters_and_zero_for_none():
    groups = np.repeat(np.arange(50), 4)
    rng = np.random.default_rng(35)
    absorbing = (rng.random(50) < 0.5)[groups]
    assert intra_cluster_correlation(absorbing, groups) > 0.95

    independent = rng.random(200) < 0.5
    assert abs(intra_cluster_correlation(independent, groups)) < 0.25


def test_rho_needs_at_least_two_clusters():
    assert np.isnan(intra_cluster_correlation([True, False], [0, 0]))


def test_two_corpora_are_resampled_separately():
    """An unpaired difference: no step is shared, so nothing may cancel."""
    rng = np.random.default_rng(36)
    a, b = rng.random(400) < 0.50, rng.random(400) < 0.80
    d = proportion_delta_ci(a, np.arange(400), b, np.arange(400), n_boot=BOOT)
    assert d.point == pytest.approx(b.mean() - a.mean())
    assert d.lo > 0
    assert bootstrap_p_value(d) < 0.05


def test_two_corpora_with_the_same_rate_give_a_difference_through_zero():
    rng = np.random.default_rng(37)
    a, b = rng.random(400) < 0.6, rng.random(400) < 0.6
    d = proportion_delta_ci(a, np.arange(400), b, np.arange(400), n_boot=BOOT)
    assert not d.excludes(0.0)


def test_wilson_matches_the_published_c1_interval():
    """The interval this project published for Qwen C1: 113 of 125.

    Pinned because the clustered interval is quoted against it, and a number
    that moves would make that comparison meaningless.
    """
    lo, hi = wilson(113, 125)
    assert (round(lo, 3), round(hi, 3)) == (0.840, 0.944)


def test_wilson_is_defined_at_the_edges():
    """An empty denominator, and a rate of exactly 1 -- where a normal
    approximation would return a zero-width interval and Wilson does not."""
    assert all(np.isnan(v) for v in wilson(0, 0))

    lo, hi = wilson(10, 10)
    assert 0.0 <= lo < 1.0
    assert hi == pytest.approx(1.0)
    assert lo > 0.6                      # not a degenerate [0, 1]
