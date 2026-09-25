"""Sampling uncertainty on the AUROCs this thesis argues from.

Chapters 5-7 turn on differences between AUROCs measured on one corpus of 500
solutions: 0.5589 against chance, 0.5740 against 0.4904, 0.6968 against 0.5742,
and -- the claim that needs an interval most -- "combining the two signals buys
0.0002", which is an assertion that a difference is *absent*. None of those
numbers carried an error bar, so none of the comparisons could be read as
anything stronger than an ordering of point estimates. This module supplies the
intervals.

WHY THE SOLUTION IS THE RESAMPLING UNIT
---------------------------------------
Bootstrapping steps independently treats 2,573 steps as 2,573 draws when they
came from 500 solutions. The obvious justification is the one this thesis is
about -- chapter 4 measures that 49.6% of wrong steps in wrong-answer solutions
are locally valid and wrong only because a premise was, so one corrupted premise
turns every step below it wrong at once and step labels within a solution are
strongly dependent.

That justification is not the right one, and the test suite says so. Clustered
*labels* alone leave an AUROC's variance essentially untouched (measured deff
0.94): AUROC is a two-sample rank statistic, so it barely notices that the class
balance of a resample moves around. What does inflate it is a per-solution shift
in the *score* -- one verbose question whose every step draws a high divergence,
one short question whose every step draws a low one (measured deff 1.45). The two
together multiply, to deff 4.3 on synthetic data built to hold both.

Both are present on this corpus, which is why the unit of resampling is the
solution -- draw 500 solutions with replacement, take all of their steps,
recompute -- and why `design_effect` *measures* the inflation for each score
instead of asserting it. `groups` is the solution id per step.

WHAT THE INTERVAL COVERS, AND WHAT IT DOES NOT
----------------------------------------------
These intervals cover the sampling variability of the *evaluation corpus* with
the score held fixed. They do not cover variability in fitting the score -- the
dev-split scaler, the probe's layer and C, the choice of which signal to report.
That component is real and this project has already measured it the honest way:
chapter 7 reports the probe at 0.8748 on the split used to select the layer and
0.6968 on test, so the selection cost is 0.18 AUROC and is quoted separately.
Reading these intervals as covering it too would double-count nothing and hide
the larger of the two effects, so: corpus variability only, stated as such.

Percentile intervals throughout. Not because percentile is the best bootstrap
interval -- BCa is better behaved for a skewed statistic -- but because with
AUROCs near 0.5 and n_pos in the hundreds the skew is mild, and a percentile
interval has no tuning to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap interval around it."""

    point: float
    lo: float
    hi: float
    se: float
    n_boot: int
    n_units: int
    #: Replicates discarded because the resample held only one class. Reported
    #: rather than silently dropped: if this is large the interval is not a
    #: percentile interval on what you think it is.
    n_degenerate: int = 0
    #: The replicates themselves, so a p-value comes off the same draw the
    #: interval did rather than a second one that would not quite agree with it.
    reps: tuple[float, ...] = field(default=(), repr=False)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def excludes(self, value: float) -> bool:
        """Whether `value` lies outside the interval, e.g. chance for an AUROC."""
        return not (self.lo <= value <= self.hi)

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


@dataclass
class _Resampler:
    """Solution-clustered bootstrap indices, drawn once and reused.

    Drawn once so that every statistic in a comparison sees the *same* 4,000
    resamples. Two AUROCs bootstrapped under independent draws give a difference
    whose interval is far too wide -- it adds two variances where the paired
    difference cancels most of one, and the cancellation is large here because
    all of these scores are evaluated on the same steps.
    """

    groups: np.ndarray
    n_boot: int
    seed: int
    n_units: int = 0
    _index: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self):
        uniq, inv = np.unique(self.groups, return_inverse=True)
        per_unit = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        rng = np.random.default_rng(self.seed)
        draws = rng.integers(0, len(uniq), size=(self.n_boot, len(uniq)))
        self.n_units = len(uniq)
        self._index = [np.concatenate([per_unit[u] for u in row]) for row in draws]

    def __iter__(self):
        return iter(self._index)


_CACHE: dict[tuple, _Resampler] = {}


def _resampler(groups: np.ndarray, n_boot: int, seed: int) -> _Resampler:
    """Memoised so one script's whole table shares a single set of resamples.

    Two purposes. It is faster -- building 4,000 index arrays over 500 solutions
    costs a few seconds and a results table asks for it a dozen times. More
    importantly it makes the intervals in that table mutually consistent: every
    score is measured on the same 4,000 corpora, so a reader comparing two rows
    is not also comparing two Monte Carlo draws.
    """
    key = (groups.tobytes(), groups.shape, n_boot, seed)
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = _Resampler(groups, n_boot, seed)
    return hit


def auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for ranking WRONG steps above correct ones.

    `correct` is True for a correct step, matching `metrics.step_detection_auroc`,
    whose value this reproduces exactly. Computed from midranks rather than
    through sklearn because the bootstrap calls it tens of thousands of times,
    and because midranks make the tie handling explicit: semantic divergence over
    K=5 samples takes only seven distinct values, so a large fraction of pairs
    are ties and each is worth half a point. A tie-blind AUROC would flatter it.
    """
    s = np.asarray(scores, dtype=float)
    pos = ~np.asarray(correct, dtype=bool)
    n_p = int(pos.sum())
    n_n = int(pos.size - n_p)
    if n_p == 0 or n_n == 0:
        return float("nan")
    r = rankdata(s)  # average ranks: a tied pair contributes exactly 0.5
    return float((r[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _percentile(reps: list[float], point: float, alpha: float, n_units: int) -> Interval:
    vals = np.asarray([v for v in reps if np.isfinite(v)], dtype=float)
    n_bad = len(reps) - vals.size
    if vals.size == 0:
        return Interval(point, float("nan"), float("nan"), float("nan"),
                        len(reps), n_units, n_bad)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(lo), float(hi), float(vals.std(ddof=1)),
                    len(reps), n_units, n_bad, tuple(float(v) for v in vals))


def auroc_ci(
    scores,
    correct,
    groups,
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
    unit: str = "solution",
) -> Interval:
    """Clustered bootstrap interval for one score's AUROC.

    `unit="step"` resamples steps independently instead, which is the interval
    the thesis would have reported had it reported one. It is here to be
    compared against, not to be used -- see `design_effect`.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=bool)
    g = np.asarray(groups)
    if unit == "step":
        g = np.arange(s.size)
    elif unit != "solution":
        raise ValueError("unit must be 'solution' or 'step'")

    res = _resampler(g, n_boot, seed)
    reps = [auroc(s[idx], y[idx]) for idx in res]
    return _percentile(reps, auroc(s, y), alpha, res.n_units)


def paired_auroc_delta_ci(
    baseline,
    contender,
    correct,
    groups,
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
    unit: str = "solution",
) -> Interval:
    """Interval for AUROC(contender) - AUROC(baseline) on the same steps.

    Paired: each replicate recomputes both AUROCs on one resample, so the shared
    corpus variability cancels. This is the right interval for every comparison
    the thesis makes between two scores, because all of them are evaluated on one
    fixed set of steps -- including the negative ones, where the claim is that a
    difference is too small to matter and the interval is what makes that a
    measurement rather than an impression.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(contender, dtype=float)
    y = np.asarray(correct, dtype=bool)
    g = np.arange(a.size) if unit == "step" else np.asarray(groups)
    if unit not in ("solution", "step"):
        raise ValueError("unit must be 'solution' or 'step'")

    res = _resampler(g, n_boot, seed)
    reps = [auroc(b[idx], y[idx]) - auroc(a[idx], y[idx]) for idx in res]
    return _percentile(reps, auroc(b, y) - auroc(a, y), alpha, res.n_units)


def design_effect(
    scores, correct, groups, *, n_boot: int = 4000, seed: int = 0
) -> dict[str, float]:
    """How much a step-level bootstrap understates the variance.

    `deff` is the variance ratio: clustered variance over independent-step
    variance. `deff` > 1 means steps within a solution carry less information
    than their count suggests, and `sqrt(deff)` is the factor by which every
    naive standard error on this corpus is too small.

    Worth reading per score rather than once for the corpus. The inflation comes
    from the score sharing a per-solution component, not from the labels being
    clustered (see the module docstring), so two scores on the same steps can
    have quite different design effects -- and a score with deff near 1 is one
    whose values do not travel with the solution.
    """
    clustered = auroc_ci(scores, correct, groups, n_boot=n_boot, seed=seed)
    iid = auroc_ci(scores, correct, groups, n_boot=n_boot, seed=seed, unit="step")
    deff = ((clustered.se / iid.se) ** 2
            if np.isfinite(iid.se) and iid.se > 0 else float("nan"))
    return {
        "se_clustered": clustered.se,
        "se_iid_steps": iid.se,
        "deff": deff,
        "se_inflation": float(np.sqrt(deff)) if np.isfinite(deff) else float("nan"),
        "n_solutions": clustered.n_units,
        "n_steps": iid.n_units,
    }


def bootstrap_p_value(interval: Interval, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for a statistic against `null`.

    The fraction of replicates on the far side of `null`, doubled, floored at
    1/n_boot because a bootstrap cannot resolve a p-value below its own
    resolution and reporting p = 0 would claim it can.
    """
    vals = np.asarray(interval.reps, dtype=float)
    if vals.size == 0:
        return float("nan")
    frac = min((vals <= null).mean(), (vals >= null).mean())
    # Capped at 1: a statistic whose every replicate sits exactly on the null
    # (a score compared against itself) makes both tails 1, and 2 is not a
    # p-value. Floored at 1/n_boot for the opposite reason.
    return float(min(1.0, max(2 * frac, 1.0 / vals.size)))


def proportion_ci(
    successes,
    groups,
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
    unit: str = "solution",
) -> Interval:
    """Clustered interval for a rate measured over steps nested in solutions.

    Unlike an AUROC, a proportion feels clustered labels directly, and this
    project's rates are the worst case for it. C2 measures that corruption is
    near-absorbing: once a premise is wrong, every step below it is globally
    wrong AND locally valid, which is precisely the event C1 counts. The
    intra-cluster correlation is therefore close to 1 by the thesis's own
    finding, and the design effect for a proportion runs at roughly
    1 + (m - 1) * rho for mean cluster size m -- so a Wilson or normal interval,
    which assumes m = 1, is too narrow by a factor that grows with how many
    steps each solution contributes.

    `successes` is one boolean per step (the numerator's event) and `groups` the
    solution id. Steps outside the denominator are simply not passed in.
    """
    y = np.asarray(successes, dtype=bool)
    g = np.arange(y.size) if unit == "step" else np.asarray(groups)
    if unit not in ("solution", "step"):
        raise ValueError("unit must be 'solution' or 'step'")
    if y.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"),
                        0, 0, 0)

    res = _resampler(g, n_boot, seed)
    reps = [float(y[idx].mean()) for idx in res]
    return _percentile(reps, float(y.mean()), alpha, res.n_units)


def proportion_delta_ci(
    successes_a,
    groups_a,
    successes_b,
    groups_b,
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Interval for rate(b) - rate(a) across two INDEPENDENT corpora.

    Unpaired, unlike `paired_auroc_delta_ci`, because the two corpora are
    different generators on different solutions and share no steps. Each side is
    resampled within itself -- its own solutions, with replacement -- and the
    difference taken per replicate.

    This is the interval to read when asking whether a rate transfers. Comparing
    two separately-quoted intervals and checking whether they overlap is a
    weaker and differently-shaped test: non-overlapping intervals imply a
    significant difference, but overlapping ones do not imply the absence of
    one.
    """
    ya, yb = np.asarray(successes_a, dtype=bool), np.asarray(successes_b, dtype=bool)
    ra = _resampler(np.asarray(groups_a), n_boot, seed)
    # A different seed for the second corpus: reusing one would correlate two
    # samples that are independent, and narrow the difference for no reason.
    rb = _resampler(np.asarray(groups_b), n_boot, seed + 1_000_003)
    reps = [float(yb[jb].mean()) - float(ya[ja].mean())
            for ja, jb in zip(ra, rb, strict=True)]
    return _percentile(reps, float(yb.mean()) - float(ya.mean()), alpha,
                       min(ra.n_units, rb.n_units))


def intra_cluster_correlation(successes, groups) -> float:
    """ANOVA estimate of rho for a binary outcome over unequal clusters.

    Reported alongside a design effect because the two say different things: the
    design effect is what the interval costs, rho is the property of the data
    that causes it. A rho near 1 on C1 is the numerical form of "corruption is
    near-absorbing" -- knowing one bad step's local validity tells you the rest
    of that solution's.
    """
    y = np.asarray(successes, dtype=float)
    g = np.asarray(groups)
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1 if inv.size else 0
    n = y.size
    if k < 2 or n <= k:
        return float("nan")

    sizes = np.bincount(inv, minlength=k).astype(float)
    means = np.bincount(inv, weights=y, minlength=k) / sizes
    grand = y.mean()

    msb = float((sizes * (means - grand) ** 2).sum()) / (k - 1)
    msw = float(((y - means[inv]) ** 2).sum()) / (n - k)
    # The unequal-cluster correction; with equal sizes it reduces to m.
    m0 = (n - (sizes**2).sum() / n) / (k - 1)
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return float("nan")
    return float(np.clip((msb - msw) / denom, -1.0, 1.0))


def proportion_design_effect(
    successes, groups, *, n_boot: int = 4000, seed: int = 0
) -> dict[str, float]:
    """What clustering costs a rate, measured and predicted side by side.

    `deff` is the bootstrap answer: clustered variance over independent-step
    variance. `deff_predicted` is 1 + (m - 1) * rho, the textbook formula, which
    is here as a check -- when the two disagree badly the cluster sizes are too
    skewed for the formula and the bootstrap is the one to believe.
    """
    clustered = proportion_ci(successes, groups, n_boot=n_boot, seed=seed)
    iid = proportion_ci(successes, groups, n_boot=n_boot, seed=seed, unit="step")
    rho = intra_cluster_correlation(successes, groups)
    m = iid.n_units / max(1, clustered.n_units)
    deff = ((clustered.se / iid.se) ** 2
            if np.isfinite(iid.se) and iid.se > 0 else float("nan"))
    return {
        "rate": clustered.point,
        "se_clustered": clustered.se,
        "se_iid_steps": iid.se,
        "deff": deff,
        "se_inflation": float(np.sqrt(deff)) if np.isfinite(deff) else float("nan"),
        "rho": rho,
        "mean_cluster_size": m,
        "deff_predicted": 1 + (m - 1) * rho if np.isfinite(rho) else float("nan"),
        "n_solutions": clustered.n_units,
        "n_steps": iid.n_units,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval, kept so the clustered one can be compared against it.

    Correct for n INDEPENDENT Bernoulli draws and wrong for steps nested in
    solutions, which is every rate in this project. It stays in the codebase
    because "the old interval was this much too narrow" is a number worth
    printing, not because anything should quote it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def minimum_detectable_delta(
    se: float, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """The smallest true difference a corpus with this paired SE could detect.

    The number that says what an unrun experiment is worth. If two scores differ
    on this corpus by less than this, the comparison cannot resolve it and a
    point estimate either way is noise -- so an experiment whose plausible effect
    is smaller than this is not worth its compute, and one whose effect would
    have to be larger than the whole result is asking for is worth less still.

    Normal approximation, which is what the paired bootstrap SE supports; two
    significant figures is the most that should be read off it.
    """
    from scipy.stats import norm

    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def auroc_resolution_ceiling(
    scores, correct, grid, *, seed: int = 0
) -> dict[str, float]:
    """What a continuous score loses when coarsened to another score's grid.

    Semantic divergence over K samples is not a continuous score: divergence
    depends only on the block sizes of the sample partition, so K=5 admits
    exactly seven values (the partitions of 5). Comparing its AUROC to a
    continuous probe's therefore confounds two things -- how much signal a score
    carries, and how finely it can express it.

    This separates them by quantising `scores` onto the empirical distribution of
    `grid` (same value counts, order preserved) and re-measuring. The drop is the
    price of the resolution alone, with the signal held fixed.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=bool)
    levels = np.sort(np.asarray(grid, dtype=float))
    if levels.size != s.size:
        raise ValueError("grid must have one value per step")

    # Rank-match: the i-th smallest score takes the i-th smallest grid value, so
    # the coarsened score has the grid's exact value distribution and keeps the
    # original's ordering. Ties in `scores` are broken by index, which is
    # arbitrary -- harmless for a continuous score, so the tie count is returned
    # and the caller can see when it is not.
    order = np.argsort(rankdata(s, method="ordinal"))
    coarse = np.empty_like(s)
    coarse[order] = levels

    full, dropped = auroc(s, y), auroc(coarse, y)
    return {
        "auroc_continuous": full,
        "auroc_on_grid": dropped,
        "resolution_cost": full - dropped,
        "n_levels": int(np.unique(levels).size),
        "n_tied_pairs_in_scores": int(s.size - np.unique(s).size),
    }
