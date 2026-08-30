"""Split conformal calibration.

The baseline calibrator: take scores on a held-out calibration set, take the
finite-sample-corrected empirical quantile, use it as the threshold.

What the guarantee says
-----------------------
Under exchangeability of calibration and test scores, the acceptance rule
{s <= q_hat} has marginal coverage at least 1 - alpha.

What it does NOT say
--------------------
* It is not model accuracy. 1 - alpha is a property of the acceptance rule.
* It is MARGINAL, i.e. averaged over the whole distribution. Coverage on any
  particular slice of questions can be far below 1 - alpha.
* Reasoning steps within a trajectory are NOT exchangeable -- step t is
  conditioned on step t-1. This calibrator is therefore the well-understood
  baseline, not the method CAR ultimately defends. See `adaptive.py`.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile of calibration scores.

    The ceil((n+1)(1-alpha))/n level -- rather than a plain (1-alpha) quantile
    -- is what makes the coverage guarantee hold at finite n instead of only
    asymptotically.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        raise ValueError("cannot calibrate on an empty score set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        # n too small to certify this alpha; the honest threshold is the max
        # observed score. Caller should treat this as "calibration set too
        # small" rather than a valid guarantee.
        return float(s.max())
    return float(np.quantile(s, level, method="higher"))


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which ceil((n+1)(1-alpha))/n <= 1 holds.

    Below this, split conformal cannot certify the requested alpha at all.
    Worth asserting in experiment setup -- it is a common silent failure.
    """
    return int(np.ceil(1.0 / alpha)) - 1


class SplitConformalCalibrator:
    """Static threshold fitted once on a calibration set."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._threshold: float | None = None
        self.n_calibration = 0

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrator used before fit()")
        return self._threshold

    def fit(self, scores: np.ndarray, labels: np.ndarray | None = None) -> "SplitConformalCalibrator":
        """Fit on calibration scores.

        `labels` is accepted for interface symmetry with the risk-control and
        adaptive calibrators, which need them, but split conformal on the raw
        score distribution does not.
        """
        s = np.asarray(scores, dtype=float)
        if labels is not None:
            # Calibrate on the scores of CORRECT steps: the acceptance region
            # should cover correct steps at rate 1-alpha.
            lab = np.asarray(labels, dtype=bool)
            if lab.shape != s.shape:
                raise ValueError("scores and labels must have the same shape")
            s = s[lab]
            if s.size == 0:
                raise ValueError("no correct steps in calibration set")
        self.n_calibration = int(s.size)
        self._threshold = conformal_quantile(s, self.alpha)
        return self

    def update(self, observed_error: int, was_exploration: bool = False) -> None:
        """No-op. Static calibration by definition does not adapt.

        Present so the agent loop can call `update()` unconditionally and the
        static-vs-adaptive ablation is a config change, not a code change.
        """
        return None
