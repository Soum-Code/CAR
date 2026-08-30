"""Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024).

Why this and not plain split conformal
--------------------------------------
Split conformal controls COVERAGE -- the probability that the true label lands
in the accepted set. What CAR actually wants to control is a RISK: the rate at
which wrong steps get accepted by the gate. Those are different quantities, and
conformal risk control is the tool built for the second one.

CRC controls E[L(lambda_hat)] <= alpha for any loss L that is monotone
non-increasing in lambda and bounded above by B. Split conformal is the special
case where L is the 0/1 miscoverage indicator.

Parametrisation note
--------------------
CAR's gate accepts a step when score <= threshold, so the false-safe loss
INCREASES with the threshold. CRC requires the loss to DECREASE with lambda.
We therefore sweep lambda over negated thresholds; `false_safe_loss` below
handles the flip so callers never have to think about it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Maps a lambda value to a per-example loss vector.
LossFn = Callable[[float], np.ndarray]


def crc_lambda(
    loss_fn: LossFn,
    lambdas: np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> float:
    """Smallest lambda whose corrected empirical risk is within alpha.

    Implements lambda_hat = inf{ lambda : (n/(n+1)) R_n(lambda) + B/(n+1) <= alpha }.

    The B/(n+1) term is the finite-sample correction; it is what makes the
    guarantee hold for a fresh test point rather than only in expectation over
    an infinite calibration set. On small calibration sets it dominates, which
    is the honest signal that you cannot certify that alpha with that much data.

    Parameters
    ----------
    lambdas:
        Increasing grid. `loss_fn` must be non-increasing along it.

    Returns
    -------
    The selected lambda. If no grid point satisfies the constraint, returns the
    largest lambda (most conservative available) -- callers should check
    `crc_is_feasible` if they need to distinguish that case.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.size == 0:
        raise ValueError("empty lambda grid")
    if not np.all(np.diff(lambdas) >= 0):
        raise ValueError("lambda grid must be non-decreasing")

    n = len(loss_fn(float(lambdas[0])))
    if n == 0:
        raise ValueError("loss function returned no examples")

    for lam in lambdas:
        risk = float(np.mean(loss_fn(float(lam))))
        corrected = (n / (n + 1)) * risk + B / (n + 1)
        if corrected <= alpha:
            return float(lam)
    return float(lambdas[-1])


def crc_is_feasible(
    loss_fn: LossFn, lambdas: np.ndarray, alpha: float, B: float = 1.0
) -> bool:
    """Whether ANY lambda on the grid achieves the target risk.

    Infeasibility is usually one of two things: the calibration set is too
    small for B/(n+1) to fit under alpha, or the score simply does not separate
    correct from incorrect steps well enough. Both are results worth reporting,
    not bugs to paper over.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    n = len(loss_fn(float(lambdas[0])))
    lam = float(lambdas[-1])
    risk = float(np.mean(loss_fn(lam)))
    return (n / (n + 1)) * risk + B / (n + 1) <= alpha


def false_safe_loss(scores: np.ndarray, labels: np.ndarray) -> LossFn:
    """Loss = 1 when a WRONG step is accepted by the gate.

    `labels` is True for a correct step. Acceptance is `score <= -lambda`, so
    growing lambda tightens the gate and the loss is non-increasing, as CRC
    requires.
    """
    s = np.asarray(scores, dtype=float)
    correct = np.asarray(labels, dtype=bool)
    if s.shape != correct.shape:
        raise ValueError("scores and labels must have the same shape")

    def loss(lam: float) -> np.ndarray:
        accepted = s <= -lam
        return (accepted & ~correct).astype(float)

    return loss


class RiskControlCalibrator:
    """Threshold chosen to bound the expected false-safe rate at alpha."""

    def __init__(self, alpha: float = 0.1, grid_size: int = 400) -> None:
        self.alpha = alpha
        self.grid_size = grid_size
        self._threshold: float | None = None
        self.feasible: bool | None = None
        self.n_calibration = 0

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrator used before fit()")
        return self._threshold

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "RiskControlCalibrator":
        s = np.asarray(scores, dtype=float)
        if labels is None:
            raise ValueError("RiskControlCalibrator requires step-correctness labels")
        self.n_calibration = int(s.size)

        # Grid over negated thresholds so the loss is non-increasing in lambda.
        lo, hi = float(s.min()), float(s.max())
        pad = 0.05 * (hi - lo + 1e-9)
        lambdas = np.linspace(-(hi + pad), -(lo - pad), self.grid_size)

        loss = false_safe_loss(s, np.asarray(labels, dtype=bool))
        self.feasible = crc_is_feasible(loss, lambdas, self.alpha)
        lam = crc_lambda(loss, lambdas, self.alpha)
        self._threshold = -lam
        return self

    def update(self, observed_error: int, was_exploration: bool = False) -> None:
        """No-op: CRC is a static procedure. See `adaptive.py` for the online case."""
        return None
