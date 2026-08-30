"""Baseline verification policies.

Each baseline is a drop-in calibrator, so every condition runs through the same
agent loop with one component swapped. That is deliberate: parallel
implementations drift, and a Pareto plot comparing two subtly different loops
is not a comparison.

The two baselines people skip, and shouldn't:

* RandomGate at matched budget. Surprisingly strong. If CAR cannot beat random
  allocation at the same number of calls, the uncertainty signal is doing
  nothing and no amount of calibration will rescue it.
* OracleGate. The upper bound. It separates "our gate is good" from "this task
  was easy", and every accuracy-vs-cost figure should carry it as a ceiling.
"""

from __future__ import annotations

import numpy as np


class FixedThresholdCalibrator:
    """Constant threshold. Base for the non-adaptive baselines."""

    def __init__(self, threshold: float) -> None:
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    def fit(self, scores=None, labels=None):
        return self

    def update(self, observed_error, **kwargs) -> None:
        return None


class NeverVerify(FixedThresholdCalibrator):
    """Plain chain-of-thought. The lower-bound control."""

    def __init__(self) -> None:
        super().__init__(float("inf"))


class AlwaysVerify(FixedThresholdCalibrator):
    """Verify every step. Reliability ceiling at maximum cost."""

    def __init__(self) -> None:
        super().__init__(float("-inf"))


class QuantileGate:
    """Heuristic threshold at a raw score quantile -- no conformal correction.

    This is the "entropy gate" / "semantic gate" family. It isolates what the
    conformal machinery contributes: same score, same loop, threshold picked by
    eyeballing a quantile instead of by calibration.
    """

    def __init__(self, quantile: float = 0.8) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrator used before fit()")
        return self._threshold

    def fit(self, scores, labels=None) -> "QuantileGate":
        self._threshold = float(np.quantile(np.asarray(scores, dtype=float), self.quantile))
        return self

    def update(self, observed_error, **kwargs) -> None:
        return None


class RandomGate:
    """Verify with fixed probability, ignoring the score entirely.

    Implemented by returning a random threshold each time it is read, so a
    `score > threshold` comparison fires with probability `rate` regardless of
    the score. Keeps the gate code path identical across conditions.
    """

    def __init__(self, rate: float = 0.3, seed: int = 0) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate
        self._rng = np.random.default_rng(seed)

    @property
    def threshold(self) -> float:
        return float("-inf") if self._rng.random() < self.rate else float("inf")

    def fit(self, scores=None, labels=None) -> "RandomGate":
        return self

    def update(self, observed_error, **kwargs) -> None:
        return None


class OracleGate:
    """Verifies exactly the steps that are actually wrong.

    Not deployable -- it reads ground-truth labels. It defines the best any
    gating policy could do at a given budget, which is the reference the whole
    accuracy-cost argument is measured against.
    """

    def __init__(self, labels_by_key: dict[str, bool]) -> None:
        self.labels_by_key = labels_by_key
        self._current_key: str | None = None

    def set_key(self, key: str) -> None:
        self._current_key = key

    @property
    def threshold(self) -> float:
        if self._current_key is None:
            return float("inf")
        correct = self.labels_by_key.get(self._current_key, True)
        return float("inf") if correct else float("-inf")

    def fit(self, scores=None, labels=None) -> "OracleGate":
        return self

    def update(self, observed_error, **kwargs) -> None:
        return None


BASELINES = {
    "cot": NeverVerify,
    "always_verify": AlwaysVerify,
    "quantile_gate": QuantileGate,
    "random_gate": RandomGate,
}


def load_baseline(name: str, **kwargs):
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name!r} (have {sorted(BASELINES)})")
    return BASELINES[name](**kwargs)
