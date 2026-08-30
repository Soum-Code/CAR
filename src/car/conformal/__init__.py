"""Calibration layer: turning an uncertainty score into a defensible threshold."""

from typing import Protocol, runtime_checkable

import numpy as np

from car.conformal.adaptive import AdaptiveCalibrator, ThresholdTrace, UpdateMode
from car.conformal.risk_control import (
    RiskControlCalibrator,
    crc_is_feasible,
    crc_lambda,
    false_safe_loss,
)
from car.conformal.split import (
    SplitConformalCalibrator,
    conformal_quantile,
    min_calibration_size,
)


@runtime_checkable
class Calibrator(Protocol):
    """Common surface so static and adaptive calibration are swappable in configs."""

    @property
    def threshold(self) -> float: ...

    def fit(self, scores: np.ndarray, labels: np.ndarray | None = None): ...

    def update(self, observed_error: int, **kwargs) -> None: ...


def load_calibrator(kind: str, **kwargs):
    """Factory keyed on config string. Enables the static-vs-adaptive ablation."""
    if kind == "split":
        return SplitConformalCalibrator(**kwargs)
    if kind in ("crc", "risk_control"):
        return RiskControlCalibrator(**kwargs)
    if kind == "adaptive":
        return AdaptiveCalibrator(**kwargs)
    raise ValueError(f"unknown calibrator: {kind!r} (expected split, crc, or adaptive)")


__all__ = [
    "AdaptiveCalibrator",
    "Calibrator",
    "RiskControlCalibrator",
    "SplitConformalCalibrator",
    "ThresholdTrace",
    "UpdateMode",
    "conformal_quantile",
    "crc_is_feasible",
    "crc_lambda",
    "false_safe_loss",
    "load_calibrator",
    "min_calibration_size",
]
