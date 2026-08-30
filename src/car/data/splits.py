"""Dev / calibration / test partitioning.

The split discipline is the difference between a conformal guarantee and a
number that looks like one:

  dev          prompt design, feature engineering, weight fitting, all
               engineering choices
  calibration  threshold estimation ONLY
  test         never touched for threshold selection, weight tuning, or prompt
               selection

Splitting is by a hash of the example id, not by a shuffled index. That makes
assignment stable when the dataset grows or is reordered -- an example that was
in test stays in test, so a later run cannot quietly promote a test item into
calibration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from car.types import Example


@dataclass
class Splits:
    dev: list[Example]
    calibration: list[Example]
    test: list[Example]

    def summary(self) -> dict[str, int]:
        return {
            "dev": len(self.dev),
            "calibration": len(self.calibration),
            "test": len(self.test),
        }


def _bucket(example_id: str, salt: str = "car-v1") -> float:
    """Stable pseudo-random number in [0, 1) derived from the example id."""
    digest = hashlib.sha256(f"{salt}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def make_splits(
    examples: list[Example],
    *,
    dev_frac: float = 0.3,
    cal_frac: float = 0.3,
    salt: str = "car-v1",
) -> Splits:
    """Partition deterministically by example id."""
    if dev_frac + cal_frac >= 1.0:
        raise ValueError("dev_frac + cal_frac must leave room for a test set")

    dev, cal, test = [], [], []
    for ex in examples:
        b = _bucket(ex.example_id, salt)
        if b < dev_frac:
            dev.append(ex)
        elif b < dev_frac + cal_frac:
            cal.append(ex)
        else:
            test.append(ex)
    return Splits(dev=dev, calibration=cal, test=test)


def assert_no_leakage(splits: Splits) -> None:
    """Fail loudly if any example appears in more than one split."""
    ids = {
        "dev": {e.example_id for e in splits.dev},
        "calibration": {e.example_id for e in splits.calibration},
        "test": {e.example_id for e in splits.test},
    }
    for a, b in (("dev", "calibration"), ("dev", "test"), ("calibration", "test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise AssertionError(
                f"leakage between {a} and {b}: {sorted(overlap)[:5]}"
            )


def check_calibration_size(splits: Splits, alpha: float, steps_per_example: int = 4) -> None:
    """Warn when the calibration set is too small to certify the target alpha.

    Split conformal cannot deliver a 1-alpha guarantee below roughly 1/alpha
    calibration points. Hitting this silently produces a threshold that looks
    valid and is not.
    """
    from car.conformal import min_calibration_size

    n = len(splits.calibration) * steps_per_example
    need = min_calibration_size(alpha)
    if n < need:
        raise ValueError(
            f"calibration set has ~{n} steps but alpha={alpha} needs at least "
            f"{need}; either lower alpha or enlarge the calibration split"
        )
