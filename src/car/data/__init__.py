"""Datasets and split discipline.

GSM8K is the PRIMARY benchmark for the propagation and allocation claims.
StrategyQA is retained for calibration and evidence-grounded verification.

The switch is evidence-based, not preference: StrategyQA graphs are 72.9%
one hop deep and only 11.2% of their steps have any non-terminal descendant,
so error propagation barely occurs in it. GSM8K has 2.4x the headroom, a
depth tail reaching 8, and 5x as many questions at depth >= 3. See
docs/FINDINGS-PROPAGATION.md.
"""

from car.data import gsm8k, strategyqa
from car.data.splits import (
    Splits,
    assert_no_leakage,
    check_calibration_size,
    make_splits,
)

# Registry so configs can name a dataset as a string.
DATASETS = {
    "gsm8k": gsm8k,
    "strategyqa": strategyqa,
}

PRIMARY = "gsm8k"


def load_examples(dataset: str, path):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r} (have {sorted(DATASETS)})")
    return DATASETS[dataset].load_examples(path)


def load_dags(dataset: str, path):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r} (have {sorted(DATASETS)})")
    return DATASETS[dataset].load_dags(path)


__all__ = [
    "DATASETS",
    "PRIMARY",
    "Splits",
    "assert_no_leakage",
    "check_calibration_size",
    "gsm8k",
    "load_dags",
    "load_examples",
    "make_splits",
    "strategyqa",
]
