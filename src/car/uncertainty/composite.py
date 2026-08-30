"""Composite uncertainty score: fuse the signals into one scalar.

Two rules this module enforces, both of which are easy to get wrong and fatal
to the results if you do:

1. Orientation. Every feature must point the same way -- higher means more
   uncertain. `mean_logprob` is naturally the other way round, so it is
   negated here, once, rather than at each call site.

2. Standardisation is fit on DEV data only. Feature scales differ by orders of
   magnitude, so the score is unusable without it, but fitting the scaler on
   calibration or test data leaks information into the threshold and silently
   invalidates the conformal guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from car.types import UncertaintyFeatures

FEATURE_KEYS = [
    "token_entropy",
    "max_surprisal",
    "mean_logprob",
    "semantic_divergence",
    "task_verifier_signal",
]

# Features where a LOWER raw value means MORE uncertainty.
_INVERTED = {"mean_logprob"}


def _oriented(feats: UncertaintyFeatures, keys: list[str]) -> np.ndarray:
    vals = []
    for k in keys:
        v = getattr(feats, k)
        vals.append(-v if k in _INVERTED else v)
    return np.asarray(vals, dtype=float)


class CompositeScorer:
    """Weighted, standardised combination of uncertainty features.

    Parameters
    ----------
    weights:
        One per key in `keys`. Setting a weight to 0.0 is how ablations are
        run ("token-only" = zero the semantic weight), which keeps the code
        path identical across conditions.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        keys: list[str] | None = None,
    ) -> None:
        self.keys = keys or list(FEATURE_KEYS)
        w = weights or dict.fromkeys(self.keys, 1.0)
        self.weights = np.asarray([w.get(k, 0.0) for k in self.keys], dtype=float)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, dev_features: list[UncertaintyFeatures]) -> CompositeScorer:
        """Fit the standardiser. DEV DATA ONLY -- see module docstring."""
        if not dev_features:
            raise ValueError("cannot fit CompositeScorer on an empty dev set")
        X = np.vstack([_oriented(f, self.keys) for f in dev_features])
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature carries no information; neutralise it rather than
        # dividing by ~0 and manufacturing a huge spurious score.
        self._std = np.where(std < 1e-8, 1.0, std)
        return self

    def score(self, feats: UncertaintyFeatures) -> float:
        """Scalar uncertainty. Higher means less trustworthy."""
        x = _oriented(feats, self.keys)
        if self._mean is not None:
            x = (x - self._mean) / self._std
        return float(np.dot(self.weights, x))

    def score_many(self, feats: list[UncertaintyFeatures]) -> np.ndarray:
        return np.asarray([self.score(f) for f in feats])

    # ---- persistence -------------------------------------------------
    # Weights and scaler are experiment artifacts. They get hashed into the
    # run manifest so a results table can always be traced to the exact
    # scorer that produced it.

    def to_dict(self) -> dict:
        return {
            "keys": self.keys,
            "weights": self.weights.tolist(),
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompositeScorer:
        obj = cls(weights=dict(zip(d["keys"], d["weights"], strict=True)), keys=d["keys"])
        if d.get("mean") is not None:
            obj._mean = np.asarray(d["mean"], dtype=float)
            obj._std = np.asarray(d["std"], dtype=float)
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CompositeScorer:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_weights_logistic(
    dev_features: list[UncertaintyFeatures],
    dev_labels: list[bool],
    keys: list[str] | None = None,
) -> CompositeScorer:
    """Fit feature weights by logistic regression on step correctness.

    Learns which signals actually predict a wrong step instead of assuming
    equal weights. Labels are `True` for a CORRECT step; we fit against
    incorrectness so positive weights mean "pushes the score up when wrong".

    Uses dev data only, and returns a scorer already standardised on that
    same dev set.
    """
    from sklearn.linear_model import LogisticRegression

    keys = keys or list(FEATURE_KEYS)
    if len(dev_features) != len(dev_labels):
        raise ValueError("features and labels must be the same length")

    scaler = CompositeScorer(keys=keys).fit(dev_features)
    X = np.vstack([_oriented(f, keys) for f in dev_features])
    X = (X - scaler._mean) / scaler._std
    y = np.asarray([0 if lab else 1 for lab in dev_labels])  # 1 = incorrect step

    if len(np.unique(y)) < 2:
        # Degenerate dev set (all steps correct or all wrong). Fall back to
        # uniform weights rather than raising -- this happens on small smoke
        # runs and should not crash the pipeline.
        return scaler

    clf = LogisticRegression(max_iter=1000).fit(X, y)
    scorer = CompositeScorer(
        weights=dict(zip(keys, clf.coef_[0].tolist(), strict=True)), keys=keys
    )
    scorer._mean, scorer._std = scaler._mean, scaler._std
    return scorer
