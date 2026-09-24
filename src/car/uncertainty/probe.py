"""A trained probe on frozen internal states, as a step-level uncertainty score.

Chapter 7's negative result rests on AUROC 0.5742: neither token-level
uncertainty nor sampling-based semantic divergence ranks a globally-wrong step.
Chapter 9 names the obvious untested alternative -- ReProbe (Ni et al.) trains a
sub-10M-parameter probe on a frozen model's internal states and matches PRMs up
to 810x larger -- and makes a falsifiable prediction: on GSM8K, where the
strongest PRMs reach parity with the probe, it should land near the ch. 5 PRM's
0.9033 rather than above it.

This is the machinery for testing that.

SELECTION DISCIPLINE, WHICH IS THE WHOLE DIFFICULTY

There are 29 candidate layers and a regularisation strength to choose, on ~900
dev steps with 3,584 features each. Picking the layer by test AUROC would
manufacture a result out of that search space, and it is the single easiest way
to get a wrong answer here.

So the split usage is deliberate and asymmetric:

    dev-train    fit the probe
    dev-select   choose the layer and C
    calibration  LEFT ALONE -- it belongs to the conformal threshold, and a
                 probe selected on it would leak into the guarantee
    test         reported once, never consulted

`select_and_fit` enforces this by construction: it never sees the test set.

THE HONEST LIMIT

ReProbe trains on far more data than ~900 steps. A null result here is
therefore weaker evidence than a null with proper training data would be, and
`learning_curve` exists so a reader can see whether the probe is saturating or
merely starved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Wide on purpose: 3,584 features on ~900 examples needs heavy regularisation,
# and where the optimum lands is itself informative.
DEFAULT_CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass
class ProbeResult:
    layer: int
    C: float
    auroc_select: float
    model: object = None
    scaler: object = None
    per_layer: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        """P(step is globally WRONG). Oriented like every other score here:
        higher means less trustworthy."""
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self.model.predict_proba(Xs)[:, 1]


def fit_one(X_train, y_train, C):
    """A single logistic probe. `y` is True for a globally-WRONG step."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.asarray(X_train, dtype=np.float64))
    model = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced", solver="lbfgs"
    )
    model.fit(scaler.transform(np.asarray(X_train, dtype=np.float64)), y_train)
    return model, scaler


def auroc(model, scaler, X, y) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=bool)
    if len(np.unique(y)) < 2:
        return float("nan")
    p = model.predict_proba(scaler.transform(np.asarray(X, dtype=np.float64)))[:, 1]
    return float(roc_auc_score(y, p))


def select_and_fit(
    states_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    select_idx: np.ndarray,
    Cs=DEFAULT_CS,
) -> ProbeResult:
    """Choose a layer and C on `select_idx`, never on test.

    `states_by_layer` maps layer index to an (n_steps, hidden) array covering
    ALL steps; the two index arrays carve out the two roles. The test set is
    not a parameter of this function on purpose -- it cannot be consulted even
    by accident.
    """
    y = np.asarray(y, dtype=bool)
    best = None
    per_layer = {}

    for layer, X in sorted(states_by_layer.items()):
        X = np.asarray(X)
        layer_best = None
        for C in Cs:
            model, scaler = fit_one(X[train_idx], y[train_idx], C)
            a = auroc(model, scaler, X[select_idx], y[select_idx])
            if a != a:
                continue
            cand = ProbeResult(layer=layer, C=C, auroc_select=a,
                               model=model, scaler=scaler)
            if layer_best is None or a > layer_best.auroc_select:
                layer_best = cand
        if layer_best is None:
            continue
        per_layer[layer] = layer_best.auroc_select
        if best is None or layer_best.auroc_select > best.auroc_select:
            best = layer_best

    if best is None:
        raise ValueError("no layer produced a usable probe")
    best.per_layer = per_layer
    return best


def learning_curve(X, y, train_idx, eval_idx, C, fractions=(0.25, 0.5, 0.75, 1.0),
                   seed=0) -> list[tuple[int, float]]:
    """AUROC against training-set size, so "it did not work" can be told apart
    from "it did not have enough data"."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(train_idx))
    y = np.asarray(y, dtype=bool)
    out = []
    for f in fractions:
        k = max(20, int(len(order) * f))
        sub = order[:k]
        if len(np.unique(y[sub])) < 2:
            continue
        model, scaler = fit_one(np.asarray(X)[sub], y[sub], C)
        out.append((k, auroc(model, scaler, np.asarray(X)[eval_idx], y[eval_idx])))
    return out
