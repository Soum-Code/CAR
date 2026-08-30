"""Evaluation metrics.

Definitions are written out explicitly because several of these names get used
loosely in the literature, and the difference between them is the difference
between a defensible claim and an overclaim.
"""

from __future__ import annotations

import numpy as np

from car.types import Decision, Trajectory


def step_detection_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of the uncertainty score for ranking wrong steps above correct ones.

    `labels` is True for a CORRECT step, so the positive class for detection is
    `~labels`. 0.5 means the score carries no information about correctness --
    which, if that is what the data says, is itself a publishable finding.
    """
    from sklearn.metrics import roc_auc_score

    y = ~np.asarray(labels, dtype=bool)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def step_detection_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = ~np.asarray(labels, dtype=bool)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores, dtype=float)))


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Bin-weighted gap between stated confidence and observed accuracy.

    Note this measures PROBABILITY calibration, which is a different property
    from conformal coverage. A model can be badly calibrated in the ECE sense
    while a conformal wrapper around it still attains its coverage target.
    """
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=bool)
    if conf.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Include the left edge on the first bin so conf == 0 is not dropped.
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(acc[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def selective_risk(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Error rate among ACCEPTED steps: P(wrong | score <= threshold).

    This is CAR's primary reliability target. It is a CONDITIONAL quantity,
    which is why plain split conformal (a marginal coverage tool) is not the
    right instrument for controlling it -- see conformal/risk_control.py.
    """
    s = np.asarray(scores, dtype=float)
    correct = np.asarray(labels, dtype=bool)
    accepted = s <= threshold
    if not accepted.any():
        return float("nan")
    return float((~correct[accepted]).mean())


def false_safe_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Wrong steps the gate let through, as a fraction of all wrong steps.

    The safety-facing view. Selective risk asks "how dirty is what I accepted";
    this asks "how much of the dirt did I miss". They move differently as the
    threshold changes and both belong in the results table.
    """
    s = np.asarray(scores, dtype=float)
    correct = np.asarray(labels, dtype=bool)
    wrong = ~correct
    if not wrong.any():
        return float("nan")
    return float((wrong & (s <= threshold)).sum() / wrong.sum())


def empirical_coverage(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Fraction of CORRECT steps that fall inside the acceptance region.

    This is what the conformal target 1 - alpha refers to. It is not accuracy,
    and reporting it as accuracy is the single most common misreading of a
    conformal result.
    """
    s = np.asarray(scores, dtype=float)
    correct = np.asarray(labels, dtype=bool)
    if not correct.any():
        return float("nan")
    return float((s[correct] <= threshold).mean())


def verification_rate(trajectories: list[Trajectory]) -> float:
    """Fraction of steps that triggered a verification call."""
    total = sum(t.n_steps for t in trajectories)
    if total == 0:
        return float("nan")
    verified = sum(
        1 for t in trajectories for r in t.steps if r.decision == Decision.VERIFY
    )
    return verified / total


def final_accuracy(trajectories: list[Trajectory]) -> float:
    scored = [t for t in trajectories if t.correct is not None]
    if not scored:
        return float("nan")
    return float(np.mean([t.correct for t in scored]))


def cost_per_question(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        return float("nan")
    return float(np.mean([t.verification_calls for t in trajectories]))


def accuracy_per_tool_call(trajectories: list[Trajectory], baseline_accuracy: float) -> float:
    """Accuracy gained per verification call, relative to a no-verification baseline.

    The efficiency number the Pareto argument rests on. Returns nan when no
    calls were made, rather than dividing by zero and reporting infinite
    efficiency.
    """
    cost = cost_per_question(trajectories)
    if not cost or np.isnan(cost) or cost <= 0:
        return float("nan")
    return float((final_accuracy(trajectories) - baseline_accuracy) / cost)


def propagation_depth(trajectory: Trajectory) -> int:
    """How many steps followed the first accepted-but-wrong step.

    The direct measure of error propagation: an early accepted mistake with
    many steps after it is the failure CAR is built to prevent.
    """
    for i, r in enumerate(trajectory.steps):
        if r.label is False and r.decision == Decision.CONTINUE:
            return trajectory.n_steps - i - 1
    return 0


def summarise(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    trajectories: list[Trajectory] | None = None,
) -> dict[str, float]:
    """One row of the results table."""
    out = {
        "auroc": step_detection_auroc(scores, labels),
        "auprc": step_detection_auprc(scores, labels),
        "selective_risk": selective_risk(scores, labels, threshold),
        "false_safe_rate": false_safe_rate(scores, labels, threshold),
        "empirical_coverage": empirical_coverage(scores, labels, threshold),
        "threshold": float(threshold),
        "n_steps": int(len(scores)),
    }
    if trajectories:
        out |= {
            "final_accuracy": final_accuracy(trajectories),
            "verification_rate": verification_rate(trajectories),
            "cost_per_question": cost_per_question(trajectories),
        }
    return out
