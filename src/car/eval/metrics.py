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


def project_correct_after_repair(traj: Trajectory) -> bool | None:
    """MODELLED final-answer correctness after repair. NOT a measurement.

    Needed for replayed corpora, where the text is fixed and `traj.correct` is
    therefore identical under every gating policy -- reporting that as accuracy
    would say the gate does nothing, which is an artifact of replay rather than
    a result. This applies the propagation model's assumption explicitly:

      * a wrong answer is rescued iff the FIRST globally-bad step was detected
        and repaired, because everything downstream inherits the corruption;
      * a right answer is lost if a false alarm "repaired" a correct step.

    Both directions are modelled, because a verifier with high scope and a high
    false-alarm rate can lose more than it saves -- exactly the trade-off the
    measured scope numbers exist to inform.

    An answer that was already correct stays correct even when the trajectory
    contains a `-` step: Math-Shepherd's labels are optimistic Monte-Carlo
    estimates of "leads to a correct answer", so a solution can carry a bad
    step and still land on the right number. With scope 0 and no false alarms
    this reduces exactly to the observed accuracy, which is the check that it
    is not fabricating movement.
    """
    if traj.correct is None:
        return None
    labelled = [r for r in traj.steps if r.label is not None]
    if not labelled:
        return traj.correct

    false_repair = any(r.label is True and r.revised for r in labelled)
    if traj.correct:
        return not false_repair

    first_bad = next((r for r in labelled if r.label is False), None)
    if first_bad is None:
        return False
    return bool(first_bad.revised) and not false_repair


def projected_accuracy(trajectories: list[Trajectory]) -> float:
    """Mean of `project_correct_after_repair`. Label it MODELLED when reporting."""
    vals = [p for p in (project_correct_after_repair(t) for t in trajectories)
            if p is not None]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


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
