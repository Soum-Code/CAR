"""Evaluation metrics and analysis."""

from car.eval.metrics import (
    accuracy_per_tool_call,
    cost_per_question,
    empirical_coverage,
    expected_calibration_error,
    false_safe_rate,
    final_accuracy,
    propagation_depth,
    selective_risk,
    step_detection_auprc,
    step_detection_auroc,
    summarise,
    verification_rate,
)

__all__ = [
    "accuracy_per_tool_call",
    "cost_per_question",
    "empirical_coverage",
    "expected_calibration_error",
    "false_safe_rate",
    "final_accuracy",
    "propagation_depth",
    "selective_risk",
    "step_detection_auprc",
    "step_detection_auroc",
    "summarise",
    "verification_rate",
]
