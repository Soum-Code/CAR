"""Evaluation metrics and analysis."""

from car.eval.inference import (
    Interval,
    auroc_ci,
    auroc_resolution_ceiling,
    bootstrap_p_value,
    design_effect,
    minimum_detectable_delta,
    paired_auroc_delta_ci,
)
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
    "Interval",
    "accuracy_per_tool_call",
    "auroc_ci",
    "auroc_resolution_ceiling",
    "bootstrap_p_value",
    "design_effect",
    "minimum_detectable_delta",
    "paired_auroc_delta_ci",
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
