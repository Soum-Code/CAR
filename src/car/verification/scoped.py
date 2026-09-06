"""A verifier whose reach is a measured parameter.

Chapter 5 measured *scope* -- the probability a verifier detects a step that is
globally wrong -- for four verifier classes on the arithmetic-blind population:

    arithmetic, step-local        0.0000
    arithmetic, unbounded lookback 0.1999
    independent judge (Qwen2.5-7B) 0.2283  at 2.0% false alarm
    task PRM (Math-Shepherd-7B)    0.9033  at 9.9% false alarm

Chapter 6 asks what follows for allocation. Connecting them needs scope to be a
knob rather than a property of whichever verifier happens to be wired in, which
is what this class is.

The modelling decision that matters
-----------------------------------
A missed detection returns SUPPORTED, not INSUFFICIENT. A verifier that fails
to see an error does not usually announce that it failed -- it says the step
looks fine. That distinction is not cosmetic: INSUFFICIENT yields no label and
the calibrator learns nothing, whereas SUPPORTED yields a label of "no error"
for a step that has one.

So a low-scope verifier does not merely help less. It feeds the calibrator
systematically wrong labels, and the calibrator converges on a threshold that
is confident about a risk it cannot see. That consequence is a prediction of
C3, and this component is what lets the pipeline show it rather than assert it.
"""

from __future__ import annotations

import numpy as np

from car.types import ReasoningStep, Verdict
from car.verification.base import VerificationResult

# Measured in ch. 5; see runs/semantic_scope_*.json and README C3.
MEASURED_SCOPE = {
    "arithmetic_local": (0.0000, 0.0000),
    "arithmetic_lookback": (0.1999, 0.0000),
    "same_model_critic": (0.0000, 0.0000),
    "independent_judge": (0.2283, 0.0200),
    "task_prm": (0.9033, 0.0987),
}


class ScopedVerifier:
    """Detects a globally-wrong step with probability `scope`.

    Parameters
    ----------
    labels:
        (question, step_id) -> global correctness. Ground truth, read by the
        simulator only; the online policy never sees it.
    scope:
        P(detect | step is globally wrong).
    false_alarm:
        P(flag | step is globally correct).
    """

    def __init__(
        self,
        labels: dict[tuple[str, int], bool],
        *,
        scope: float,
        false_alarm: float = 0.0,
        label: str = "scoped",
        seed: int = 0,
    ) -> None:
        self.labels = labels
        self.scope = float(np.clip(scope, 0.0, 1.0))
        self.false_alarm = float(np.clip(false_alarm, 0.0, 1.0))
        self.label = label
        self._rng = np.random.default_rng(seed)
        self.calls = 0
        self.detections = 0
        self.false_alarms = 0
        self.misses = 0

    @classmethod
    def from_measured(cls, labels, kind: str, **kwargs) -> "ScopedVerifier":
        if kind not in MEASURED_SCOPE:
            raise ValueError(
                f"unknown verifier class {kind!r}; have {sorted(MEASURED_SCOPE)}"
            )
        scope, fa = MEASURED_SCOPE[kind]
        return cls(labels, scope=scope, false_alarm=fa, label=kind, **kwargs)

    @property
    def name(self) -> str:
        return f"{self.label}(scope={self.scope:.4f},fa={self.false_alarm:.4f})"

    def verify(self, step: ReasoningStep, question: str) -> VerificationResult:
        self.calls += 1
        correct = self.labels.get((question, step.step_id))
        if correct is None:
            return VerificationResult(Verdict.INSUFFICIENT, detail="no label")

        if not correct:
            if self._rng.random() < self.scope:
                self.detections += 1
                return VerificationResult(
                    Verdict.CONTRADICTED,
                    detail=f"{self.label}: detected",
                    revised_claim=f"{step.claim} (repaired)",
                )
            # Missed. Says it looks fine, which is what a real verifier does.
            self.misses += 1
            return VerificationResult(Verdict.SUPPORTED, detail=f"{self.label}: missed")

        if self._rng.random() < self.false_alarm:
            self.false_alarms += 1
            return VerificationResult(
                Verdict.CONTRADICTED,
                detail=f"{self.label}: false alarm",
                revised_claim=f"{step.claim} (spuriously revised)",
            )
        return VerificationResult(Verdict.SUPPORTED, detail=f"{self.label}: sound")

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "detections": self.detections,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "observed_detection_rate": self.detections / max(1, self.detections + self.misses),
        }
