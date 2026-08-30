"""Simulated verifier for CPU development.

Needed because the deterministic verifiers cannot say anything about mock steps
-- `CalculatorVerifier` finds no arithmetic in them and correctly returns
INSUFFICIENT, so the adaptive calibrator never receives a label and the control
loop never closes. Without this, the full CAR loop is not exercisable without a
GPU and a real dataset.

Reads the known latent correctness from `MockBackend`, with a configurable
error rate so that verifier fallibility can itself be studied -- a perfect
verifier is an unrealistic assumption and the ablation table should say what
happens when the checker is wrong.
"""

from __future__ import annotations

import numpy as np

from car.types import ReasoningStep, Verdict
from car.verification.base import VerificationResult


class SimulatedVerifier:
    """Consults the simulator's ground truth, optionally with noise.

    Parameters
    ----------
    error_rate:
        Probability the verifier returns the wrong verdict. 0.0 is an oracle;
        raise it to model an imperfect retriever or a fallible LLM judge.
    insufficient_rate:
        Probability the verifier finds no usable evidence and abstains. Matters
        because an INSUFFICIENT verdict yields NO label -- the calibrator must
        not treat "I could not check" as "it was fine".
    """

    def __init__(
        self,
        backend,
        *,
        error_rate: float = 0.0,
        insufficient_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.error_rate = float(np.clip(error_rate, 0.0, 1.0))
        self.insufficient_rate = float(np.clip(insufficient_rate, 0.0, 1.0))
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    @property
    def name(self) -> str:
        return f"simulated(err={self.error_rate},insuf={self.insufficient_rate})"

    def verify(self, step: ReasoningStep, question: str) -> VerificationResult:
        self.calls += 1

        if self._rng.random() < self.insufficient_rate:
            return VerificationResult(Verdict.INSUFFICIENT, detail="no evidence found")

        # The generator keys latent correctness on "<example_id>::<target>";
        # the step claim is written as "[example_id] target" by
        # MockStepGenerator, so recover the key from it.
        key = self._key_from_claim(step.claim)
        correct, _ = self.backend.latent_correctness(key)

        if self._rng.random() < self.error_rate:
            correct = not correct  # verifier itself errs

        return VerificationResult(
            Verdict.SUPPORTED if correct else Verdict.CONTRADICTED,
            detail="simulated ground truth",
            revised_claim=None if correct else f"{step.claim} (revised)",
        )

    @staticmethod
    def _key_from_claim(claim: str) -> str:
        # "[q3] target_1" -> "q3::target_1"
        if claim.startswith("[") and "] " in claim:
            example_id, target = claim[1:].split("] ", 1)
            return f"{example_id}::{target}"
        return claim
